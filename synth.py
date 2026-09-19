from dataclasses import dataclass
import random
import time

from datasets import Dataset
from simple_parsing import parse
from transformers import set_seed

from shared.load import load_llm, load_longhealth
from shared.modules import generate
from shared.prompts import make_id, make_id_inp, make_msg_sys, make_msg_user, make_txt_inst, make_txt_cot, make_txt_sys
from shared.utils import print_args


@dataclass
class Args:
    name_dataset: str
    name_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    n_batch: int = 10  # authors use 32
    i_start: int = 0
    i_end: int = 1000  # authors use 2048
    n_save: int = 25
    n_tok_que: int = 512
    n_tok_ans: int = 1024
    temp_que: float = 0.6
    temp_ans: float = 0.0
    p_cot: float = 0.2
    seed: int = 42  # authors use 82


def main() -> None:
    # kv cache is the context of the corpus that the model understands.
    # training it with simple next token prediction only teaches it to copy the corpus, which is memorizing, not understanding.
    # so make convos (question + answer) written by model that sees context, and train kv cache to act as the context,
    # so model answers various questions well without seeing the corpus.
    # 1. pick random context and seed instructions. (same as authors)
    # 2. generate questions from context and seed instructions.
    # 3. generate answers from context and questions, sometimes adding cot instruction.
    # 4. save context, questions, cot instructions and answers as rows.
    args = parse(Args)
    print_args(args)
    model, tokenizer = load_llm(args.name_model)
    type_model = model.config.model_type
    _, txts_ctx = load_longhealth()
    rows = []
    print(
        f"{'i_sample':<11}"
        f"{'dt_sample':<11}"
        f"{'n_tok_sys':<11}"
    )  # fmt: skip
    for i_sample in range(args.i_start, args.i_end):
        t_start = time.time()
        # seed per sample so ranges can run in parallel.
        set_seed(args.seed + i_sample)
        txt_ctx = random.choice(random.choice(txts_ctx))
        msg_sys = make_msg_sys(make_txt_sys(txt_ctx))
        id_sys = make_id(tokenizer, [msg_sys], type_model, model.device)
        n_tok_sys = id_sys.shape[1]
        msgs_inp_que = []
        for i_batch in range(args.n_batch):
            txt_inst = make_txt_inst()
            msg_inst = make_msg_user(txt_inst)
            msgs_inp_que.append([msg_sys, msg_inst])
        id_inp_que, mask_inp_que = make_id_inp(tokenizer, msgs_inp_que, type_model, model.device)
        id_gen_que, _ = generate(model, tokenizer, id_inp_que, mask_inp_que, args.n_tok_que, args.temp_que)
        txts_que = tokenizer.batch_decode(id_gen_que, skip_special_tokens=True)
        txts_cot = []
        msgs_inp_ans = []
        for i_batch in range(args.n_batch):
            # sometimes add short cot instruction to question before generating answer.
            txt_cot = None
            txt_que_cot = txts_que[i_batch]
            if random.random() < args.p_cot:
                txt_cot = make_txt_cot()
                txt_que_cot = f"{txts_que[i_batch]}\n\n{txt_cot}"
            txts_cot.append(txt_cot)
            msg_que = make_msg_user(txt_que_cot)
            msgs_inp_ans.append([msg_sys, msg_que])
        id_inp_ans, mask_inp_ans = make_id_inp(tokenizer, msgs_inp_ans, type_model, model.device)
        id_gen_ans, mask_gen_ans = generate(model, tokenizer, id_inp_ans, mask_inp_ans, args.n_tok_ans, args.temp_ans)
        txts_ans = []
        for i_batch in range(args.n_batch):
            is_pad = mask_gen_ans[i_batch] == 0
            id_ans = id_gen_ans[i_batch, ~is_pad]
            # keep eos token to distinguish finished answer from truncated one for context distillation.
            txt_ans = tokenizer.decode(id_ans, skip_special_tokens=False)
            txts_ans.append(txt_ans)
        for i_batch in range(args.n_batch):
            row = {
                "txt_ctx": txt_ctx,
                "txt_que": txts_que[i_batch],
                "txt_cot": txts_cot[i_batch],
                "txt_ans": txts_ans[i_batch],
            }
            rows.append(row)
        dt_sample = time.time() - t_start
        print(
            f"{f'{i_sample}':<11}"
            f"{f'{dt_sample:.1f}s':<11}"
            f"{n_tok_sys:<11}"
        )  # fmt: skip
        if (i_sample + 1) % args.n_save == 0 or i_sample + 1 == args.i_end:
            dataset = Dataset.from_list(rows)
            dataset.push_to_hub(args.name_dataset, split=f"i{args.i_start:04d}_{args.i_end:04d}")
            print(f"saved {len(rows)} rows")


if __name__ == "__main__":
    main()
