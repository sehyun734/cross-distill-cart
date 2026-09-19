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
    n_batch: int = 10  # 저자는 32
    i_start: int = 0
    i_end: int = 1000  # 저자는 2048
    n_save: int = 25
    n_tok_que: int = 512
    n_tok_ans: int = 1024
    temp_que: float = 0.6
    temp_ans: float = 0.0
    p_cot: float = 0.2
    seed: int = 42  # 저자는 82


def main() -> None:
    # kv cache는 모델이 이해한 corpus의 문맥. cartridge의 목표는 corpus를 입력으로 넣지 않고
    # 학습된 작은 kv cache만으로 corpus에 대한 다양한 질문에 답할 수 있도록 하는 것.
    # 이때 다음 토큰 예측으로만 학습하면 corpus를 그대로 써 내려가는 법만 배움. 이는 암기일 뿐, 이해가 아님.
    # 따라서 corpus 일부를 본 모델이 만든 대화(question + answer)로 데이터를 만들고,
    # 학습 때는 corpus 대신 kv cache를 보고 같은 답변을 하도록 학습.
    # 1. context와 seed instruction을 랜덤으로 고름. (저자와 동일)
    # 2. context와 seed instruction으로 question을 생성.
    # 3. context와 question으로 answer를 생성. 이때 가끔 cot instruction을 추가.
    # 4. context, question, cot instruction, answer를 저장.
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
        # sample마다 seed를 따로 걸어서 구간별로 병렬로 돌려도 결과가 같도록 함.
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
            # 끝까지 생성된 answer와 토큰 상한선에 의해 중간서 잘린 answer를 구분하기 위해 eos 토큰을 남김.
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
