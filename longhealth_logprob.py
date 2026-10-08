from dataclasses import dataclass
import time

from simple_parsing import parse
import torch
from torch.nn.functional import log_softmax
from transformers import DynamicCache

from primitives.dataset import load_longhealth_corpus, load_longhealth_questions
from primitives.hub import get_pending_targets, load_cartridge, save_result
from primitives.model import load_model
from primitives.prompts import DATE_STRING, LLAMA_USER_TURN
from primitives.utils import format_duration


@dataclass
class Config:
    model_name: str = "Llama-3.2-1B-Instruct"


def main() -> None:
    config = parse(Config)
    targets = get_pending_targets(config.model_name, "longhealth", "logprob")
    corpus = load_longhealth_corpus()
    samples = load_longhealth_questions(use_cot=False)
    model, tokenizer = load_model(config.model_name)

    turns = [LLAMA_USER_TURN.format(content=question) for question, _, _ in samples]
    question_ids = [tokenizer(turn, add_special_tokens=False, return_tensors="pt")["input_ids"][0] for turn in turns]
    answer_prefix_ids = tokenizer("<answer>\n", add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    for target in targets:
        method = target.split("/")[0]
        if method == "icl":
            corpus_ids = tokenizer.apply_chat_template([{"role": "system", "content": corpus}], date_string=DATE_STRING, return_tensors="pt")["input_ids"]
            cache = DynamicCache()
            model.model(input_ids=corpus_ids.to(model.device), past_key_values=cache)
            key_value = torch.stack([torch.stack([layer.keys, layer.values]) for layer in cache.layers])
            bos = []
        elif method == "cartridge":
            sink, cartridge = load_cartridge(target, model.device)
            key_value = torch.cat([sink, cartridge.to(sink.dtype)], dim=-2)
            bos = []
        elif method == "base":
            key_value = None
            bos = [torch.tensor([tokenizer.bos_token_id])]

        corrects = []
        log_time = time.perf_counter()
        for step, ((_, options, answer), sample_question_ids) in enumerate(zip(samples, question_ids)):
            scores = []
            for option in options:
                option_ids = tokenizer(option.strip(), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                input_ids = torch.cat([*bos, sample_question_ids, answer_prefix_ids, option_ids]).unsqueeze(0).to(model.device)
                cache = DynamicCache(ddp_cache_data=key_value) if key_value is not None else None
                hidden = model.model(input_ids=input_ids, past_key_values=cache).last_hidden_state[0, -len(option_ids) - 1 : -1]
                logprobs = log_softmax(model.lm_head(hidden).float(), dim=-1)
                scores.append(logprobs.gather(-1, option_ids[:, None].to(model.device)).mean().item())
            corrects.append(options[scores.index(max(scores))].strip().lower() == answer)

            if (step + 1) % 20 == 0:
                print(f"question {step + 1}/{len(samples)} accuracy {sum(corrects) / len(corrects):.2%} time {format_duration(time.perf_counter() - log_time)}", flush=True)
                log_time = time.perf_counter()

        accuracy = sum(corrects) / len(samples)
        print(f"{target} accuracy {accuracy:.2%}", flush=True)
        result = {"accuracy": accuracy, "corrects": corrects}
        save_result("longhealth", target, "logprob", result)


if __name__ == "__main__":
    main()
