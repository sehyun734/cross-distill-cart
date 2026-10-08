from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re
import time

from simple_parsing import parse
from transformers import set_seed

from primitives.dataset import load_longhealth_corpus, load_longhealth_questions
from primitives.hub import get_pending_targets, load_cartridge, save_result
from primitives.model import generate, generate_vllm, load_model, load_vllm
from primitives.prompts import LLAMA_USER_TURN
from primitives.utils import format_duration


@dataclass
class Config:
    model_name: str = "Llama-3.2-1B-Instruct"
    batch_size: int = 32
    max_generation_len: int = 1024
    temperature: float = 0.3
    seed: int = 0


def parse_answer(
    text: str,
    options: list[str],
) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match is None:
        return None
    options = [option.strip().lower() for option in options]
    prediction = match.group(1).strip().lower()
    return max(options, key=lambda option: SequenceMatcher(None, prediction, option).ratio())


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    targets = get_pending_targets(config.model_name, "longhealth", file_name)
    samples = load_longhealth_questions()

    if f"icl/{config.model_name}" in targets:
        corpus = load_longhealth_corpus()
        model, _ = load_vllm(config.model_name, max_sequence_len=131072)
        chats = [(corpus, question) for question, _, _ in samples]
        outputs = generate_vllm(model, chats, config.max_generation_len, config.temperature, seed=config.seed)
        texts = [output.text for output in outputs]
        corrects = [parse_answer(text, options) == answer for text, (_, options, answer) in zip(texts, samples)]
        accuracy = sum(corrects) / len(samples)
        print(f"icl/{config.model_name} accuracy {accuracy:.2%}", flush=True)
        result = {"accuracy": accuracy, "corrects": corrects, "texts": texts}
        save_result("longhealth", f"icl/{config.model_name}", file_name, result)
        model.llm_engine.engine_core.shutdown()

    model, tokenizer = load_model(config.model_name)
    turns = [LLAMA_USER_TURN.format(content=question) for question, _, _ in samples]
    question_ids = [tokenizer(turn, add_special_tokens=False, return_tensors="pt")["input_ids"][0] for turn in turns]
    num_batches = math.ceil(len(samples) / config.batch_size)
    for target in targets:
        method = target.split("/")[0]
        if method == "icl":
            continue
        if method == "cartridge":
            sink, cartridge = load_cartridge(target, model.device)
        elif method == "base":
            sink = None
            cartridge = None

        set_seed(config.seed)
        texts = []
        corrects = []
        log_time = time.perf_counter()
        for step, offset in enumerate(range(0, len(samples), config.batch_size)):
            batch = samples[offset : offset + config.batch_size]
            batch_question_ids = question_ids[offset : offset + config.batch_size]
            output_ids = generate(model, tokenizer, batch_question_ids, config.max_generation_len, config.temperature, sink=sink, cartridge=cartridge)
            batch_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            texts.extend(batch_texts)
            corrects.extend(parse_answer(text, options) == answer for text, (_, options, answer) in zip(batch_texts, batch))

            print(f"batch {step + 1}/{num_batches} accuracy {sum(corrects) / len(corrects):.2%} time {format_duration(time.perf_counter() - log_time)}", flush=True)
            log_time = time.perf_counter()

        accuracy = sum(corrects) / len(samples)
        print(f"{target} accuracy {accuracy:.2%}", flush=True)
        result = {"accuracy": accuracy, "corrects": corrects, "texts": texts}
        save_result("longhealth", target, file_name, result)


if __name__ == "__main__":
    main()
