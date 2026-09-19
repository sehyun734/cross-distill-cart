import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from shared.prompts import make_txt_ctx, make_txt_record


def load_llm(
    name_model: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model = AutoModelForCausalLM.from_pretrained(name_model, dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(name_model, padding_side="left")
    match model.config.model_type:
        case "llama":
            # 저자는 answer를 하나씩 생성했지만, 여기서는 batch로 한 번에 생성함.
            # 이때 batch 생성에는 pad 토큰이 필요한데, llama는 pad 토큰이 없고 eos 토큰을 같이 씀.
            # 따라서 eos 토큰과 pad 토큰을 구분하여 명시함.
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        case _:
            raise NotImplementedError(model.config.model_type)
    return model, tokenizer


def load_longhealth(
    n_patient: int = 10,
) -> tuple[str, list[list[str]]]:
    patients = requests.get("https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json").json()
    txt_corpus = "Below is a panel of patient records."
    txts_ctx = []
    for i_patient in range(1, n_patient + 1):
        key_patient = f"patient_{i_patient:02d}"
        patient = patients[key_patient]
        txts_note = []
        txts_ctx_patient = []
        for key_note, txt_note_raw in patient["texts"].items():
            txt_note = f"<{key_note}>\n{txt_note_raw}\n</{key_note}>"
            txts_note.append(txt_note)
            txt_ctx = make_txt_ctx(key_patient, patient, txt_note)
            txts_ctx_patient.append(txt_ctx)
        txts_ctx.append(txts_ctx_patient)
        txt_record = make_txt_record(key_patient, patient, txts_note)
        txt_corpus += "\n\n" + txt_record
    return txt_corpus, txts_ctx
