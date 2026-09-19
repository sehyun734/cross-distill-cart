import torch
from torch import Tensor
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizerBase


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    id_inp: Tensor,
    mask_inp: Tensor,
    n_tok_max: int,
    temp: float,
) -> tuple[Tensor, Tensor]:
    id_out = model.generate(
        input_ids=id_inp,
        attention_mask=mask_inp,
        max_new_tokens=n_tok_max,
        do_sample=temp > 0,
        temperature=temp if temp > 0 else None,
        top_p=1.0,
        top_k=None,
        pad_token_id=tokenizer.pad_token_id,
        prefill_chunk_size=2048,
    )
    id_gen = id_out[:, id_inp.shape[1] :]
    id_eos = torch.tensor(model.generation_config.eos_token_id, device=model.device)
    is_eos = torch.isin(id_gen, id_eos)
    # 첫 eos 토큰 뒤의 토큰은 전부 마스킹함.
    mask_gen = ((is_eos.cumsum(dim=1) - is_eos.long()) == 0).long()
    return id_gen, mask_gen


def forward(
    model: PreTrainedModel,
    id_inp: Tensor,
    cache: DynamicCache | None = None,
) -> Tensor:
    return model(input_ids=id_inp, past_key_values=cache, use_cache=False).logits
