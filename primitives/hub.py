import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from safetensors.torch import load_file, save
import torch
from torch import Tensor


def get_repo_name() -> str:
    return f"{HfApi().whoami()['name']}/cross-distill-cart"


def save_synthesis(
    path: str,
    rows: list[dict],
    tensors: dict[str, Tensor],
) -> None:
    data = save(tensors)
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=f"{path}/rows.json", path_or_fileobj=json.dumps(rows, indent=2).encode()),
        CommitOperationAdd(path_in_repo=f"{path}/logprobs.safetensors", path_or_fileobj=data),
    ]
    api.create_commit(repo_name, operations=operations, commit_message=f"Save {path}")
    print(f"saved {path}/rows.json, {path}/logprobs.safetensors", flush=True)


def save_cartridge(
    path: str,
    sink: Tensor,
    cartridge: Tensor,
) -> None:
    data = save({"sink": sink.detach().cpu().contiguous(), "cartridge": cartridge.detach().cpu().contiguous()})
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    api.upload_file(path_or_fileobj=data, path_in_repo=f"{path}/cartridge.safetensors", repo_id=repo_name)
    print(f"saved {path}/cartridge.safetensors", flush=True)


def load_cartridge(
    path: str,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    tensors = load_file(hf_hub_download(get_repo_name(), f"{path}/cartridge.safetensors"), device=str(device))
    return tensors["sink"], tensors["cartridge"]


def get_pending_targets(
    model_name: str,
    result_name: str,
    file_name: str,
    methods: tuple[str, ...] = ("base", "icl", "cartridge"),
) -> list[str]:
    files = set(HfApi().list_repo_files(get_repo_name()))
    targets = [f"base/{model_name}", f"icl/{model_name}"]
    for file in sorted(files):
        if file.startswith(f"cartridge/{model_name}/") and file.endswith("/cartridge.safetensors"):
            targets.append(str(Path(file).parent))
    targets = [target for target in targets if target.split("/")[0] in methods and f"{result_name}/{target}/{file_name}.json" not in files]
    print(f"{result_name} pending {len(targets)}", *targets, sep="\n  ", flush=True)
    return targets


def save_result(
    result_name: str,
    target: str,
    file_name: str,
    result: dict,
) -> None:
    path = f"{result_name}/{target}/{file_name}.json"
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    api.upload_file(path_or_fileobj=json.dumps(result, indent=2).encode(), path_in_repo=path, repo_id=repo_name)
    print(f"saved {path}", flush=True)
