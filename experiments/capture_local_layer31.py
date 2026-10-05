"""Capture layer 31 from the exact local checkpoint without network access."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import sys
from pathlib import Path


def load_broken_hf_package() -> None:
    """Load the already-installed package whose directory was renamed by pip."""
    package = Path(sys.executable).parent / "Lib" / "site-packages" / "~uggingface_hub"
    if not package.exists():
        raise RuntimeError(f"Local huggingface_hub package not found: {package}")
    original_version = importlib.metadata.version
    importlib.metadata.version = lambda name: "1.24.0" if name == "huggingface-hub" else original_version(name)
    spec = importlib.util.spec_from_file_location(
        "huggingface_hub",
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["huggingface_hub"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)


load_broken_hf_package()

import numpy as np
import torch
from safetensors import safe_open

from experiments.common import ROOT, load_config, seed_execution, write_json
import rack_kv.llama_trace as lt


MODEL_DIR = Path(r"C:\Users\Niki\Downloads\llama31-pinned")
REVISION = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"
OUTPUT = ROOT / ".tmp" / "stage11c_local_layer31_capture"


class LocalShard:
    def __init__(self, path: Path, fetched_tensor_hashes: dict[str, str]):
        self.path = path
        self.fetched_tensor_hashes = fetched_tensor_hashes

    def _get(self, name: str) -> torch.Tensor:
        with safe_open(str(self.path), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def fetch_tensors(self, tensor_names, *, max_gap_bytes=8 * 1024 * 1024):
        result = {}
        for name in tensor_names:
            tensor = self._get(name)
            self.fetched_tensor_hashes[name] = hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            result[name] = tensor
        return result

    def fetch_rows(self, tensor_name: str, rows):
        tensor = self._get(tensor_name)
        row_indices = list(rows)
        result = tensor[row_indices].contiguous()
        self.fetched_tensor_hashes[f"{tensor_name}/rows"] = hashlib.sha256(result.view(torch.uint8).numpy().tobytes()).hexdigest()
        return result


def local_from_repo(cls, *, repo_id, revision, filename, download_policy, fetched_tensor_hashes, timeout_s=120.0):
    if repo_id != "NousResearch/Meta-Llama-3.1-8B" or revision != REVISION:
        raise ValueError("Local capture received an unexpected model identity.")
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return LocalShard(path, fetched_tensor_hashes)


def local_text_file(repo_id, revision, filename, destination, *, download_policy):
    if repo_id != "NousResearch/Meta-Llama-3.1-8B" or revision != REVISION:
        raise ValueError("Local capture received an unexpected model identity.")
    source = MODEL_DIR / filename
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination, "local_file"


lt._RemoteSafetensorsShard.from_repo = classmethod(local_from_repo)
lt._download_text_file = local_text_file


def main() -> None:
    config = load_config(ROOT / "configs" / "rack_kv_v1.yaml")
    seed_execution(config)
    source = ROOT / config["existing_capture"] / "llama31_layer0_trace.safetensors"
    from rack_kv.stage2 import validate_compact_trace

    frozen = validate_compact_trace(source, allow_nonzero_layer=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = lt.run_multilayer_llama31_capture(
        output_dir=OUTPUT,
        repo_id=config["model"],
        repo_revision=config["revision"],
        prompt_token_ids=frozen.token_ids,
        capture_layer_indices=(31,),
        selected_query_heads=tuple(range(config["query_heads"])),
        allow_insecure_tls=False,
    )
    write_json(OUTPUT / "local_capture_identity.json", {
        "model": config["model"],
        "revision": config["revision"],
        "model_directory": str(MODEL_DIR),
        "network_access": False,
        "selected_query_heads": list(result.selected_query_heads),
        "selected_kv_heads": list(result.selected_kv_heads),
        "query_to_kv_heads": list(result.query_to_kv_heads),
        "layer_indices": list(result.selected_layer_indices),
    })
    print(json.dumps({"status": "CAPTURE_OK", "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
