"""Layerwise forensic recapture; this does not modify RACK-KV algorithms."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.common import ROOT, load_config, seed_execution, write_json
OLD_ROOT = ROOT / ".tmp" / "stage3_multilayer_full" / "capture"
NEW_ROOT = ROOT / ".tmp" / "trace_forensics_old_heads"
LAYERS = (0, 8, 16, 24, 31)
HEADS = (0, 1, 4)
OUT = ROOT / "results" / "trace_forensics"


def _sha_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _sha_ints(values: tuple[int, ...]) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.detach().float() - b.detach().float()).abs().reshape(-1).numpy()
    return {
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "p95_abs_diff": float(np.percentile(diff, 95)),
        "exact_equal": bool(torch.equal(a, b)),
        "within_0.01": bool(np.max(diff) <= 0.01),
    }


def _environment(config: dict) -> dict:
    versions = {}
    for package in ("torch", "transformers", "huggingface-hub", "safetensors", "gmpy2"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "versions": versions,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "attention_implementation": "eager",
        "dtype": config["dtype"],
        "autocast_enabled": False,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else False,
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32) if torch.cuda.is_available() else False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "model_training": False,
        "torch_threads": torch.get_num_threads(),
        "config": config,
    }


def main() -> None:
    config = load_config(ROOT / "configs" / "rack_kv_v1.yaml")
    seed_execution(config)
    OUT.mkdir(parents=True, exist_ok=True)

    # Install the same local-only hub import shim used by the capture harness
    # before importing transformers-dependent project modules.
    from experiments import capture_local_layer31 as local_capture
    from rack_kv.stage2 import validate_compact_trace

    frozen = validate_compact_trace(OLD_ROOT / "llama31_layer0_trace.safetensors", allow_nonzero_layer=True)
    write_json(OUT / "current_environment.json", _environment(config))
    write_json(OUT / "input_hashes.json", {
        "token_ids_sha256": _sha_ints(frozen.token_ids),
        "token_positions_sha256": _sha_ints(frozen.query_positions),
        "visible_lengths_sha256": _sha_ints(frozen.visible_lengths),
        "token_count": len(frozen.token_ids),
        "query_positions": list(config["query_positions"]),
        "selected_query_heads": list(HEADS),
    })

    # Importing the local harness installs the exact local-only shard reader.
    import rack_kv.llama_trace as lt

    if not all((NEW_ROOT / f"llama31_layer{layer}_trace.safetensors").exists() for layer in LAYERS):
        lt.run_multilayer_llama31_capture(
            output_dir=NEW_ROOT,
            repo_id=config["model"],
            repo_revision=config["revision"],
            prompt_token_ids=list(frozen.token_ids),
            capture_layer_indices=LAYERS,
            selected_query_heads=HEADS,
            allow_insecure_tls=False,
        )
    _ = local_capture

    rows = []
    for layer in LAYERS:
        old = validate_compact_trace(OLD_ROOT / f"llama31_layer{layer}_trace.safetensors", allow_nonzero_layer=True)
        new = validate_compact_trace(NEW_ROOT / f"llama31_layer{layer}_trace.safetensors", allow_nonzero_layer=True)
        for head_index, head in enumerate(HEADS):
            for name, old_tensor, new_tensor in (
                ("queries", old.queries[:, head_index, :], new.queries[:, head_index, :]),
                ("model_head_outputs", old.model_head_outputs[:, head_index, :], new.model_head_outputs[:, head_index, :]),
            ):
                rows.append({"layer": layer, "tensor": name, "head": head, **_stats(old_tensor, new_tensor)})
        for kv_head in sorted(set(old.query_to_kv_heads)):
            old_index = old.selected_kv_heads.index(kv_head)
            new_index = new.selected_kv_heads.index(kv_head)
            for name, old_tensor, new_tensor in (
                ("final_keys", old.final_keys[old_index], new.final_keys[new_index]),
                ("final_values", old.final_values[old_index], new.final_values[new_index]),
            ):
                rows.append({"layer": layer, "tensor": name, "head": kv_head, **_stats(old_tensor, new_tensor)})

    write_json(OUT / "historical_environment.json", json.loads((OLD_ROOT / "dependency_versions.json").read_text(encoding="utf-8")))
    write_json(OUT / "environment_diff.json", {
        "current": json.loads((OUT / "current_environment.json").read_text(encoding="utf-8")),
        "historical": json.loads((OUT / "historical_environment.json").read_text(encoding="utf-8")),
        "capture_identity": {"model": config["model"], "revision": config["revision"], "layers": list(LAYERS), "heads": list(HEADS)},
    })
    write_json(OUT / "diagnosis.json", {
        "historical_trace_reproduced": all(row["within_0.01"] for row in rows),
        "earliest_divergent_layer": next((layer for layer in LAYERS if any(row["layer"] == layer and not row["within_0.01"] for row in rows)), None),
        "first_divergent_tensor": next((row["tensor"] for row in rows if not row["within_0.01"]), None),
        "decision": "FALLBACK_REQUIRED" if any(not row["within_0.01"] for row in rows) else "HISTORICAL_TRACE_REPRODUCED",
        "new_capture": str(NEW_ROOT),
    })
    import csv
    with (OUT / "layerwise_trace_diff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "FORENSICS_COMPLETE", "rows": len(rows), "new_capture": str(NEW_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
