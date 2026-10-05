"""Capture complete representative GQA heads for V2 validation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.common import ROOT, load_config, write_json
from rack_kv.llama_trace import run_multilayer_llama31_capture
from rack_kv.stage2 import validate_compact_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rack_kv_v1.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default=None, help="Comma-separated layers to capture; defaults to all configured representative layers.")
    parser.add_argument("--allow-insecure-tls", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    layers = tuple(config["layers"] if args.layers is None else int(part.strip()) for part in args.layers.split(",")) if args.layers is not None else tuple(config["layers"])
    source = validate_compact_trace(ROOT / config["existing_capture"] / "llama31_layer0_trace.safetensors", allow_nonzero_layer=True)
    result = run_multilayer_llama31_capture(
        output_dir=args.output,
        repo_id=config["model"], repo_revision=config["revision"],
        prompt_token_ids=source.token_ids, capture_layer_indices=layers,
        selected_query_heads=tuple(range(int(config["query_heads"]))),
        allow_insecure_tls=args.allow_insecure_tls,
    )
    comparison = {}
    for layer in layers:
        new = validate_compact_trace(args.output / f"llama31_layer{layer}_trace.safetensors", allow_nonzero_layer=True)
        old_path = ROOT / config["existing_capture"] / f"llama31_layer{layer}_trace.safetensors"
        if old_path.exists():
            old = validate_compact_trace(old_path, allow_nonzero_layer=True)
            overlap = [head for head in old.selected_query_heads if head in new.selected_query_heads]
            old_indices = [new.selected_query_heads.index(head) for head in overlap]
            old_overlap_indices = [old.selected_query_heads.index(head) for head in overlap]
            difference = (new.queries[:, old_indices, :].float() - old.queries[:, old_overlap_indices, :].float()).abs()
            comparison[str(layer)] = {"status": "compared_overlap", "overlap_heads": overlap, "max_abs_difference": float(difference.max()), "mean_abs_difference": float(difference.mean()), "matches_exact_bfloat16": bool(float(difference.max()) == 0.0)}
        else:
            comparison[str(layer)] = {"status": "new_layer_no_frozen_overlap", "overlap_heads": [], "max_abs_difference": None, "mean_abs_difference": None, "matches_exact_bfloat16": None}
    write_json(args.output / "trace_reproducibility.json", {"source_selected_heads": list(source.selected_query_heads), "captured_query_heads": list(range(int(config["query_heads"]))), "comparison": comparison})
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
