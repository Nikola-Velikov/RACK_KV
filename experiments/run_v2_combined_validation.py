"""Replay frozen representative payloads for real V2 logical-skip validation.

This intentionally contains no model-loading, trace-capture, or tensor-download
code.  It is the single deferred expensive validation command for flat and
hierarchical anisotropic execution.
"""
from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import numpy as np

from experiments.common import ROOT, load_config, read_json, write_csv, write_json
from experiments.run_anisotropic_shadow import _case_payload
from rack_kv.anisotropic import execute_anisotropic_flat
from rack_kv.hierarchy import execute_anisotropic_hierarchical
from rack_kv.gqa import gqa_groups
from rack_kv.stage2 import validate_compact_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rack_kv_v1.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fanout", type=int, default=4)
    parser.add_argument("--flat-ranks", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-only case limit; omit for all frozen cases.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based start index for bounded replay chunks.")
    parser.add_argument("--skip-hierarchy", action="store_true")
    parser.add_argument("--only-hierarchy", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    stage4_root = ROOT / ".tmp/stage4_baselines_full"
    stage4 = read_json(ROOT / config["existing_representative"])
    rows = [row for row in stage4["case_results"] if row["method_name"] == "rack_kv"]
    start = max(0, int(args.start_index))
    end = None if args.limit is None else start + int(args.limit)
    rows = rows[start:end]
    traces = {}
    results = []
    for ordinal, row in enumerate(rows, 1):
        layer = int(row["layer_index"])
        if layer not in traces:
            traces[layer] = validate_compact_trace(ROOT / config["existing_capture"] / f"llama31_layer{layer}_trace.safetensors", allow_nonzero_layer=True)
        trace = traces[layer]
        query = trace.queries[int(row["record_index"]), int(row["query_local_index"])].float().numpy().astype(np.float64)
        blocks, recent_keys, recent_values = _case_payload(row, stage4_root)
        common = dict(query=query, recent_keys=recent_keys, recent_values=recent_values, historical_blocks=blocks, tolerance=float(config["epsilon"]), precision=int(config["mpfr_precision"]), attention_scale=float(trace.scaling))
        if not args.only_hierarchy:
          for rank in args.flat_ranks:
            outcome = execute_anisotropic_flat(rank=rank, **common)
            results.append({"case_key": row["case_key"], "mode": "rack_v1_sphere" if rank == 0 else outcome.mode, "layer": layer, "rank": rank, "skipped_blocks": len(outcome.skipped_block_starts), "skipped_tokens": sum(block.block_len for block in blocks if block.header.block_start in outcome.skipped_block_starts), "attention_l2_error": float(np.linalg.norm(outcome.compression_only_output - outcome.output)), "certificate_bound": float(outcome.certificate.certificate_bound), "mpfr_fallback": outcome.certificate.numerical_fallback_used})
        if not args.skip_hierarchy and not args.only_hierarchy:
            outcome = execute_anisotropic_hierarchical(rank=8, fanout=args.fanout, **common)
            results.append({"case_key": row["case_key"], "mode": outcome.mode, "layer": layer, "rank": 8, "skipped_blocks": len(outcome.skipped_block_starts), "skipped_tokens": sum(block.block_len for block in blocks if block.header.block_start in outcome.skipped_block_starts), "attention_l2_error": float(np.linalg.norm(outcome.compression_only_output - outcome.output)), "certificate_bound": float(outcome.traversal.certificate_bound), "mpfr_fallback": outcome.traversal.numerical_fallback_used, "nodes_considered": outcome.traversal.nodes_considered, "nodes_certified": outcome.traversal.nodes_certified, "nodes_descended": outcome.traversal.nodes_descended, "descendant_leaf_evaluations_avoided": outcome.traversal.descendant_leaf_evaluations_avoided})
        if args.only_hierarchy:
            outcome = execute_anisotropic_hierarchical(rank=8, fanout=args.fanout, **common)
            results.append({"case_key": row["case_key"], "mode": outcome.mode, "layer": layer, "rank": 8, "skipped_blocks": len(outcome.skipped_block_starts), "skipped_tokens": sum(block.block_len for block in blocks if block.header.block_start in outcome.skipped_block_starts), "attention_l2_error": float(np.linalg.norm(outcome.compression_only_output - outcome.output)), "certificate_bound": float(outcome.traversal.certificate_bound), "mpfr_fallback": outcome.traversal.numerical_fallback_used, "nodes_considered": outcome.traversal.nodes_considered, "nodes_certified": outcome.traversal.nodes_certified, "nodes_descended": outcome.traversal.nodes_descended, "descendant_leaf_evaluations_avoided": outcome.traversal.descendant_leaf_evaluations_avoided})
        if ordinal % 25 == 0 or ordinal == len(rows):
            print(f"validated {ordinal}/{len(rows)} frozen cases", flush=True)
        # MPFR intervals and per-node basis arrays are deliberately local to a
        # case; collect them before the next case to keep long replays bounded.
        del outcome, blocks, recent_keys, recent_values, query
        gc.collect()
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = f"_{start}_{start + len(rows)}" if args.start_index or args.limit is not None else ""
    write_csv(args.output / f"v2_validation_cases{suffix}.csv", results)
    summaries = {}
    for mode in sorted({row["mode"] for row in results}):
        grouped = [row for row in results if row["mode"] == mode]
        summaries[mode] = {"cases": len(grouped), "skipped_blocks": sum(row["skipped_blocks"] for row in grouped), "skipped_tokens": sum(row["skipped_tokens"] for row in grouped), "mean_attention_l2_error": float(np.mean([row["attention_l2_error"] for row in grouped])), "mpfr_fallbacks": sum(bool(row["mpfr_fallback"]) for row in grouped)}
    groups = gqa_groups(num_attention_heads=int(config["query_heads"]), num_key_value_heads=int(config["kv_heads"]))
    observed_heads = set(int(row["query_head_global"]) for row in rows)
    incomplete_groups = {group_index: list(group) for group_index, group in enumerate(groups) if not set(group).issubset(observed_heads)}
    gqa_status = {
        "status": "ready_only_with_complete_mapped_query_head_traces" if incomplete_groups else "eligible_for_replay",
        "observed_query_heads": sorted(observed_heads),
        "incomplete_groups": incomplete_groups,
        "reason": "The frozen representative trace samples heads 0, 1, and 4, not every four-head GQA group." if incomplete_groups else None,
        "future_api": "rack_kv.gqa.certify_gqa_hierarchical_group",
    }
    write_json(args.output / f"summary{suffix}.json", {"scope": "Frozen representative payload replay; no model inference.", "epsilon": config["epsilon"], "fanout": args.fanout, "start_index": start, "case_count": len(rows), "summary": summaries, "gqa_physical_eligibility": gqa_status})


if __name__ == "__main__":
    main()
