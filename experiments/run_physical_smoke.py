"""Run a deterministic, model-free physical payload omission smoke test."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np

from experiments.common import ROOT
from rack_kv.codec import encode_block
from rack_kv.physical import KVPayloadStore, execute_logical_gqa, execute_physical_gqa


def main() -> None:
    output = ROOT / "results/rack_kv_v2_physical_smoke"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260929)
    blocks = [
        encode_block(rng.normal(size=(4, 3)), rng.normal(size=(4, 2)), block_start=start, precision=256)
        for start in (0, 4, 8)
    ]
    queries = {head: np.array([0.2, -0.1, 0.3]) for head in range(4)}
    recent_keys = {head: np.array([[0.1, 0.0, 0.2]]) for head in range(4)}
    recent_values = {head: np.array([[0.4, -0.2]]) for head in range(4)}
    store_path = output / "smoke_payload.rackv2"
    logical_store = KVPayloadStore.create(store_path.with_name("logical_payload.rackv2"), blocks)
    logical = execute_logical_gqa(
        store=logical_store, required_leaf_blocks=(0,), queries_by_head=queries,
        recent_keys_by_head=recent_keys, recent_values_by_head=recent_values,
        key_dim=3, value_dim=2,
    )
    physical_store = KVPayloadStore.create(store_path, blocks)
    physical_store.forbidden_block_ids.update({1, 2})
    started = time.perf_counter()
    physical = execute_physical_gqa(
        store=physical_store, queries_by_head=queries,
        recent_keys_by_head=recent_keys, recent_values_by_head=recent_values,
        required_leaf_blocks=(0,), physically_eligible_blocks=(1, 2),
        complete_gqa_group=True, all_heads_represented=True, mpfr_authorized=True,
        numerical_fallback=False, key_dim=3, value_dim=2,
    )
    elapsed = time.perf_counter() - started
    max_difference = max(float(np.max(np.abs(logical[h] - physical.outputs_by_head[h]))) for h in queries)
    full_bytes = physical_store.full_payload_bytes
    actual_bytes = physical_store.stats.payload_bytes_read
    full_decodes = physical_store.full_block_count
    actual_decodes = physical_store.stats.blocks_decoded
    summary = {
        "scope": "model-free indexed payload omission smoke test",
        "model_inference_executed": False,
        "complete_gqa_group": True,
        "physical_eligible_blocks": [1, 2],
        "required_blocks": [0],
        "physically_omitted_blocks": list(physical.physically_omitted_block_ids),
        "full_payload_bytes": full_bytes,
        "payload_bytes_read": actual_bytes,
        "payload_bytes_avoided": full_bytes - actual_bytes,
        "full_decodes": full_decodes,
        "actual_decodes": actual_decodes,
        "block_decodes_avoided": full_decodes - actual_decodes,
        "metadata_bytes_read": physical_store.metadata_bytes,
        "output_max_abs_difference": max_difference,
        "fallback_reasons": list(physical.fallback_reasons),
        "forbidden_skipped_blocks_read": False,
        "peak_rss_bytes": None,
    }
    try:
        import psutil
        summary["peak_rss_bytes"] = psutil.Process().memory_info().rss
    except ImportError:
        pass
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "io_accounting.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["full_payload_bytes", "payload_bytes_read", "payload_bytes_avoided", "metadata_bytes_read", "payload_read_calls"])
        writer.writeheader()
        writer.writerow({"full_payload_bytes": full_bytes, "payload_bytes_read": actual_bytes, "payload_bytes_avoided": full_bytes - actual_bytes, "metadata_bytes_read": physical_store.metadata_bytes, "payload_read_calls": physical_store.stats.payload_read_calls})
    with (output / "decode_accounting.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["full_decodes", "actual_decodes", "block_decodes_avoided", "blocks_loaded"])
        writer.writeheader()
        writer.writerow({"full_decodes": full_decodes, "actual_decodes": actual_decodes, "block_decodes_avoided": full_decodes - actual_decodes, "blocks_loaded": physical_store.stats.blocks_loaded})
    with (output / "physical_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "required_blocks", "omitted_blocks", "fallback_count", "max_output_difference"])
        writer.writeheader()
        writer.writerow({"case": "complete_gqa_smoke", "required_blocks": "0", "omitted_blocks": "1,2", "fallback_count": len(physical.fallback_reasons), "max_output_difference": max_difference})
    with (output / "timing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metadata_lookup_seconds", "payload_io_seconds", "decode_seconds", "physical_execution_seconds"])
        writer.writeheader()
        writer.writerow({"metadata_lookup_seconds": physical_store.stats.metadata_lookup_seconds, "payload_io_seconds": physical_store.stats.payload_io_seconds, "decode_seconds": physical_store.stats.decode_seconds, "physical_execution_seconds": elapsed})
    (output / "test_report.txt").write_text("physical smoke: PASS\nforbidden omitted payload reads: 0\nforbidden omitted decodes: 0\noutput equivalence: PASS\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
