"""Record the model-free, fail-closed partial Step-8 status.

The available Stage-4 payload cases were captured for selected query heads,
not complete GQA groups.  This command deliberately reports no physical
omission rather than inferring unanimity from incomplete evidence.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.common import ROOT, load_config, read_json, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/rack_kv_v2_step8_partial")
    args = parser.parse_args()
    config = load_config(ROOT / "configs/rack_kv_v1.yaml")
    layers = [0, 8, 16, 24]
    capture_root = ROOT / ".tmp/stage6_gqa_full_heads/capture"
    stage4 = read_json(ROOT / config["existing_representative"])
    available = []
    for layer in layers:
        path = capture_root / f"llama31_layer{layer}_trace.safetensors"
        available.append({"layer": layer, "trace_exists": path.exists(), "trace_bytes": path.stat().st_size if path.exists() else 0})
    rows = [row for row in stage4["case_results"] if row.get("method_name") == "rack_kv" and int(row["layer_index"]) in layers]
    started = time.perf_counter()
    case_rows = [{
        "scope": "PARTIAL - 4 OF 5 REPRESENTATIVE LAYERS",
        "layer": int(row["layer_index"]),
        "case_key": row["case_key"],
        "complete_gqa_evidence": False,
        "physical_eligibility": False,
        "reason": "Stage-4 payload evidence contains selected query heads only; fail-closed physical path loads normally.",
        "payload_bytes_read": int(row.get("total_serialized_bytes", 0)),
        "payload_bytes_avoided": 0,
        "blocks_decoded": int(row.get("decoded_blocks", 0) or 0),
        "blocks_decoded_avoided": 0,
    } for row in rows]
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "physical_cases.csv", case_rows)
    write_csv(args.output / "timing.csv", [{"component": "physical_path", "status": "not_claimed_without_complete_GQA", "seconds": time.perf_counter() - started}])
    summary = {
        "status": "PARTIAL - 4 OF 5 REPRESENTATIVE LAYERS",
        "scientific_scope": "No model inference or trace capture; existing evidence only.",
        "layers": layers,
        "trace_inventory": available,
        "stage4_cases": len(rows),
        "complete_gqa_groups_available_to_physical_path": False,
        "physical_eligible_fraction": 0.0,
        "payload_bytes_actually_read": sum(row["payload_bytes_read"] for row in case_rows),
        "payload_bytes_avoided": 0,
        "decode_count": sum(row["blocks_decoded"] for row in case_rows),
        "decodes_avoided": 0,
        "certification_time_seconds": None,
        "mpfr_time_seconds": None,
        "io_time_seconds": None,
        "decode_time_seconds": None,
        "total_physical_execution_time_seconds": None,
        "peak_rss_bytes": None,
        "reason_for_fail_closed_zero": "Physical omission requires all four mapped query heads for each shared KV region; selected-head Stage-4 cases are insufficient.",
    }
    write_json(args.output / "summary.json", summary)
    (args.output / "test_report.txt").write_text("PARTIAL Step-8 only. No model computation. Physical omission fail-closed because complete per-region GQA evidence is unavailable.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
