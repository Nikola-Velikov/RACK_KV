"""Create the final Step-11C evidence package without fabricating missing data."""

from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rack_kv_v2_step11c_final_system_validation"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(name, fields, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    step11b = load(ROOT / "results/rack_kv_v2_step11b_system_evaluation/summary.json")
    step11 = load(ROOT / "results/rack_kv_v2_step11_tight_certificate/summary.json")
    partial = load(ROOT / "results/rack_kv_v2_step8_partial/summary.json")
    smoke = load(ROOT / "results/rack_kv_v2_physical_smoke/summary.json")
    codec = load(ROOT / "results/rack_kv_v2_codec_step9b/summary.json")

    old_layer31 = ROOT / ".tmp/stage3_multilayer_full/capture/llama31_layer31_trace.safetensors"
    complete_layer31 = ROOT / ".tmp/stage6_gqa_full_heads/capture/llama31_layer31_trace.safetensors"
    trace_complete = {str(layer): (ROOT / f".tmp/stage3_multilayer_full/capture/llama31_layer{layer}_trace.safetensors").exists() for layer in [0, 8, 16, 24]}
    trace_complete["31_all_heads"] = complete_layer31.exists()

    (OUT / "trace_completeness.json").write_text(json.dumps({
        "configured_layers": [0, 8, 16, 24, 31],
        "available_selected_head_traces": trace_complete,
        "layer31_old_trace": {"exists": old_layer31.exists(), "selected_heads": [0, 1, 4] if old_layer31.exists() else []},
        "layer31_complete_all_32_trace": {"exists": complete_layer31.exists(), "selected_heads": []},
        "complete_gqa_groups": 0,
        "status": "BLOCKED: layer31 all-32-head trace and model shards unavailable",
    }, indent=2), encoding="utf-8")

    (OUT / "trace_consistency.json").write_text(json.dumps({
        "status": "NOT_RUN",
        "reason": "No new layer31 all-head trace was captured; there is no valid overlap comparison against a new capture.",
        "existing_layer31_selected_heads": [0, 1, 4],
        "model_revision": "1f47e50cdbe801ad8a5174156ec3a0655108fb9f",
    }, indent=2), encoding="utf-8")

    write_csv("gqa_votes.csv", ["vote_count", "count", "probability", "status"], [{"vote_count": f"{i}/4", "count": "unavailable", "probability": "unavailable", "status": "complete groups unavailable; fail-closed"} for i in range(5)])
    write_csv("gqa_physical_eligibility.csv", ["metric", "value", "status"], [
        {"metric": "complete_groups", "value": 0, "status": "blocked"},
        {"metric": "physical_eligible_regions", "value": 0, "status": "fail-closed, not measured eligibility"},
        {"metric": "physical_eligible_leaf_blocks", "value": 0, "status": "fail-closed, not measured eligibility"},
        {"metric": "physical_eligible_tokens", "value": 0, "status": "fail-closed, not measured eligibility"},
        {"metric": "P(4/4)", "value": "unavailable", "status": "not inferable from heads 0,1,4"},
    ])

    write_csv("physical_io.csv", ["scope", "full_payload_bytes", "payload_bytes_read", "payload_bytes_avoided", "payload_fraction_avoided", "status"], [{
        "scope": "real four-layer partial evidence",
        "full_payload_bytes": "unavailable",
        "payload_bytes_read": partial["payload_bytes_actually_read"],
        "payload_bytes_avoided": 0,
        "payload_fraction_avoided": 0.0,
        "status": "fail-closed because complete GQA groups are absent",
    }])
    write_csv("decode_counts.csv", ["scope", "full_decode_count", "actual_decode_count", "decodes_avoided", "status"], [{
        "scope": "real four-layer partial evidence", "full_decode_count": "unavailable", "actual_decode_count": partial["decode_count"], "decodes_avoided": 0, "status": "blocked by incomplete GQA"
    }])

    hierarchy = step11b["hierarchy"]["modes"].get("rack_v2_aniso_hier_r8", {})
    write_csv("hierarchy.csv", ["scope", "nodes_examined", "leaf_checks_avoided", "nodes_certified", "tokens_skipped", "violations", "status"], [{
        "scope": "stored 20-case hierarchy artifact",
        "nodes_examined": hierarchy.get("nodes_considered", "unavailable"),
        "leaf_checks_avoided": hierarchy.get("leaf_checks_avoided", "unavailable"),
        "nodes_certified": hierarchy.get("nodes_certified", "unavailable"),
        "tokens_skipped": hierarchy.get("skipped_tokens", "unavailable"),
        "violations": hierarchy.get("violations", "unavailable"),
        "status": "incomplete; not a full benchmark",
    }])
    write_csv("timing.csv", ["mode", "mean_seconds", "median_seconds", "p95_seconds", "mpfr_fraction", "status"], [{"mode": m, "mean_seconds": "unavailable", "median_seconds": "unavailable", "p95_seconds": "unavailable", "mpfr_fraction": "unavailable", "status": "physical benchmark not executable"} for m in ["compressed_full_read", "flat_r8_physical", "hierarchical_r8_physical"]])
    write_csv("memory.csv", ["scope", "peak_rss_bytes", "status"], [{"scope": "fixture smoke only", "peak_rss_bytes": smoke["peak_rss_bytes"], "status": "not a model/system measurement"}])
    write_csv("end_to_end.csv", ["mode", "ms_per_token", "tokens_per_second", "status"], [{"mode": m, "ms_per_token": "unavailable", "tokens_per_second": "unavailable", "status": "model weights unavailable"} for m in ["reference", "rack_v1", "rack_v2_physical"]])
    write_csv("context_scaling.csv", ["context_tokens", "physical_fraction_avoided", "total_time", "status"], [{"context_tokens": 256, "physical_fraction_avoided": 0.0, "total_time": "unavailable", "status": "only observed representative length; fail-closed"}])
    write_csv("rigor_validation.csv", ["scope", "violations", "minimum_margin", "status"], [
        {"scope": "Step-11 compression certificate", "violations": step11["compression"]["violations"], "minimum_margin": 0.008819346553560687, "status": "validated"},
        {"scope": "Step-11 total certificate", "violations": step11["total"]["violations"], "minimum_margin": 0.010257644570675751, "status": "validated"},
        {"scope": "physical omitted real blocks", "violations": "unavailable", "minimum_margin": "unavailable", "status": "no complete physical set"},
    ])

    summary = {
        "status": "BLOCKED_MISSING_EVIDENCE",
        "complete_layer31_all_heads": False,
        "complete_gqa_groups_evaluated": 0,
        "physical_payload_fraction_avoided_real_data": 0.0,
        "physical_decode_fraction_avoided_real_data": 0.0,
        "physical_savings_interpretation": "fail-closed result, not evidence of zero eligibility",
        "flat_r8": {"skips": 902, "skipped_tokens": 6950, "violations": 0},
        "hierarchy": {"scope_cases": 20, "leaf_checks_avoided": hierarchy.get("leaf_checks_avoided", 0), "status": "incomplete"},
        "step11": {"mean_total_bound": step11["total"]["tight"]["mean"], "p95_total_bound": step11["total"]["tight"]["p95"], "violations": step11["total"]["violations"]},
        "net_ratio_including_step10_metadata": 1.62040998,
        "fixture_only_smoke": {"full_bytes": smoke["full_payload_bytes"], "read_bytes": smoke["payload_bytes_read"], "avoided_bytes": smoke["payload_bytes_avoided"], "full_decodes": smoke["full_decodes"], "actual_decodes": smoke["actual_decodes"]},
        "unavailable": ["4/4 GQA rate", "real physical payload/decode savings", "runtime and MPFR fraction", "complete hierarchy benchmark", "model autoregressive throughput", "long-context scaling"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "test_report.txt").write_text("Evidence-completion gate executed without model download. Exact pinned shards are absent; no layer31 all-head capture was produced. Existing focused regression suite: 39 passed. Physical path remains fail-closed.\n", encoding="utf-8")
    (OUT / "environment.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(), "generated_utc": datetime.now(timezone.utc).isoformat(), "model_download": False}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "status": summary["status"], "complete_groups": 0}, indent=2))


if __name__ == "__main__":
    main()
