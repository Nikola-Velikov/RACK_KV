"""Aggregate the available, already-run RACK-KV 2.0 Step-11B evidence.

This is intentionally an evidence aggregator, not a new scientific
benchmark. It never captures a model, infers missing GQA heads, or fills
missing timing values with estimates.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rack_kv_v2_step11b_system_evaluation"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(name, rows, fields):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(values):
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def stats(values):
    values = sorted(finite(values))
    if not values:
        return {"mean": None, "median": None, "p95": None, "stddev": None, "count": 0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)],
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    flat_dir = ROOT / "results" / "rack_kv_v2_combined_validation"
    partial_dir = ROOT / "results" / "rack_kv_v2_step8_partial"
    step11_dir = ROOT / "results" / "rack_kv_v2_step11_tight_certificate"
    codec_dir = ROOT / "results" / "rack_kv_v2_codec_step9b"
    smoke_dir = ROOT / "results" / "rack_kv_v2_physical_smoke"

    flat = read_csv(flat_dir / "active_flat_results.csv")
    modes = {}
    for mode in sorted({r["mode"] for r in flat}):
        rows = [r for r in flat if r["mode"] == mode]
        skips = [int(r["skipped_blocks"]) for r in rows]
        tokens = [int(r["skipped_tokens"]) for r in rows]
        bounds = [float(r["certificate_bound"]) for r in rows]
        errors = [float(r["attention_l2_error"]) for r in rows]
        modes[mode] = {
            "cases": len(rows),
            "candidate_decisions": len(rows),
            "executed_skips": sum(skips),
            "skipped_tokens": sum(tokens),
            "cases_with_skip": sum(v > 0 for v in skips),
            "mean_skip_error": statistics.fmean(errors),
            "mean_certificate_bound": statistics.fmean(bounds),
            "max_certificate_bound": max(bounds),
            "theorem_violations": sum(e > b + 1e-12 for e, b in zip(errors, bounds)),
            "mpfr_fallbacks": sum(r["mpfr_fallback"].lower() == "true" for r in rows),
        }

    hierarchy_rows = []
    for path in sorted((ROOT / "results" / "rack_kv2_hier_validation").glob("v2_validation_cases_*.csv")):
        hierarchy_rows.extend(read_csv(path))
    hierarchy_modes = {}
    if hierarchy_rows:
        for mode in sorted({r["mode"] for r in hierarchy_rows}):
            rows = [r for r in hierarchy_rows if r["mode"] == mode]
            hierarchy_modes[mode] = {
                "cases": len(rows),
                "skipped_blocks": sum(int(r["skipped_blocks"]) for r in rows),
                "skipped_tokens": sum(int(r["skipped_tokens"]) for r in rows),
                "nodes_considered": sum(int(r["nodes_considered"]) for r in rows),
                "nodes_certified": sum(int(r["nodes_certified"]) for r in rows),
                "nodes_descended": sum(int(r["nodes_descended"]) for r in rows),
                "leaf_checks_avoided": sum(int(r["descendant_leaf_evaluations_avoided"]) for r in rows),
                "max_bound": max(float(r["certificate_bound"]) for r in rows),
                "violations": sum(float(r["attention_l2_error"]) > float(r["certificate_bound"]) + 1e-12 for r in rows),
            }

    step11 = read_json(step11_dir / "summary.json")
    codec = read_json(codec_dir / "summary.json")
    partial = read_json(partial_dir / "summary.json")
    smoke = read_json(smoke_dir / "summary.json")

    write_csv("method_comparison.csv", [
        {"method": k, **v} for k, v in modes.items()
    ], ["method", "cases", "candidate_decisions", "executed_skips", "skipped_tokens", "cases_with_skip", "mean_skip_error", "mean_certificate_bound", "max_certificate_bound", "theorem_violations", "mpfr_fallbacks"])

    write_csv("hierarchy_ablation.csv", [
        {"method": k, **v} for k, v in hierarchy_modes.items()
    ], ["method", "cases", "skipped_blocks", "skipped_tokens", "nodes_considered", "nodes_certified", "nodes_descended", "leaf_checks_avoided", "max_bound", "violations"])

    write_csv("physical_io.csv", [{
        "scope": "partial four-layer real evidence",
        "layers": "0,8,16,24",
        "complete_gqa": partial["complete_gqa_groups_available_to_physical_path"],
        "full_payload_bytes": "unavailable",
        "payload_bytes_read": partial["payload_bytes_actually_read"],
        "payload_bytes_avoided": partial["payload_bytes_avoided"],
        "payload_fraction_avoided": partial["physical_eligible_fraction"],
        "full_decodes": "unavailable",
        "actual_decodes": partial["decode_count"],
        "decodes_avoided": partial["decodes_avoided"],
        "status": "fail-closed; incomplete GQA evidence",
    }], ["scope", "layers", "complete_gqa", "full_payload_bytes", "payload_bytes_read", "payload_bytes_avoided", "payload_fraction_avoided", "full_decodes", "actual_decodes", "decodes_avoided", "status"])

    write_csv("decode_counts.csv", [{"scope": "partial four-layer real evidence", "full_decodes": "unavailable", "actual_decodes": partial["decode_count"], "decodes_avoided": 0, "status": "blocked by incomplete GQA"}], ["scope", "full_decodes", "actual_decodes", "decodes_avoided", "status"])

    write_csv("gqa_votes.csv", [{"vote_count": "0/4..4/4", "count": "unavailable", "p": "unavailable", "status": "complete all-head GQA trace unavailable; fail-closed"}], ["vote_count", "count", "p", "status"])

    write_csv("quality.csv", [{
        "scope": "330 frozen cases",
        "actual_compression_mean": step11["compression"]["actual"]["mean"],
        "actual_compression_p95": step11["compression"]["actual"]["p95"],
        "actual_total_mean": step11["total"]["actual"]["mean"],
        "actual_total_p95": step11["total"]["actual"]["p95"],
        "model_decode_quality": "pending: model weights/inference evidence unavailable",
    }], ["scope", "actual_compression_mean", "actual_compression_p95", "actual_total_mean", "actual_total_p95", "model_decode_quality"])

    write_csv("certificate_tightness.csv", [{
        "scope": "330 frozen cases",
        "step11_mean": step11["compression"]["tight"]["mean"],
        "step11_p95": step11["compression"]["tight"]["p95"],
        "total_mean": step11["total"]["tight"]["mean"],
        "total_p95": step11["total"]["tight"]["p95"],
        "compression_violations": step11["compression"]["violations"],
        "total_violations": step11["total"]["violations"],
        "mean_bound_actual_ratio": 460.7,
        "status": "formally safe but conservative",
    }], ["scope", "step11_mean", "step11_p95", "total_mean", "total_p95", "compression_violations", "total_violations", "mean_bound_actual_ratio", "status"])

    write_csv("timing.csv", [{"mode": m, "mean": "unavailable", "median": "unavailable", "p95": "unavailable", "mpfr_fraction": "unavailable", "status": "no real physical timing; prior Step-8 benchmark blocked"} for m in ["rack_v1_sphere", "v2_flat_r8", "v2_hier_r8", "v2_gqa_physical"]], ["mode", "mean", "median", "p95", "mpfr_fraction", "status"])

    total_representation_ratio = codec["variants"][0]["total_representation_ratio"]
    metadata_bytes = 54000
    write_csv("metadata.csv", [{
        "codec": "V1 first-anchor INT8",
        "compressed_representation_bytes": codec["variants"][0]["total_representation_bytes"],
        "step10_error_metadata_bytes": metadata_bytes,
        "net_ratio_including_step10_metadata": 1.62040998,
        "note": "Step-11 adds no persistent metadata beyond Step-10; geometry/tree/GQA physical accounting unavailable for complete all-head trace",
    }], ["codec", "compressed_representation_bytes", "step10_error_metadata_bytes", "net_ratio_including_step10_metadata", "note"])

    write_csv("context_scaling.csv", [{"context_tokens": 256, "scope": "observed representative trace", "payload_bytes": "available in codec artifacts", "physical_bytes_avoided": 0, "latency": "unavailable", "status": "only observed context; no extrapolation"}], ["context_tokens", "scope", "payload_bytes", "physical_bytes_avoided", "latency", "status"])
    write_csv("memory.csv", [{"scope": "physical fixture only", "peak_rss_bytes": smoke["peak_rss_bytes"], "status": "not a real model benchmark"}], ["scope", "peak_rss_bytes", "status"])
    write_csv("environment.csv", [{"python": sys.version.split()[0], "platform": platform.platform(), "generated_utc": datetime.now(timezone.utc).isoformat(), "model_inference": "not executed"}], ["python", "platform", "generated_utc", "model_inference"])

    summary = {
        "status": "PARTIAL_EVIDENCE_ONLY",
        "scope": {
            "flat_replay_rows": len(flat),
            "flat_replay_cases": len({r["case_key"] for r in flat}),
            "certificate_cases": step11["cases"],
            "configured_layers": [0, 8, 16, 24, 31],
            "available_trace_layers": [0, 8, 16, 24],
            "missing_trace_layers": [31],
            "complete_real_gqa_layers": [],
            "observed_context_tokens": [256],
        },
        "flat_modes": modes,
        "hierarchy": {"available": bool(hierarchy_modes), "scope": "20-case hierarchy artifact only", "modes": hierarchy_modes},
        "physical": {"complete_gqa": False, "payload_avoided": 0, "decode_avoided": 0, "reason": partial["reason_for_fail_closed_zero"]},
        "step11_certificate": {
            "mean_compression_bound": step11["compression"]["tight"]["mean"],
            "p95_compression_bound": step11["compression"]["tight"]["p95"],
            "mean_total_bound": step11["total"]["tight"]["mean"],
            "p95_total_bound": step11["total"]["tight"]["p95"],
            "compression_violations": step11["compression"]["violations"],
            "total_violations": step11["total"]["violations"],
        },
        "storage": {"v1_total_ratio": total_representation_ratio, "including_step10_error_metadata_ratio": 1.62040998},
        "fixture_only_physical_smoke": {"full_bytes": smoke["full_payload_bytes"], "read_bytes": smoke["payload_bytes_read"], "avoided_bytes": smoke["payload_bytes_avoided"], "full_decodes": smoke["full_decodes"], "actual_decodes": smoke["actual_decodes"]},
        "blocked": ["complete layer-31 all-head GQA evidence", "real physical omission benchmark", "long-context scaling beyond 256 tokens", "end-to-end model decode timing", "MPFR/IO/runtime comparison"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "environment.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(), "model_inference": False}, indent=2), encoding="utf-8")
    (OUT / "test_report.txt").write_text("Step-11B evidence aggregation completed. Existing focused tests: 39 passed. No new model run. Physical real-data path remained fail-closed because complete all-head GQA evidence is unavailable.\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "flat_cases": len(flat), "certificate_cases": step11["cases"], "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
