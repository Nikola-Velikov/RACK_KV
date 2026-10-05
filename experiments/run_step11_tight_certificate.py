"""Run the Step-11 tight compression certificate on frozen cases."""
from __future__ import annotations

import argparse
import csv
import gc
import math
import time
from pathlib import Path

import numpy as np

from experiments.common import ROOT, load_config, read_json, write_csv, write_json
from experiments.run_anisotropic_shadow import _case_payload
from rack_kv.anisotropic import execute_anisotropic_flat
from rack_kv.certificate import exact_reference_output_mpfr
from rack_kv.compression_certificate import (
    CompressionErrorMetadata,
    build_error_metadata,
    certify_compression_tight,
)
from rack_kv.stage2 import validate_compact_trace


def attention(query, keys, values, scale):
    logits = (keys @ query) * scale
    weights = np.exp(logits - logits.max())
    return (weights[:, None] * values).sum(axis=0) / weights.sum()


def softmax(logits):
    x = logits - logits.max()
    e = np.exp(x)
    return e / e.sum()


def _float_stats(values):
    values = [float(x) for x in values]
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/rack_kv_v2_step11_tight_certificate")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()
    config = load_config(ROOT / "configs/rack_kv_v1.yaml")
    stage4_root = ROOT / ".tmp/stage4_baselines_full"
    stage4 = read_json(ROOT / config["existing_representative"])
    all_rows = [r for r in stage4["case_results"] if r.get("method_name") == "rack_kv"]
    rows = all_rows[args.start_index:] if args.limit is None else all_rows[args.start_index:args.start_index + args.limit]
    args.output = args.output / f"chunk_{args.start_index}_{args.start_index + len(rows)}"

    traces = {}
    ablation_rows, probability_rows, center_rows, mass_rows = [], [], [], []
    value_rows, metadata_rows, total_rows, rigor_rows, timing_rows = [], [], [], [], []
    for ordinal, row in enumerate(rows, 1):
        started = time.perf_counter()
        layer = int(row["layer_index"])
        traces.setdefault(layer, validate_compact_trace(ROOT / config["existing_capture"] / f"llama31_layer{layer}_trace.safetensors", allow_nonzero_layer=True))
        trace = traces[layer]
        visible = int(trace.visible_lengths[int(row["record_index"])])
        kv_local = trace.kv_head_to_local[int(row["kv_head_global"])]
        original_keys = trace.final_keys[kv_local, :visible].float().numpy().astype(np.float64)
        original_values = trace.final_values[kv_local, :visible].float().numpy().astype(np.float64)
        query = trace.queries[int(row["record_index"]), int(row["query_local_index"])].float().numpy().astype(np.float64)
        blocks, recent_keys, recent_values = _case_payload(row, stage4_root)
        reconstructed_keys = np.vstack([b.decode_key_block() for b in blocks] + [recent_keys])
        reconstructed_values = np.vstack([b.decode_value_block() for b in blocks] + [recent_values])
        historical = visible - int(config["recent_window"])
        ranges = [(int(b.header.block_start), int(b.header.block_start + b.block_len)) for b in blocks]
        token_metadata = build_error_metadata(original_keys, reconstructed_keys, original_values, reconstructed_values, precision=int(config["mpfr_precision"]), block_ranges=ranges)
        block_kappa = tuple(max(token_metadata.kappa[start:end]) for start, end in ranges)
        block_eta = tuple(max(token_metadata.eta[start:end]) for start, end in ranges)
        block_kappa_tokens = tuple(bound for (start, end), bound in zip(ranges, block_kappa) for _ in range(end - start))
        block_eta_tokens = tuple(bound for (start, end), bound in zip(ranges, block_eta) for _ in range(end - start))
        zero = type(token_metadata.kappa[0])(0)
        block_metadata = CompressionErrorMetadata(
            block_kappa_tokens + (zero,) * int(config["recent_window"]),
            block_eta_tokens + (zero,) * int(config["recent_window"]),
            max(block_kappa), max(block_eta), len(ranges) * 8,
        )
        reconstructed_output = attention(query, reconstructed_keys, reconstructed_values, float(trace.scaling))
        centers = {
            "zero": np.zeros(reconstructed_values.shape[1], dtype=np.float64),
            "mean": reconstructed_values.mean(axis=0),
            "ohat": reconstructed_output,
        }
        block_cert = certify_compression_tight(query, reconstructed_keys, reconstructed_values, block_metadata, precision=int(config["mpfr_precision"]), attention_scale=float(trace.scaling), centers=centers)
        token_cert = certify_compression_tight(query, reconstructed_keys, reconstructed_values, token_metadata, precision=int(config["mpfr_precision"]), attention_scale=float(trace.scaling), centers=centers) if args.start_index + ordinal <= 20 else None

        exact_output = np.asarray([float(x) for x in exact_reference_output_mpfr(query, original_keys, original_values, precision=int(config["mpfr_precision"]), attention_scale=float(trace.scaling))])
        actual_comp = float(np.linalg.norm(exact_output - reconstructed_output))
        outcome = execute_anisotropic_flat(rank=8, query=query, recent_keys=recent_keys, recent_values=recent_values, historical_blocks=blocks, tolerance=float(config["epsilon"]), precision=int(config["mpfr_precision"]), attention_scale=float(trace.scaling))
        actual_skip = float(np.linalg.norm(outcome.compression_only_output - outcome.output))
        actual_total = float(np.linalg.norm(exact_output - outcome.output))
        skip_bound = float(outcome.certificate.certificate_bound)
        exact_p = softmax((original_keys @ query) * float(trace.scaling))
        phat = softmax((reconstructed_keys @ query) * float(trace.scaling))
        tight_lo = np.asarray([float(x) for x in block_cert.lower_probabilities])
        tight_hi = np.asarray([float(x) for x in block_cert.upper_probabilities])
        interval_violations = int(np.sum(exact_p < tight_lo) + np.sum(exact_p > tight_hi))
        actual_probability_error = float(np.linalg.norm(((exact_p - phat)[:, None] * reconstructed_values).sum(axis=0)))
        comp_bound = float(block_cert.e_comp_tight)
        total_bound = comp_bound + skip_bound
        ablation_rows.append({
            "case_key": row["case_key"], "layer": layer, "actual_compression_error": actual_comp,
            "actual_probability_output_error": actual_probability_error,
            "e_comp_step10": float(block_cert.e_comp_step10), "e_comp_coordinate": float(block_cert.e_comp_coordinate),
            "e_comp_mass": float(block_cert.e_comp_mass), "e_comp_centered": float(block_cert.e_comp_centered),
            "e_comp_tight": comp_bound, "e_value_step10": float(block_cert.e_value_step10),
            "e_value_simplex": float(block_cert.e_value_simplex), "e_value_tight": float(block_cert.e_value_tight),
            "e_probability_step10": float(block_cert.e_probability_step10), "e_probability_coordinate": float(block_cert.e_probability_coordinate),
            "e_probability_mass_zero": float(block_cert.e_probability_mass_zero), "e_probability_zero_center": float(block_cert.e_probability_zero_center),
            "e_probability_mean_center": float(block_cert.e_probability_mean_center), "e_probability_ohat_center": float(block_cert.e_probability_ohat_center),
            "e_probability_tight": float(block_cert.e_probability_tight), "selected_center": block_cert.selected_center,
            "token_e_comp_tight": float(token_cert.e_comp_tight) if token_cert else None,
            "token_e_value_tight": float(token_cert.e_value_tight) if token_cert else None,
            "token_e_probability_tight": float(token_cert.e_probability_tight) if token_cert else None,
            "step10_violation": actual_comp > float(block_cert.e_comp_step10), "tight_violation": actual_comp > comp_bound,
            "interval_violations": interval_violations,
        })
        total_rows.append({"case_key": row["case_key"], "actual_compression_error": actual_comp, "actual_skip_error": actual_skip, "actual_total_error": actual_total, "compression_bound_tight": comp_bound, "skip_bound": skip_bound, "total_bound_tight": total_bound, "total_violation": actual_total > total_bound, "skipped_blocks": len(outcome.skipped_block_starts)})
        center_rows.append({"case_key": row["case_key"], "zero_center": float(block_cert.e_probability_zero_center), "mean_center": float(block_cert.e_probability_mean_center), "ohat_center": float(block_cert.e_probability_ohat_center), "selected_center": block_cert.selected_center})
        mass_rows.append({"case_key": row["case_key"], "mass_T": float(block_cert.mass_t), "coordinate_probability": float(block_cert.e_probability_coordinate), "mass_zero_probability": float(block_cert.e_probability_mass_zero), "actual_probability_output_error": actual_probability_error})
        value_rows.append({"case_key": row["case_key"], "value_step10": float(block_cert.e_value_step10), "value_simplex": float(block_cert.e_value_simplex), "value_tight": float(block_cert.e_value_tight), "actual_compression_error": actual_comp})
        metadata_rows.append({"case_key": row["case_key"], "block_metadata_bytes": len(ranges) * 8, "token_metadata_bytes": historical * 8, "block_count": len(ranges), "historical_tokens": historical})
        rigor_rows.append({"case_key": row["case_key"], "compression_margin": comp_bound - actual_comp, "total_margin": total_bound - actual_total, "probability_margin": float(block_cert.e_probability_tight) - actual_probability_error, "tight_violation": actual_comp > comp_bound, "total_violation": actual_total > total_bound, "interval_violations": interval_violations})
        for i, (lo, hi) in enumerate(zip(block_cert.lower_probabilities, block_cert.upper_probabilities)):
            probability_rows.append({"case_key": row["case_key"], "token": i, "exact_probability_float64": float(exact_p[i]), "lower_tight": str(lo), "upper_tight": str(hi), "contains_exact_float64": bool(exact_p[i] >= float(lo) and exact_p[i] <= float(hi))})
        timing_rows.append({"case_key": row["case_key"], "certificate_seconds": time.perf_counter() - started})
        if ordinal % 25 == 0 or ordinal == len(rows):
            print(f"step11-evaluated {ordinal}/{len(rows)} cases", flush=True)
        del blocks, outcome, token_metadata, block_metadata
        gc.collect()

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "certificate_ablation.csv", ablation_rows)
    write_csv(args.output / "probability_bounds.csv", probability_rows)
    write_csv(args.output / "centering_ablation.csv", center_rows)
    write_csv(args.output / "mass_conservation.csv", mass_rows)
    write_csv(args.output / "value_term.csv", value_rows)
    write_csv(args.output / "block_vs_token.csv", metadata_rows)
    write_csv(args.output / "total_certificate.csv", total_rows)
    write_csv(args.output / "timing.csv", timing_rows)
    write_csv(args.output / "rigor_validation.csv", rigor_rows)
    stats = {"cases": len(rows), "interval_violations": sum(x["interval_violations"] for x in ablation_rows), "compression_violations": sum(x["tight_violation"] for x in ablation_rows), "total_violations": sum(x["total_violation"] for x in total_rows), "selected_centers": {name: sum(x["selected_center"] == name for x in center_rows) for name in ("zero", "mean", "ohat")}}
    write_json(args.output / "summary.json", stats)
    (args.output / "test_report.txt").write_text("Step-11 tight certificate replay completed with 256-bit directed rounding.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
