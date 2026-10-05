"""Validate the rigorous compression and composed local error certificates."""
from __future__ import annotations

import argparse
import csv
import gc
import math
from pathlib import Path

import numpy as np

from experiments.common import ROOT, load_config, read_json, write_csv, write_json
from experiments.run_anisotropic_shadow import _case_payload
from rack_kv.anisotropic import execute_anisotropic_flat
from rack_kv.certificate import exact_reference_output_mpfr
from rack_kv.compression_certificate import build_error_metadata, certify_compression, CompressionErrorMetadata
from rack_kv.stage2 import validate_compact_trace


def as_float(value):
    return float(value)


def attention(query, keys, values, scale):
    logits = (keys @ query) * scale
    weights = np.exp(logits - logits.max())
    return (weights[:, None] * values).sum(axis=0) / weights.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/rack_kv_v2_step10_compression_certificate")
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
    compression_rows, metadata_rows, breakdown_rows, total_rows, rigor_rows, probability_rows = [], [], [], [], [], []
    for ordinal, row in enumerate(rows, 1):
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
        metadata = build_error_metadata(original_keys, reconstructed_keys, original_values, reconstructed_values, precision=int(config["mpfr_precision"]), block_ranges=ranges)
        block_kappa = tuple(max(metadata.kappa[start:end]) for start, end in ranges)
        block_eta = tuple(max(metadata.eta[start:end]) for start, end in ranges)
        # Expand one outward-safe block maximum over every token in that
        # block. This is deliberately looser than token metadata and is the
        # representation used by the practical block certificate.
        block_kappa_per_token = tuple(
            bound for (start, end), bound in zip(ranges, block_kappa)
            for _ in range(end - start)
        )
        block_eta_per_token = tuple(
            bound for (start, end), bound in zip(ranges, block_eta)
            for _ in range(end - start)
        )
        block_metadata = CompressionErrorMetadata(
            block_kappa_per_token + (type(metadata.kappa[0])(0),) * int(config["recent_window"]),
            block_eta_per_token + (type(metadata.eta[0])(0),) * int(config["recent_window"]),
            block_kappa,
            block_eta,
            len(ranges) * 8,
        )
        # Exact recent entries are deliberately zero-error metadata.
        block_metadata = CompressionErrorMetadata(block_metadata.kappa[:-int(config["recent_window"])] + (type(metadata.kappa[0])(0),) * int(config["recent_window"]), block_metadata.eta[:-int(config["recent_window"])] + (type(metadata.eta[0])(0),) * int(config["recent_window"]), block_metadata.block_kappa, block_metadata.block_eta, block_metadata.metadata_bytes)
        # Token-level metadata is an ablation sample. Keep the sample global
        # across chunked runs rather than restarting it in every chunk.
        token_cert = certify_compression(query, reconstructed_keys, reconstructed_values, metadata, precision=int(config["mpfr_precision"]), attention_scale=float(trace.scaling)) if args.start_index + ordinal <= 20 else None
        block_cert = certify_compression(query, reconstructed_keys, reconstructed_values, block_metadata, precision=int(config["mpfr_precision"]), attention_scale=float(trace.scaling))
        exact_output = np.asarray([float(x) for x in exact_reference_output_mpfr(query, original_keys, original_values, precision=int(config["mpfr_precision"]), attention_scale=float(trace.scaling))])
        reconstructed_output = attention(query, reconstructed_keys, reconstructed_values, float(trace.scaling))
        actual_comp = float(np.linalg.norm(exact_output - reconstructed_output))
        outcome = execute_anisotropic_flat(rank=8, query=query, recent_keys=recent_keys, recent_values=recent_values, historical_blocks=blocks, tolerance=float(config["epsilon"]), precision=int(config["mpfr_precision"]), attention_scale=float(trace.scaling))
        kept = outcome.output
        actual_total = float(np.linalg.norm(exact_output - kept))
        actual_skip = float(np.linalg.norm(outcome.compression_only_output - kept))
        skip_bound = float(outcome.certificate.certificate_bound)
        compression_rows.append({"case_key": row["case_key"], "layer": layer, "actual_compression_error": actual_comp, "block_cert": as_float(block_cert.e_comp), "token_cert": as_float(token_cert.e_comp) if token_cert else None, "block_e_value": as_float(block_cert.e_value), "block_e_probability": as_float(block_cert.e_probability), "token_e_value": as_float(token_cert.e_value) if token_cert else None, "token_e_probability": as_float(token_cert.e_probability) if token_cert else None, "block_violation": actual_comp > as_float(block_cert.e_comp), "token_violation": actual_comp > as_float(token_cert.e_comp) if token_cert else None})
        total_rows.append({"case_key": row["case_key"], "actual_compression_error": actual_comp, "actual_skip_error": actual_skip, "actual_total_error": actual_total, "compression_bound": as_float(block_cert.e_comp), "skip_bound": skip_bound, "total_bound": as_float(block_cert.e_comp) + skip_bound, "total_violation": actual_total > as_float(block_cert.e_comp) + skip_bound, "skipped_blocks": len(outcome.skipped_block_starts)})
        metadata_rows.append({"case_key": row["case_key"], "block_count": len(ranges), "block_metadata_bytes": len(ranges) * 8, "token_metadata_bytes": historical * 8, "block_bytes_per_token": (len(ranges) * 8) / max(historical, 1), "token_bytes_per_token": 8.0})
        breakdown_rows.append({"case_key": row["case_key"], "block_e_value": as_float(block_cert.e_value), "block_e_probability": as_float(block_cert.e_probability), "block_probability_fallback": as_float(block_cert.e_probability_fallback), "token_e_value": as_float(token_cert.e_value) if token_cert else None, "token_e_probability": as_float(token_cert.e_probability) if token_cert else None})
        rigor_rows.append({"case_key": row["case_key"], "block_margin": as_float(block_cert.e_comp) - actual_comp, "token_margin": as_float(token_cert.e_comp) - actual_comp if token_cert else None, "skip_margin": skip_bound - actual_skip, "total_margin": as_float(block_cert.e_comp) + skip_bound - actual_total, "block_violation": actual_comp > as_float(block_cert.e_comp), "token_violation": actual_comp > as_float(token_cert.e_comp) if token_cert else None, "total_violation": actual_total > as_float(block_cert.e_comp) + skip_bound})
        for index, (lo, hi, dist) in enumerate(zip(token_cert.lower_probabilities, token_cert.upper_probabilities, token_cert.probability_distances)) if token_cert else ():
            if index < min(visible, 64):
                probability_rows.append({"case_key": row["case_key"], "token": index, "lower": str(lo), "upper": str(hi), "distance": str(dist)})
        if ordinal % 25 == 0 or ordinal == len(rows):
            print(f"step10-evaluated {ordinal}/{len(rows)} cases", flush=True)
        del blocks, outcome, metadata, block_metadata
        gc.collect()
    args.output.mkdir(parents=True, exist_ok=True)
    token_rows = [r for r in compression_rows if r["token_cert"] is not None]
    write_json(args.output / "summary.json", {"scope": "Frozen 330-case trace validation; no model inference.", "cases": len(rows), "precision": int(config["mpfr_precision"]), "recent_window": int(config["recent_window"]), "codec": "V1 first-token INT8", "block_certificate": {"violations": sum(bool(r["block_violation"]) for r in compression_rows), "mean_bound": float(np.mean([r["block_cert"] for r in compression_rows])), "p95_bound": float(np.percentile([r["block_cert"] for r in compression_rows], 95))}, "token_certificate": {"sample_cases": len(token_rows), "violations": sum(bool(r["token_violation"]) for r in token_rows), "mean_bound": float(np.mean([r["token_cert"] for r in token_rows])) if token_rows else None, "p95_bound": float(np.percentile([r["token_cert"] for r in token_rows], 95)) if token_rows else None}, "total_certificate": {"violations": sum(bool(r["total_violation"]) for r in total_rows)}})
    write_csv(args.output / "compression_cases.csv", compression_rows)
    write_csv(args.output / "error_metadata.csv", metadata_rows)
    write_csv(args.output / "probability_intervals.csv", probability_rows)
    write_csv(args.output / "bound_breakdown.csv", breakdown_rows)
    write_csv(args.output / "block_vs_token_metadata.csv", metadata_rows)
    write_csv(args.output / "total_certificate.csv", total_rows)
    write_csv(args.output / "rigor_validation.csv", rigor_rows)
    write_csv(args.output / "metadata_accounting.csv", metadata_rows)
    (args.output / "test_report.txt").write_text("Step-10 replay completed with 256-bit directed-rounding bounds; focused tests run separately.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
