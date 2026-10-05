"""Replay the frozen representative benchmark with shadow anisotropic bounds.

This command reads only frozen traces and serialized Stage-4 payloads.  It has
no model-loading or inference path and never changes V1 masks or outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import gmpy2
import numpy as np

from experiments.common import ROOT, inventory, load_config, read_json, sha256, write_csv, write_json
from rack_kv.anisotropic import (
    anisotropic_logit_upper_bound,
    build_anisotropic_summary,
    prepare_anisotropic_case,
    progressive_anisotropic_shadow,
)
from rack_kv.codec import deserialize_block_container
from rack_kv.stage2 import validate_compact_trace
from rack_kv.stage4 import _bf16_bytes_to_numpy, _parse_payload
from rack_kv.types import DecodeSchedule


RANKS = (0, 1, 2, 4, 8, 16)


def _percentile(values, percentile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else None


def _finite_ratio(numerator, denominator, floor=1e-30):
    return float(numerator / max(denominator, floor))


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value):
    if value in (None, ""):
        return None
    return float(value)


def _bool(value):
    return str(value).lower() == "true"


def _section_shapes(header):
    return {str(item["name"]): tuple(int(v) for v in item.get("shape", ())) for item in header["sections"]}


def _case_payload(row, stage4_root):
    path = stage4_root / row["payload_relative_path"]
    if sha256(path) != row["payload_sha256"]:
        raise ValueError(f"payload hash mismatch: {path}")
    header, sections = _parse_payload(path.read_bytes())
    shapes = _section_shapes(header)
    container = deserialize_block_container(sections["historical_block_container"])
    blocks = [container.deserialize_block(index) for index in range(container.block_count)]
    recent_keys = _bf16_bytes_to_numpy(sections["recent_keys_bf16"], shape=shapes["recent_keys_bf16"])
    recent_values = _bf16_bytes_to_numpy(sections["recent_values_bf16"], shape=shapes["recent_values_bf16"])
    return blocks, recent_keys, recent_values


def _actual_outputs(query, blocks, recent_keys, recent_values, skipped_starts, *, precision, scale):
    full_keys = np.vstack([*(block.decode_key_block() for block in blocks), recent_keys])
    full_values = np.vstack([*(block.decode_value_block() for block in blocks), recent_values])
    kept_blocks = [block for block in blocks if block.header.block_start not in skipped_starts]
    kept_keys = np.vstack([*(block.decode_key_block() for block in kept_blocks), recent_keys])
    kept_values = np.vstack([*(block.decode_value_block() for block in kept_blocks), recent_values])
    def attention(keys, values):
        logits = (keys @ query) * scale
        weights = np.exp(logits - logits.max())
        return (weights[:, None] * values).sum(axis=0) / weights.sum()

    full = attention(full_keys, full_values)
    kept = attention(kept_keys, kept_values)
    actual = float(np.linalg.norm(full - kept))
    return actual, None


def _summary_row(geometry_id, summary):
    singular = summary.singular_values
    mean_singular = float(singular.mean()) if singular.size else 0.0
    return {
        "geometry_id": geometry_id,
        "center_mode": summary.center_mode,
        "rank": summary.rank,
        "block_start": summary.block_start,
        "block_size": summary.block_len,
        "head_dim": summary.head_dim,
        "rho_sphere": summary.rho_sphere,
        "rho_res": float(summary.rho_res_upper),
        "rho_res_mpfr_text": summary.rho_res_upper_text,
        "rho_reduction_fraction": 1.0 - float(summary.rho_res_upper) / summary.rho_sphere if summary.rho_sphere else 0.0,
        "effective_rank": summary.effective_rank,
        "anisotropy_sigma1_over_mean": float(singular[0] / mean_singular) if mean_singular else 0.0,
        "coefficient_lower": summary.coefficient_lower.tolist(),
        "coefficient_upper": summary.coefficient_upper.tolist(),
        "basis_sha256": hashlib.sha256(summary.basis.tobytes(order="C")).hexdigest(),
        "metadata_bytes": summary.metadata_bytes,
        "metadata_bytes_per_block_token": summary.metadata_bytes / summary.block_len,
        "build_seconds": summary.build_seconds,
        "basis_selection": "float64_svd_fixed_metadata",
        "residual_proof": "explicit_mpfr_interval_residual",
    }


def _summarize_rank(rank, candidate_rows, case_rows, block_rows, rigor_rows):
    prefix = "old" if rank == 0 else f"aniso_r{rank}"
    slack = [float(row[f"{prefix}_logit_slack"]) for row in candidate_rows]
    mass_gap = [float(row[f"{prefix}_log_mass_ratio"]) for row in candidate_rows]
    selected = [row for row in candidate_rows if row[f"{prefix}_would_skip"]]
    selected_errors = [float(row[f"{prefix}_actual_skip_error"]) for row in selected if row[f"{prefix}_actual_skip_error"] is not None]
    selected_bounds = [float(row[f"{prefix}_rigorous_bound"]) for row in selected]
    ratios = [_finite_ratio(bound, error) for bound, error in zip(selected_bounds, selected_errors)]
    blocks = [row for row in block_rows if int(row["rank"]) == rank and row["center_mode"] == "first_token"]
    cases = [row for row in case_rows if int(row["rank"]) == rank and row["center_mode"] == "first_token"]
    rigor = [row for row in rigor_rows if int(row["rank"]) == rank and row["center_mode"] == "first_token"]
    return {
        "rank": rank,
        "center_mode": "first_token",
        "candidate_decisions": len(candidate_rows),
        "mean_logit_bound_slack": statistics.fmean(slack),
        "median_logit_bound_slack": statistics.median(slack),
        "p95_logit_bound_slack": _percentile(slack, 95),
        "mean_log_mass_ratio": statistics.fmean(mass_gap),
        "median_log_mass_ratio": statistics.median(mass_gap),
        "p95_log_mass_ratio": _percentile(mass_gap, 95),
        "would_be_certified_skips": len(selected),
        "weighted_would_be_skip_fraction": len(selected) / len(candidate_rows),
        "ordinary_multitoken_blocks_certified": sum(int(row["block_size"]) > 1 for row in selected),
        "ordinary_full_8_token_blocks_certified": sum(int(row["block_size"]) == 8 for row in selected),
        "singleton_blocks_certified": sum(int(row["block_size"]) == 1 for row in selected),
        "rigorous_false_safe_count": sum(bool(row["false_safe"]) for row in rigor),
        "rigorous_violation_count": sum(bool(row["rigorous_violation"]) for row in rigor),
        "mpfr_numerical_fallback_count": sum(bool(row["numerical_fallback"]) for row in cases),
        "mean_certificate_bound_to_actual_error_ratio": statistics.fmean(ratios) if ratios else None,
        "metadata_size_bytes_unique_blocks": sum(int(row["metadata_bytes"]) for row in blocks),
        "estimated_metadata_bytes_per_kv_token": (
            sum(int(row["metadata_bytes"]) for row in blocks) / sum(int(row["block_size"]) for row in blocks)
            if blocks else None
        ),
        "summary_build_time_seconds": sum(float(row["build_seconds"]) for row in blocks),
        "mean_per_query_certificate_seconds": statistics.fmean(float(row["evaluation_seconds"]) for row in cases),
        "p95_per_query_certificate_seconds": _percentile([float(row["evaluation_seconds"]) for row in cases], 95),
    }


def _write_plots(output, candidate_rows, rank_summaries, block_rows, geometry_lookup):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = ["sphere", "r1", "r2", "r4", "r8", "r16"]
    prefixes = ["old", "aniso_r1", "aniso_r2", "aniso_r4", "aniso_r8", "aniso_r16"]

    def save(name):
        plt.tight_layout()
        plt.savefig(plot_dir / f"{name}.pdf")
        plt.close()

    plt.figure(figsize=(7, 4))
    plt.boxplot([[float(row[f"{p}_logit_slack"]) for row in candidate_rows] for p in prefixes], tick_labels=labels, showfliers=False)
    plt.ylabel("Certified logit cap - exact maximum")
    plt.xlabel("Directional rank")
    save("01_logit_slack_by_rank")

    plt.figure(figsize=(7, 4))
    plt.boxplot([[float(row[f"{p}_log_mass_ratio"]) for row in candidate_rows] for p in prefixes], tick_labels=labels, showfliers=False)
    plt.ylabel("log(U_m / true block mass)")
    plt.xlabel("Directional rank")
    save("02_log_mass_ratio_by_rank")

    plt.figure(figsize=(7, 4))
    plt.bar(labels, [100 * item["weighted_would_be_skip_fraction"] for item in rank_summaries])
    plt.ylabel("Would-be certified block decisions (%)")
    save("03_skip_fraction_by_rank")

    plt.figure(figsize=(7, 4))
    plt.bar(labels, [item["ordinary_multitoken_blocks_certified"] for item in rank_summaries])
    plt.ylabel("Certified multi-token blocks")
    save("04_multitoken_skips_by_rank")

    ratios = []
    for row in candidate_rows:
        if row["aniso_r4_would_skip"] and row["aniso_r4_actual_skip_error"] is not None:
            ratios.append(_finite_ratio(float(row["aniso_r4_rigorous_bound"]), float(row["aniso_r4_actual_skip_error"])))
    plt.figure(figsize=(7, 4))
    if ratios:
        plt.hist(np.log10(np.asarray(ratios)), bins=min(30, max(5, len(ratios))))
    plt.xlabel("log10(certificate bound / observed error), rank 4")
    plt.ylabel("Count")
    save("05_certificate_bound_ratio")

    rank4_blocks = [row for row in block_rows if int(row["rank"]) == 4 and row["center_mode"] == "first_token"]
    plt.figure(figsize=(5, 5))
    plt.scatter([float(r["rho_sphere"]) for r in rank4_blocks], [float(r["rho_res"]) for r in rank4_blocks], s=10, alpha=.55)
    maximum = max([float(r["rho_sphere"]) for r in rank4_blocks] + [1.0])
    plt.plot([0, maximum], [0, maximum], color="black", linewidth=.8, linestyle="--")
    plt.xlabel("Spherical radius")
    plt.ylabel("Rank-4 residual radius")
    save("06_sphere_vs_rank4_radius")

    improvement_by_geometry = {}
    for row in candidate_rows:
        improvement_by_geometry.setdefault(row["geometry_id"], []).append(float(row["old_logit_slack"]) - float(row["aniso_r4_logit_slack"]))
    xs, ys = [], []
    for geometry_id, improvements in improvement_by_geometry.items():
        if geometry_id in geometry_lookup:
            xs.append(float(geometry_lookup[geometry_id]["effective_rank_energy_entropy"]))
            ys.append(statistics.fmean(improvements))
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, s=10, alpha=.55)
    plt.xlabel("Effective rank")
    plt.ylabel("Mean rank-4 cap improvement")
    save("07_effective_rank_vs_improvement")

    plt.figure(figsize=(6, 4))
    plt.plot([item["metadata_size_bytes_unique_blocks"] / 1024 for item in rank_summaries],
             [item["mean_logit_bound_slack"] for item in rank_summaries], marker="o")
    for label, item in zip(labels, rank_summaries):
        plt.annotate(label, (item["metadata_size_bytes_unique_blocks"] / 1024, item["mean_logit_bound_slack"]))
    plt.xlabel("Unique-block metadata (KiB)")
    plt.ylabel("Mean logit-bound slack")
    save("08_metadata_vs_tightness")

    layers = sorted({int(row["layer"]) for row in candidate_rows})
    plt.figure(figsize=(7, 4))
    for prefix, label in (("aniso_r2", "rank 2"), ("aniso_r4", "rank 4"), ("aniso_r8", "rank 8")):
        values = []
        for layer in layers:
            subset = [row for row in candidate_rows if int(row["layer"]) == layer]
            values.append(statistics.fmean(float(row["old_logit_slack"]) - float(row[f"{prefix}_logit_slack"]) for row in subset))
        plt.plot(layers, values, marker="o", label=label)
    plt.xlabel("Transformer layer")
    plt.ylabel("Mean reduction in logit slack")
    plt.legend()
    save("09_layer_improvement")

    plt.figure(figsize=(5, 5))
    plt.scatter([float(row["old_logit_slack"]) for row in candidate_rows],
                [float(row["aniso_r4_logit_slack"]) for row in candidate_rows], s=6, alpha=.35)
    maximum = max(float(row["old_logit_slack"]) for row in candidate_rows)
    plt.plot([0, maximum], [0, maximum], color="black", linewidth=.8, linestyle="--")
    plt.xlabel("Sphere logit slack")
    plt.ylabel("Rank-4 logit slack")
    save("10_sphere_vs_rank4_scatter")


def run(args):
    os.chdir(ROOT)
    config = load_config(args.config)
    ranks = tuple(args.ranks)
    if ranks != tuple(sorted(set(ranks))) or 0 not in ranks:
        raise ValueError("ranks must be unique, sorted, and include rank zero.")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    baseline = ROOT / "results/rack_kv_v1_baseline"
    stage4_root = ROOT / ".tmp/stage4_baselines_full"
    stage4 = read_json(ROOT / config["existing_representative"])
    rows = [row for row in stage4["case_results"] if row["method_name"] == "rack_kv"]
    rows.sort(key=lambda row: (row["layer_index"], row["query_position"], row["query_local_index"]))
    if args.limit_cases:
        rows = rows[: args.limit_cases]
    old_candidates = _read_csv(baseline / "candidate_tightness.csv")
    old_by_key = {(row["case_key"], int(row["block_start"])): row for row in old_candidates}
    geometry_rows = _read_csv(baseline / "geometry_blocks.csv")
    geometry_lookup = {
        row["geometry_id"]: row for row in geometry_rows if row["representation"] == "reconstructed"
    }
    traces = {}
    summary_cache = {}
    block_summary_rows = []
    candidate_rows = []
    case_rows = []
    rigor_rows = []
    rank0_mismatches = []
    checkpoint_dir = output / "case_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    known_block_rows = set()
    initial_v1_audit = {name: sha256(ROOT / name) for name in read_json(ROOT / "configs/rack_kv_v1.lock.json")["source_files"]}

    for case_index, row in enumerate(rows, 1):
        checkpoint_path = checkpoint_dir / f"{row['case_key']}.json"
        if args.resume and checkpoint_path.exists():
            checkpoint = read_json(checkpoint_path)
            candidate_rows.extend(checkpoint["candidate_rows"])
            case_rows.extend(checkpoint["case_rows"])
            rigor_rows.extend(checkpoint["rigor_rows"])
            for block_row in checkpoint["block_summary_rows"]:
                key = (block_row["geometry_id"], int(block_row["rank"]), block_row["center_mode"])
                if key not in known_block_rows:
                    block_summary_rows.append(block_row)
                    known_block_rows.add(key)
            if case_index % 10 == 0 or case_index == len(rows):
                print(f"Shadow resume {case_index}/{len(rows)} cases; {len(candidate_rows)} candidate decisions", flush=True)
            continue
        candidate_begin = len(candidate_rows)
        case_begin = len(case_rows)
        rigor_begin = len(rigor_rows)
        block_begin = len(block_summary_rows)
        layer = int(row["layer_index"])
        if layer not in traces:
            traces[layer] = validate_compact_trace(
                ROOT / config["existing_capture"] / f"llama31_layer{layer}_trace.safetensors",
                allow_nonzero_layer=True,
            )
        trace = traces[layer]
        query = trace.queries[int(row["record_index"]), int(row["query_local_index"])].float().numpy().astype(np.float64)
        blocks, recent_keys, recent_values = _case_payload(row, stage4_root)
        prepared_case = prepare_anisotropic_case(
            query=query,
            recent_keys=recent_keys,
            recent_values=recent_values,
            historical_blocks=blocks,
            precision=int(config["mpfr_precision"]),
            attention_scale=float(trace.scaling),
        )
        case_candidates = [old_by_key[(row["case_key"], block.header.block_start)] for block in blocks]
        output_rows = []
        for block, old in zip(blocks, case_candidates):
            geometry_id = old["geometry_id"]
            output_rows.append({
                "case_key": row["case_key"],
                "layer": layer,
                "query_position": int(row["query_position"]),
                "query_head": int(row["query_head_global"]),
                "kv_head": int(row["kv_head_global"]),
                "block_index": int(old["block_index"]),
                "block_start": block.header.block_start,
                "block_size": block.block_len,
                "geometry_id": geometry_id,
                "old_sphere_logit_upper": float(old["upper_logit_bound"]),
                "actual_max_scaled_logit": float(old["actual_max_block_logit"]),
                "old_logit_slack": float(old["logit_bound_slack"]),
                "old_log_U_m": float(old["log_U_m"]),
                "actual_log_block_mass": float(old["log_actual_block_mass"]),
                "old_log_mass_ratio": float(old["log_mass_bound_ratio"]),
                "rho_sphere": float(old["rho"]),
            })
        rank_results = {}
        actual_output_cache = {}
        for rank in ranks:
            summaries = []
            for block, old in zip(blocks, case_candidates):
                key = (old["geometry_id"], rank, "first_token")
                if key not in summary_cache:
                    summary_cache[key] = build_anisotropic_summary(
                        block,
                        rank=rank,
                        center_mode="first_token",
                        precision=int(config["mpfr_precision"]),
                    )
                    if key not in known_block_rows:
                        block_summary_rows.append(_summary_row(old["geometry_id"], summary_cache[key]))
                        known_block_rows.add(key)
                summaries.append(summary_cache[key])
            result = progressive_anisotropic_shadow(
                query=query,
                recent_keys=recent_keys,
                recent_values=recent_values,
                historical_blocks=blocks,
                summaries=summaries,
                tolerance=float(config["epsilon"]),
                schedule=DecodeSchedule.LARGEST_U_TIMES_NU,
                precision=int(config["mpfr_precision"]),
                attention_scale=float(trace.scaling),
                prepared_case=prepared_case,
            )
            rank_results[rank] = result
            actual_error = None
            rigorous_error = None
            violation = False
            false_safe = False
            if result.skipped_block_starts:
                skip_key = tuple(sorted(result.skipped_block_starts))
                if skip_key not in actual_output_cache:
                    actual_output_cache[skip_key] = _actual_outputs(
                        query,
                        blocks,
                        recent_keys,
                        recent_values,
                        set(result.skipped_block_starts),
                        precision=int(config["mpfr_precision"]),
                        scale=float(trace.scaling),
                    )
                actual_error, rigorous_error = actual_output_cache[skip_key]
                violation = actual_error > float(result.certificate_bound)
                false_safe = actual_error > float(config["epsilon"])
            rigor_rows.append({
                "case_key": row["case_key"],
                "rank": rank,
                "center_mode": "first_token",
                "skipped_block_count": len(result.skipped_block_starts),
                "actual_skip_error": actual_error,
                "rigorous_actual_error_upper": float(rigorous_error) if rigorous_error is not None else None,
                "theorem_bound": float(result.certificate_bound),
                "validation_arithmetic": "stable_float64_observed_error_vs_mpfr_rigorous_theorem_bound",
                "rigorous_violation": violation,
                "false_safe": false_safe,
                "numerical_fallback": result.numerical_fallback_used,
            })
            case_rows.append({
                "case_key": row["case_key"],
                "rank": rank,
                "center_mode": "first_token",
                "would_certify": result.would_certify,
                "decoded_blocks": len(result.decoded_block_starts),
                "skipped_blocks": len(result.skipped_block_starts),
                "iterations": result.iterations,
                "certificate_bound": float(result.certificate_bound),
                "certificate_bound_mpfr_text": str(result.certificate_bound),
                "actual_skip_error": actual_error,
                "rigorous_actual_error_upper": float(rigorous_error) if rigorous_error is not None else None,
                "evaluation_seconds": result.evaluation_seconds,
                "numerical_fallback": result.numerical_fallback_used,
            })
            decision_by_start = {decision.block_start: decision for decision in result.decisions}
            for out, block, summary, old in zip(output_rows, blocks, summaries, case_candidates):
                cap = anisotropic_logit_upper_bound(
                    query,
                    summary,
                    precision=int(config["mpfr_precision"]),
                    attention_scale=float(trace.scaling),
                ).upper
                upper = float(np.nextafter(float(cap), np.inf))
                log_u = math.log(block.block_len) + upper
                prefix = "old" if rank == 0 else f"aniso_r{rank}"
                decision = decision_by_start[block.header.block_start]
                out[f"{prefix}_logit_upper"] = upper
                out[f"{prefix}_logit_slack"] = upper - float(old["actual_max_block_logit"])
                out[f"{prefix}_log_U_m"] = log_u
                out[f"{prefix}_log_mass_ratio"] = log_u - float(old["log_actual_block_mass"])
                out[f"{prefix}_rigorous_bound"] = float(decision.certificate_bound)
                out[f"{prefix}_rigorous_bound_mpfr_text"] = str(decision.certificate_bound)
                out[f"{prefix}_would_skip"] = decision.would_skip
                out[f"{prefix}_actual_skip_error"] = actual_error if decision.would_skip else None
                out[f"rho_res_r{rank}"] = float(summary.rho_res_upper)
                if rank == 0:
                    if str(cap) != old["upper_logit_bound_mpfr_text"]:
                        rank0_mismatches.append({"case_key": row["case_key"], "block_start": block.header.block_start,
                                                 "old": old["upper_logit_bound_mpfr_text"], "new": str(cap)})
                rigor_rows.append({
                    "case_key": row["case_key"],
                    "rank": rank,
                    "center_mode": "first_token",
                    "block_start": block.header.block_start,
                    "record_scope": "block_logit_and_mass",
                    "exact_max_scaled_logit": float(old["actual_max_block_logit"]),
                    "rigorous_logit_upper": upper,
                    "logit_sound": float(old["actual_max_block_logit"]) <= upper,
                    "true_log_mass": float(old["log_actual_block_mass"]),
                    "rigorous_log_mass_upper": log_u,
                    "mass_sound": float(old["log_actual_block_mass"]) <= log_u,
                    "rigorous_violation": False,
                    "false_safe": False,
                    "numerical_fallback": False,
                })
        expected_old = set(row["extra"]["skipped_block_starts"])
        observed_old = set(rank_results[0].skipped_block_starts)
        if expected_old != observed_old:
            rank0_mismatches.append({"case_key": row["case_key"], "old_skips": sorted(expected_old), "new_skips": sorted(observed_old)})
        candidate_rows.extend(output_rows)
        write_json(checkpoint_path, {
            "schema": "rack_kv_v2_anisotropic_case_checkpoint_1",
            "case_key": row["case_key"],
            "candidate_rows": candidate_rows[candidate_begin:],
            "case_rows": case_rows[case_begin:],
            "rigor_rows": rigor_rows[rigor_begin:],
            "block_summary_rows": block_summary_rows[block_begin:],
        })
        if case_index % 10 == 0 or case_index == len(rows):
            print(f"Shadow replay {case_index}/{len(rows)} cases; {len(candidate_rows)} candidate decisions", flush=True)

    if rank0_mismatches:
        write_json(output / "rank0_mismatches.json", rank0_mismatches)
        raise AssertionError(f"rank zero failed to recover V1 in {len(rank0_mismatches)} records")
    if len(rows) == 330 and len(candidate_rows) != 6750:
        raise AssertionError(f"expected 6750 decisions, found {len(candidate_rows)}")
    for record in rigor_rows:
        if record.get("record_scope") == "block_logit_and_mass" and not (record["logit_sound"] and record["mass_sound"]):
            raise AssertionError("rigorous anisotropic block bound failed")

    rank_summaries = [_summarize_rank(rank, candidate_rows, case_rows, block_summary_rows, rigor_rows) for rank in ranks]
    by_rank = {item["rank"]: item for item in rank_summaries}
    monotonic = {}
    for low, high in zip(ranks, ranks[1:]):
        lp = "old" if low == 0 else f"aniso_r{low}"
        hp = f"aniso_r{high}"
        cap_ok = sum(float(row[f"{hp}_logit_upper"]) <= float(row[f"{lp}_logit_upper"]) for row in candidate_rows)
        cert_ok = sum(float(row[f"{hp}_rigorous_bound"]) <= float(row[f"{lp}_rigorous_bound"]) for row in candidate_rows)
        monotonic[f"r{high}_le_r{low}"] = {
            "logit_cap_fraction": cap_ok / len(candidate_rows),
            "decision_bound_fraction": cert_ok / len(candidate_rows),
            "nonmonotonic_logit_cap_cases": len(candidate_rows) - cap_ok,
            "nonmonotonic_decision_bound_cases": len(candidate_rows) - cert_ok,
        }
    best = min(rank_summaries[1:], key=lambda item: (item["mean_log_mass_ratio"], item["metadata_size_bytes_unique_blocks"]))
    final_v1_audit = {name: sha256(ROOT / name) for name in initial_v1_audit}
    v1_unchanged = initial_v1_audit == final_v1_audit
    summary = {
        "schema": "rack_kv_v2_anisotropic_shadow_1",
        "scientific_scope": "Shadow-only replacement of the V1 spherical block mass cap on the frozen 6,750 representative decisions.",
        "model_inference_calls": 0,
        "trace_recaptures": 0,
        "attention_outputs_modified": 0,
        "epsilon": config["epsilon"],
        "mpfr_precision": config["mpfr_precision"],
        "case_count": len(rows),
        "candidate_decision_count": len(candidate_rows),
        "unique_block_count": len({row["geometry_id"] for row in candidate_rows}),
        "ranks": list(ranks),
        "rank_zero_exact_v1_mismatch_count": 0,
        "rank_summary": rank_summaries,
        "best_tightness_rank": best["rank"],
        "monotonicity": monotonic,
        "v1_frozen_sources_unchanged": v1_unchanged,
        "centroid_ablation": "not_run_in_primary_command",
        "shared_basis": {
            "status": "interface_implemented_not_evaluated",
            "reason": "The frozen representative corpus contains one evaluation trace and no independent calibration trace; evaluation blocks were not reused to fit a shared basis.",
        },
        "limitations": [
            "Counterfactual shadow decisions do not alter attention execution.",
            "Per-block float64 bases are diagnostic metadata, not a production storage format.",
            "No physical GQA block omission or latency claim is made.",
        ],
    }
    if not v1_unchanged:
        raise AssertionError("frozen V1 source changed during shadow replay")
    write_csv(output / "candidate_comparison.csv", candidate_rows)
    write_csv(output / "block_summaries.csv", block_summary_rows)
    write_csv(output / "rank_summary.csv", rank_summaries)
    write_csv(output / "rigor_validation.csv", rigor_rows)
    write_csv(output / "case_summary.csv", case_rows)
    write_json(output / "summary.json", summary)
    write_json(output / "v1_nonregression.json", {"unchanged": v1_unchanged, "source_hashes": final_v1_audit})
    _write_plots(output, candidate_rows, rank_summaries, block_summary_rows, geometry_lookup)
    manifest = {
        "schema": "rack_kv_v2_anisotropic_shadow_manifest_1",
        "created_utc_epoch": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "command": [sys.executable, "-m", "experiments.run_anisotropic_shadow", *sys.argv[1:]],
        "inputs": {
            "v1_config_sha256": sha256(args.config),
            "v1_lock_sha256": sha256(ROOT / "configs/rack_kv_v1.lock.json"),
            "stage4_results_sha256": sha256(ROOT / config["existing_representative"]),
            "candidate_tightness_sha256": sha256(baseline / "candidate_tightness.csv"),
        },
        "files": inventory(output),
    }
    write_json(output / "manifest.json", manifest)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=ROOT / "configs/rack_kv_v1.yaml")
    result.add_argument("--output", type=Path, default=ROOT / "results/rack_kv_v2_anisotropic_shadow")
    result.add_argument("--ranks", type=int, nargs="+", default=list(RANKS))
    result.add_argument("--limit-cases", type=int)
    result.add_argument("--resume", action="store_true")
    return result


def main():
    args = parser().parse_args()
    summary = run(args)
    print(json.dumps({"output": str(args.output), "rank_summary": summary["rank_summary"]}, indent=2))


if __name__ == "__main__":
    main()
