"""Post-evaluation diagnostics. No returned value is used by the scientific method."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from rack_kv.codec import CompressedBlock, deserialize_block_container
from rack_kv.certificate import _block_logit_cap_interval
from rack_kv.rigorous import exact_vector
from rack_kv.stage4 import _parse_payload, _bf16_bytes_to_numpy
from rack_kv.stage2 import validate_compact_trace
from .common import ROOT, sha256, write_csv, write_json


def geometry(keys):
    keys = np.asarray(keys, dtype=np.float64)
    if keys.ndim != 2 or len(keys) == 0 or not np.isfinite(keys).all():
        raise ValueError("Geometry requires a finite, nonempty key matrix.")
    centroid = keys.mean(axis=0)
    centered = keys - centroid
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    total = float(energy.sum())
    ratios = energy / total if total else np.zeros_like(energy)
    positive = ratios[ratios > 0]
    effective = float(np.exp(-np.sum(positive * np.log(positive)))) if total else 0.0
    mean = float(singular.mean())
    return {"centroid": centroid.tolist(), "first_token_anchor": keys[0].tolist(),
            "radius_first_anchor": float(np.linalg.norm(keys - keys[0], axis=1).max()),
            "radius_centroid": float(np.linalg.norm(centered, axis=1).max()),
            "singular_values": singular.tolist(), "explained_variance_ratios": ratios.tolist(),
            "effective_rank_energy_entropy": effective,
            "anisotropy_sigma1_over_mean": float(singular[0] / mean) if mean else 0.0,
            "centered_energy": total,
            **{f"residual_energy_rank_{rank}": float(energy[rank:].sum()) for rank in (1, 2, 4, 8, 16)},
            **{f"residual_energy_fraction_rank_{rank}": float(energy[rank:].sum() / total) if total else 0.0
               for rank in (1, 2, 4, 8, 16)}}


def exp_finite(value):
    if value > math.log(np.finfo(float).max):
        return None
    return math.exp(value)


def candidate_diagnostics(query, block, scaling, precision):
    keys, _ = block.decode_block()
    logits = (keys @ query) * scaling
    maximum = float(logits.max())
    log_mass = maximum + math.log(float(np.exp(logits - maximum).sum()))
    cap = _block_logit_cap_interval(exact_vector(query, precision=precision), block,
                                   precision=precision, attention_scale=scaling).upper
    upper = float(np.nextafter(float(cap), np.inf))
    log_u = math.log(block.block_len) + upper
    return {"q_norm": float(np.linalg.norm(query)),
            "anchor_norm": float(np.linalg.norm(block.header.anchor_key.astype(float))),
            "rho": float(block.header.rho_upper), "nu": float(block.header.nu_upper),
            "upper_logit_bound": upper, "upper_logit_bound_mpfr_text": str(cap),
            "actual_max_block_logit": maximum, "logit_bound_slack": upper - maximum,
            "U_m": exp_finite(log_u), "actual_unnormalized_block_mass": exp_finite(log_mass),
            "log_U_m": log_u, "log_actual_block_mass": log_mass,
            "mass_bound_ratio": exp_finite(log_u - log_mass),
            "log_mass_bound_ratio": log_u - log_mass,
            "diagnostic_arithmetic": "FP64 measurements; MPFR cap; never skip authorization"}


def profile_results(rows, payload_root, capture_dir, config, output):
    geometry_rows, candidate_rows, certificate_rows, storage_rows = [], [], [], []
    traces, measured_geometry = {}, {}
    tested = 0
    rack_rows = [r for r in rows if r["method_name"] == "rack_kv"]
    for row_index, row in enumerate(rack_rows):
        layer = row["layer_index"]
        if layer not in traces:
            trace = validate_compact_trace(Path(capture_dir) / f"llama31_layer{layer}_trace.safetensors", allow_nonzero_layer=True)
            if trace.checkpoint_revision != config["revision"]:
                raise ValueError("Trace revision differs from the frozen model.")
            traces[layer] = trace
        trace = traces[layer]
        payload_path = Path(payload_root) / row["payload_relative_path"]
        if sha256(payload_path) != row["payload_sha256"]:
            raise ValueError(f"Serialized payload hash mismatch: {payload_path}")
        header, sections = _parse_payload(payload_path.read_bytes())
        container = deserialize_block_container(sections["historical_block_container"])
        blocks = [container.deserialize_block(i) for i in range(container.block_count)]
        full_keys = np.vstack([b.decode_key_block() for b in blocks])
        full_values = np.vstack([b.decode_value_block() for b in blocks])
        query = trace.queries[row["record_index"], row["query_local_index"]].float().numpy().astype(float)
        starts = row["extra"]["skipped_block_starts"]
        bound, error = row["certificate_upper_bound"], row["observed_skipping_error"]
        identity = {k: row[k] for k in ("layer_index", "query_position", "query_head_global", "kv_head_global", "case_key")}
        certificate_rows.append({**identity, "candidate_blocks": len(blocks),
            "certified_skips": len(starts), "non_skips": len(blocks) - len(starts),
            "prefilter_rejections": 0, "prefilter_status": "not_used_in_representative_evaluator",
            "MPFR_evaluations": row.get("reproduction_observer", {}).get("progressive_returned_steps"),
            "MPFR_evaluations_status": "progressive_returned_steps_including_terminal_state" if "reproduction_observer" in row else "progressive_iteration_count_not_recorded",
            "epsilon": config["epsilon"], "actual_error": error, "rigorous_bound": bound,
            "rigorous_bound_text": row["extra"]["certificate_bound_text"],
            "rigorous_error_upper_text": row["extra"]["rigorous_skip_error_upper_text"],
            "bound_slack": bound - error,
            "bound_to_error_ratio": bound / max(error, config["numerical_floor"]),
            "numerical_floor": config["numerical_floor"], "skipped_block_ids": starts,
            "error_scope": "joint_selected_skipped_set",
            "fallbacks": row["numerical_fallbacks"], "false_safe_count": row["false_safe_count"],
            "rigorous_violation_count": row["rigorous_interval_violations"]})
        offset = 0
        for j, block in enumerate(blocks):
            begin = container.block_offsets[j]
            direct = CompressedBlock.deserialize(container.buffer[begin:begin + container.block_lengths[j]],
                                                 key_dim=container.key_dim, value_dim=container.value_dim)
            kd, vd = direct.decode_block()
            if not (np.array_equal(kd, full_keys[offset:offset + block.block_len]) and
                    np.array_equal(vd, full_values[offset:offset + block.block_len])):
                raise AssertionError("Independent byte-offset decoding mismatch")
            if direct.serialize() != block.serialize():
                raise AssertionError("Serializer roundtrip mismatch")
            offset += block.block_len
            tested += 1
            key = (layer, row["kv_head_global"], block.header.block_start, block.block_len)
            if key not in measured_geometry:
                geometry_id = ":".join(map(str, key))
                start = block.header.block_start
                original = trace.final_keys[trace.kv_head_to_local[row["kv_head_global"]], start:start + block.block_len].float().numpy()
                for representation, data in (("original", original), ("reconstructed", kd)):
                    geometry_rows.append({"geometry_id": geometry_id, "layer_index": layer,
                        "kv_head_global": row["kv_head_global"], "block_index": j,
                        "block_start": start, "block_size": block.block_len,
                        "representation": representation, **geometry(data)})
                measured_geometry[key] = geometry_id
            skipped = block.header.block_start in starts
            candidate_rows.append({**identity, "geometry_id": measured_geometry[key],
                "block_index": j, "block_start": block.header.block_start, "block_size": block.block_len,
                **candidate_diagnostics(query, block, trace.scaling, config["mpfr_precision"]),
                "certified": skipped, "certificate_result": "certified_skip" if skipped else "retained",
                "actual_skipping_error": error if skipped else None,
                "certificate_bound": bound if skipped else None,
                "certificate_bound_ratio": bound / max(error, config["numerical_floor"]) if skipped else None,
                "error_scope": "joint_selected_skipped_set" if skipped else "not_measured_for_unselected_set",
                "payload_sha256": row["payload_sha256"]})
        anchors = sum(b.header.anchor_key.nbytes + b.header.anchor_value.nbytes for b in blocks)
        residuals = sum(b.key_residuals.nbytes + b.value_residuals.nbytes for b in blocks)
        scales = 4 * len(blocks)
        metadata = 16 * len(blocks)
        wrapper = len(payload_path.read_bytes()) - sum(len(v) for v in sections.values())
        recent = len(sections["recent_keys_bf16"]) + len(sections["recent_values_bf16"])
        total = anchors + residuals + scales + metadata + container.index_bytes + container.header_bytes + wrapper + recent
        if total != row["total_serialized_bytes"]:
            raise AssertionError("Disjoint authoritative payload byte accounting failed")
        storage_rows.append({**identity, "scope": "one_query_case_one_KV_head_prefix",
            "full_kv_bytes": row["full_kv_reference_bytes"], "rack_kv_bytes": total,
            "anchor_bytes": anchors, "residual_bytes": residuals, "scale_bytes": scales,
            "metadata_bytes": metadata, "index_bytes": container.index_bytes,
            "container_header_bytes": container.header_bytes + wrapper, "recent_window_bytes": recent,
            "certificate_exclusive_bytes": 0,
            "payload_bytes": anchors + residuals, "payload_bytes_is_subtotal": True,
            "compression_ratio": row["compression_ratio_vs_full_kv"],
            "storage_saving_percent": row["memory_saving_fraction_vs_full_kv"] * 100})
        if (row_index + 1) % 30 == 0:
            print(f"Profiled {row_index + 1}/{len(rack_rows)} cases; {tested} block decisions", flush=True)
    write_csv(output / "geometry_blocks.csv", geometry_rows)
    write_csv(output / "candidate_tightness.csv", candidate_rows)
    write_csv(output / "certificate_cases.csv", certificate_rows)
    write_csv(output / "storage_breakdown.csv", storage_rows)
    random_access = {"blocks_tested": tested, "mismatches": 0, "maximum_difference": 0.0,
                     "scope": "all serialized representative RACK block accesses including repeated prefixes"}
    write_json(output / "random_access.json", random_access)
    return {"random_access": random_access, "geometry_unique_blocks": len(measured_geometry),
            "candidate_count": len(candidate_rows), "profile_only": True}
