from __future__ import annotations

import math
from pathlib import Path
import numpy as np

from .common import read_json, verify_frozen_input, write_csv, write_json, sha256

BYTE_FIELDS = ("encoded_key_bytes", "encoded_value_bytes", "scales_bytes", "metadata_bytes",
               "indices_bytes", "block_page_metadata_bytes", "recent_window_bytes")


def summarize(rows):
    groups, identities = {}, set()
    for row in rows:
        identity = (row["case_key"], row["method_name"], row["mode"])
        if identity in identities:
            raise ValueError(f"Duplicate scientific record: {identity}")
        identities.add(identity)
        if sum(row[k] for k in BYTE_FIELDS) != row["total_serialized_bytes"]:
            raise ValueError(f"Disjoint storage mismatch: {identity}")
        if not math.isfinite(row["attention_output_l2_error"]):
            raise ValueError(f"Nonfinite error: {identity}")
        groups.setdefault(row["method_name"] + ":" + row["mode"], []).append(row)
    summary = {}
    for name, items in groups.items():
        errors = np.array([r["attention_output_l2_error"] for r in items])
        entry = {"case_count": len(items), "mean_L2": float(errors.mean()),
                 "median_L2": float(np.median(errors)), "P95_L2": float(np.quantile(errors, .95)),
                 "max_L2": float(errors.max()), "std_L2": float(errors.std()),
                 "mean_serialized_bytes": float(np.mean([r["total_serialized_bytes"] for r in items])),
                 "mean_compression_ratio": float(np.mean([r["compression_ratio_vs_full_kv"] for r in items])),
                 "mean_storage_saving_percent": float(np.mean([r["memory_saving_fraction_vs_full_kv"] * 100 for r in items]))}
        if name == "rack_kv:native":
            entry.update({k: sum(r[k] for r in items) for k in
                          ("candidate_blocks", "certified_skipped_blocks", "false_safe_count", "rigorous_interval_violations", "numerical_fallbacks")})
            entry["weighted_skip_fraction"] = entry["certified_skipped_blocks"] / max(1, entry["candidate_blocks"])
            ratios = [r["certificate_upper_bound"] / max(r["observed_skipping_error"], 1e-30)
                      for r in items if r["certified_skipped_blocks"]]
            entry["mean_certificate_bound_to_error_ratio_certified_cases"] = float(np.mean(ratios)) if ratios else None
        summary[name] = entry
    return summary


def export_full_model(path, output, config, historical=True):
    from rack_kv.stage5 import _aggregate_metric_records, _aggregate_certificate_records, STAGE5_METHOD_IMPLEMENTATION_VERSIONS
    path = Path(path)
    if historical:
        verify_frozen_input(path)
    data = read_json(path)
    if not data.get("full_transformer_layers_executed") or data.get("transformer_layer_count") != 32:
        raise ValueError("Full-model evidence is not a complete 32-layer evaluation.")
    expected_prompts = config["full_model"]["prompts"]
    if data["prompt_names"] != expected_prompts or not data["required_stream_audit"]["all_required_streams_complete"]:
        raise ValueError("Required full-model prompt/stream set is incomplete.")
    metrics, table, token_rows, certs, evidence = {}, [], [], [], []
    for prompt in expected_prompts:
        mp = path.parent / "metric_records" / f"{prompt}.json"
        cp = path.parent / "certificate_records" / f"{prompt}.json"
        if historical:
            evidence.extend([verify_frozen_input(mp), verify_frozen_input(cp)])
        records = read_json(mp)["records_by_method"]
        certs.extend(read_json(cp)["records"])
        audit = data["required_stream_audit"]["prompts"][prompt]
        for method in config["full_model"]["methods"]:
            stream = audit["methods"][method]
            if not all(stream[k] for k in ("all_32_layers_completed", "final_rmsnorm_and_lm_head_completed", "final_logits_exist")):
                raise ValueError(f"Incomplete stream {prompt}/{method}")
            if stream["method_implementation_version"] != STAGE5_METHOD_IMPLEMENTATION_VERSIONS[method]:
                raise ValueError(f"Legacy arithmetic checkpoint {prompt}/{method}")
            items = records[method]
            if len(items) != audit["expected_scored_token_count"]:
                raise ValueError("Token count does not match accepted scoring range")
            aggregate = _aggregate_metric_records(items)
            metrics.setdefault(method, []).extend(items)
            table.append({"prompt": prompt, "method": method, **aggregate})
            token_rows.extend({"prompt": prompt, "method": method, **r} for r in items)
    aggregates = {method: _aggregate_metric_records(rows) for method, rows in metrics.items()}
    for method, aggregate in aggregates.items():
        for key in ("mean_nll", "perplexity", "mean_kl_divergence", "mean_top1_agreement"):
            if not math.isclose(aggregate[key], data["aggregate_method_metrics"][method][key], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Stale quality aggregate: {method}/{key}")
        storage = data["aggregate_storage"][method]
        expected_category_scope = (storage["cumulative_total_serialized_bytes"] if method == "full_kv"
                                   else storage["modified_layers_cumulative_total_serialized_bytes"])
        if sum(storage["category_sums"].values()) != expected_category_scope:
            raise ValueError("Full-model disjoint categories do not match their authoritative layer scope")
        if not storage["byte_accounting_consistent"] or storage["disjoint_category_total_bytes"] != expected_category_scope:
            raise ValueError("Full-model storage consistency flag/scope mismatch")
    if data["aggregate_storage"]["rack_kv_compression_only"]["category_sums"]["certificate_metadata_bytes"]:
        raise ValueError("Compression-only certificate-only metadata must be zero")
    for row in certs:
        if row["skipped_block_starts"] and (not row["mpfr_invoked"] or not row["certificate_bound_text"]):
            raise ValueError("Skip lacks an MPFR proof record")
        if row["rigorous_interval_violation"] or row["false_safe_count"]:
            raise ValueError("Certificate invariant regression in accepted full-model records")
    certificate_summary = _aggregate_certificate_records(certs, num_attention_heads=config["query_heads"], num_key_value_heads=config["kv_heads"])
    if certificate_summary != data["aggregate_certificate"]:
        raise ValueError("Full-model certificate aggregate does not match source records")
    write_csv(output / "full_model_results.csv", table + [{"prompt": "aggregate", "method": m, **a} for m, a in aggregates.items()])
    write_csv(output / "full_model_token_records.csv", token_rows)
    write_json(output / "full_model_certificate_records.json", certs)
    write_json(output / "full_model_storage.json", data["aggregate_storage"])
    write_json(output / "full_model_checkpoint_audit.json", data["required_stream_audit"])
    write_json(output / "full_model_certificate_summary.json", certificate_summary)
    return {"status": "historical_token_records_reaggregated" if historical else "recomputed",
            "metrics": aggregates, "storage": data["aggregate_storage"],
            "certificate": data["aggregate_certificate"], "source_sha256": sha256(path), "evidence": evidence,
            "limitation": "Two short deterministic prompts; no new forward pass in historical mode."}
