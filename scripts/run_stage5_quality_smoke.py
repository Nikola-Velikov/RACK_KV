from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rack_kv.stage5 import (  # noqa: E402
    STAGE5_EXPERIMENT_SCOPE_LAYER0_SAFETY_VALIDATION,
    STAGE5_CERTIFICATE_DIR,
    STAGE5_CHECKPOINT_DIR,
    STAGE5_DEFAULT_CAPTURE_DIR,
    STAGE5_DEFAULT_OUTPUT_DIR,
    STAGE5_DEFAULT_REVIEW_ZIP,
    STAGE5_DEFAULT_TENSOR_CACHE_DIR,
    STAGE5_MANIFEST_JSON,
    STAGE5_METHODS,
    STAGE5_METHOD_VERSION,
    STAGE5_METRIC_DIR,
    STAGE5_MODIFIED_LAYERS,
    STAGE5_PROMPT_CORPUS_VERSION,
    STAGE5_PROMPT_DIR,
    STAGE5_PROMPT_CHECKPOINT_SCHEMA,
    STAGE5_REPORT_MD,
    STAGE5_RESULT_VERSION,
    STAGE5_STORAGE_ACCOUNTING_SCHEMA,
    STAGE5_RESULTS_JSON,
    STAGE5_REVIEW_SOURCE_FILES,
    STAGE5_RUN_SCHEMA,
    STAGE5_SCORE_START,
    STAGE5_STREAM_CHECKPOINT_SCHEMA,
    STAGE5_TEST_LOG,
    STAGE5_PROVENANCE_DIR,
    Stage5ExecutionError,
    _aggregate_certificate_records,
    _aggregate_metric_records,
    _json_roundtrip,
    _method_storage_summary,
    _sha256_file,
    _source_snapshot_sha256,
    _stage5_method_impl_version,
    _stage5_wrapper_total_bytes,
    _write_json_atomic,
    _write_text_atomic,
    run_stage5_quality_smoke,
    stage5_dependency_versions,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or rebuild Stage 5 quality artifacts.")
    parser.add_argument("--output-dir", default=str(STAGE5_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--review-zip", default=str(STAGE5_DEFAULT_REVIEW_ZIP))
    parser.add_argument("--capture-dir", default=str(STAGE5_DEFAULT_CAPTURE_DIR))
    parser.add_argument("--tensor-cache-dir", default=str(STAGE5_DEFAULT_TENSOR_CACHE_DIR))
    parser.add_argument("--repo-id", default="NousResearch/Meta-Llama-3.1-8B")
    parser.add_argument("--repo-revision", default="1f47e50cdbe801ad8a5174156ec3a0655108fb9f")
    parser.add_argument("--allow-insecure-tls", action="store_true")
    parser.add_argument("--max-rss-gb", type=float, default=8.0)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--prompt-names", default=None, help="Comma-separated Stage 5 prompt names to execute.")
    parser.add_argument("--method-names", default=None, help="Comma-separated Stage 5 method names to execute.")
    parser.add_argument("--prompt-token-count", type=int, default=None, help="Optional deterministic token-count override for profiling or reduced runs.")
    parser.add_argument("--profile-progress-every", type=int, default=None, help="Optional per-stream token progress interval for profiling output.")
    parser.add_argument("--max-layer-index", type=int, default=None, help="Optional inclusive maximum layer index for validation runs.")
    parser.add_argument("--skip-full-kv-equivalence", action="store_true", help="Skip the short stock-vs-custom full-KV equivalence regression.")
    parser.add_argument(
        "--rebuild-layer0-safety-package-from-checkpoints",
        action="store_true",
        help="Rebuild the layer-0 safety-validation package from existing checkpoints and records without rerunning model streams.",
    )
    parser.add_argument(
        "--repair-storage-summary-from-checkpoints",
        action="store_true",
        help="Repair Stage 5 storage-summary aggregates from existing checkpoints without model execution.",
    )
    parser.add_argument(
        "--finalize-from-checkpoints",
        action="store_true",
        help="Finalize the Stage 5 full package from existing checkpoints without model execution.",
    )
    parser.add_argument(
        "--exclude-partial-prompts",
        default=None,
        help="Comma-separated prompt names to preserve only as excluded partial provenance.",
    )
    return parser.parse_args()


def _load_records(output_dir: Path) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    metric_root = output_dir / STAGE5_METRIC_DIR
    certificate_root = output_dir / STAGE5_CERTIFICATE_DIR
    metric_records: dict[str, list[dict]] = {}
    certificate_records: dict[str, list[dict]] = {}
    for path in sorted(metric_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metric_records[path.stem] = [record for records in payload["records_by_method"].values() for record in records]
    for path in sorted(certificate_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        certificate_records[path.stem] = list(payload["records"])
    return metric_records, certificate_records


def _aggregate_prompt_method_records(output_dir: Path) -> dict[str, dict[str, list[dict]]]:
    metric_root = output_dir / STAGE5_METRIC_DIR
    combined: dict[str, dict[str, list[dict]]] = {}
    for path in sorted(metric_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        combined[path.stem] = {
            method_name: list(records)
            for method_name, records in payload["records_by_method"].items()
        }
    return combined


def _aggregate_all_methods(prompt_records: dict[str, dict[str, list[dict]]]) -> dict[str, dict[str, float]]:
    method_records: dict[str, list[dict]] = {}
    for prompt_payload in prompt_records.values():
        for method_name, records in prompt_payload.items():
            method_records.setdefault(method_name, []).extend(records)
    return {
        method_name: _aggregate_metric_records(records)
        for method_name, records in method_records.items()
    }


def _parse_csv_names(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _expected_scored_token_count(token_count: int) -> int:
    return max(int(token_count) - int(STAGE5_SCORE_START) - 1, 0)


def _prompt_paths(output_dir: Path, prompt_name: str) -> dict[str, Path]:
    return {
        "source": output_dir / STAGE5_PROMPT_DIR / f"{prompt_name}.txt",
        "metadata": output_dir / STAGE5_PROMPT_DIR / f"{prompt_name}_metadata.json",
        "token_ids": output_dir / STAGE5_PROMPT_DIR / f"{prompt_name}_token_ids.json",
    }


def _load_prompt_metadata(output_dir: Path, prompt_name: str) -> dict[str, Any]:
    metadata_path = _prompt_paths(output_dir, prompt_name)["metadata"]
    if not metadata_path.exists():
        raise Stage5ExecutionError(f"Missing Stage 5 prompt metadata for {prompt_name}: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("schema") != STAGE5_PROMPT_CORPUS_VERSION:
        raise Stage5ExecutionError(
            f"Prompt metadata for {prompt_name} has schema {payload.get('schema')!r}, expected {STAGE5_PROMPT_CORPUS_VERSION!r}."
        )
    return payload


def _load_prompt_metric_payload(output_dir: Path, prompt_name: str) -> dict[str, Any]:
    path = output_dir / STAGE5_METRIC_DIR / f"{prompt_name}.json"
    if not path.exists():
        raise Stage5ExecutionError(f"Missing Stage 5 metric records for {prompt_name}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("prompt_name") != prompt_name:
        raise Stage5ExecutionError(f"Metric payload prompt mismatch for {prompt_name}: {payload.get('prompt_name')!r}.")
    return payload


def _load_prompt_certificate_payload(output_dir: Path, prompt_name: str) -> dict[str, Any]:
    path = output_dir / STAGE5_CERTIFICATE_DIR / f"{prompt_name}.json"
    if not path.exists():
        raise Stage5ExecutionError(f"Missing Stage 5 certificate records for {prompt_name}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("prompt_name") != prompt_name:
        raise Stage5ExecutionError(
            f"Certificate payload prompt mismatch for {prompt_name}: {payload.get('prompt_name')!r}."
        )
    return payload


def _numeric_array(records: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(record[key]) for record in records], dtype=np.float64)


def _aggregate_metric_records_strict(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise Stage5ExecutionError("Cannot aggregate an empty Stage 5 metric record set.")

    nlls = _numeric_array(records, "nll")
    delta_nlls = _numeric_array(records, "delta_nll_vs_full")
    kls = _numeric_array(records, "kl_divergence")
    jss = _numeric_array(records, "jensen_shannon_divergence")
    logit_l2 = _numeric_array(records, "logit_l2_error")
    rel_l2 = _numeric_array(records, "relative_logit_l2_error")
    max_abs = _numeric_array(records, "max_abs_logit_component_error")
    cosine = _numeric_array(records, "cosine_similarity")
    top1 = np.asarray([1.0 if bool(record["top1_agreement_with_full"]) else 0.0 for record in records], dtype=np.float64)
    top5_contains = np.asarray([1.0 if bool(record["top5_contains_full_top1"]) else 0.0 for record in records], dtype=np.float64)
    top5_overlap = np.asarray([float(record.get("top5_set_overlap_count", 0)) for record in records], dtype=np.float64)
    abs_rank_change = np.asarray([abs(int(record.get("target_token_rank_change", 0))) for record in records], dtype=np.float64)
    all_values = (nlls, delta_nlls, kls, jss, logit_l2, rel_l2, max_abs, cosine, top1, top5_contains, top5_overlap, abs_rank_change)
    if not all(np.all(np.isfinite(values)) for values in all_values):
        raise Stage5ExecutionError("Encountered non-finite Stage 5 metric values while aggregating results.")

    return {
        "case_count": int(len(records)),
        "mean_nll": float(np.mean(nlls)),
        "median_nll": float(np.median(nlls)),
        "total_nll": float(np.sum(nlls)),
        "perplexity": float(math.exp(np.mean(nlls))),
        "mean_delta_nll_vs_full": float(np.mean(delta_nlls)),
        "median_delta_nll_vs_full": float(np.median(delta_nlls)),
        "p95_delta_nll_vs_full": float(np.quantile(delta_nlls, 0.95)),
        "max_delta_nll_vs_full": float(np.max(delta_nlls)),
        "mean_top1_agreement": float(np.mean(top1)),
        "mean_top5_contains_full_top1": float(np.mean(top5_contains)),
        "mean_top5_set_overlap_count": float(np.mean(top5_overlap)),
        "median_top5_set_overlap_count": float(np.median(top5_overlap)),
        "p95_top5_set_overlap_count": float(np.quantile(top5_overlap, 0.95)),
        "min_top5_set_overlap_count": float(np.min(top5_overlap)),
        "mean_kl_divergence": float(np.mean(kls)),
        "median_kl_divergence": float(np.median(kls)),
        "p95_kl_divergence": float(np.quantile(kls, 0.95)),
        "max_kl_divergence": float(np.max(kls)),
        "mean_js_divergence": float(np.mean(jss)),
        "median_js_divergence": float(np.median(jss)),
        "p95_js_divergence": float(np.quantile(jss, 0.95)),
        "max_js_divergence": float(np.max(jss)),
        "mean_logit_l2_error": float(np.mean(logit_l2)),
        "median_logit_l2_error": float(np.median(logit_l2)),
        "p95_logit_l2_error": float(np.quantile(logit_l2, 0.95)),
        "max_logit_l2_error": float(np.max(logit_l2)),
        "mean_relative_logit_l2_error": float(np.mean(rel_l2)),
        "median_relative_logit_l2_error": float(np.median(rel_l2)),
        "p95_relative_logit_l2_error": float(np.quantile(rel_l2, 0.95)),
        "max_relative_logit_l2_error": float(np.max(rel_l2)),
        "mean_max_abs_logit_component_error": float(np.mean(max_abs)),
        "median_max_abs_logit_component_error": float(np.median(max_abs)),
        "p95_max_abs_logit_component_error": float(np.quantile(max_abs, 0.95)),
        "max_max_abs_logit_component_error": float(np.max(max_abs)),
        "mean_cosine_similarity": float(np.mean(cosine)),
        "median_cosine_similarity": float(np.median(cosine)),
        "min_cosine_similarity": float(np.min(cosine)),
        "mean_abs_target_token_rank_change": float(np.mean(abs_rank_change)),
        "median_abs_target_token_rank_change": float(np.median(abs_rank_change)),
        "p95_abs_target_token_rank_change": float(np.quantile(abs_rank_change, 0.95)),
        "max_abs_target_token_rank_change": float(np.max(abs_rank_change)),
        "metrics_valid": True,
        "invalid_reason": None,
    }


def _rebuild_stream_storage_summary(
    *,
    method_name: str,
    payload: Mapping[str, Any],
    full_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    settings = payload.get("settings", {})
    sequence_length = int(settings.get("token_count", 0))
    total_layers = int(settings.get("total_layers_to_run", 0))
    modified_layers = tuple(int(value) for value in settings.get("modified_layers", STAGE5_MODIFIED_LAYERS))
    full_layer_exact_per_token = _full_layer_exact_per_token_from_checkpoint(dict(full_checkpoint))

    if method_name == "full_kv":
        existing_storage = dict(payload.get("method_aggregate", {}).get("storage", {}))
        if not existing_storage:
            raise Stage5ExecutionError("Full-KV checkpoint is missing method_aggregate.storage.")
        token_totals = [int(value) for value in existing_storage.get("token_totals_bytes", ())]
        full_reference_totals = [int(value) for value in existing_storage.get("full_reference_totals_bytes", ())]
        category_sums = {key: int(value) for key, value in existing_storage.get("category_sums", {}).items()}
        cumulative_total = int(existing_storage.get("cumulative_total_serialized_bytes", sum(token_totals)))
        cumulative_full_reference = int(existing_storage.get("cumulative_full_reference_bytes", sum(full_reference_totals)))
        if len(token_totals) != sequence_length or len(full_reference_totals) != sequence_length:
            raise Stage5ExecutionError(
                f"Full-KV storage summary length mismatch: token_totals={len(token_totals)} full_reference={len(full_reference_totals)} expected={sequence_length}."
            )
        if sum(category_sums.values()) != cumulative_total:
            raise Stage5ExecutionError("Full-KV storage category sums do not match cumulative serialized bytes.")
        rebuilt_storage = dict(existing_storage)
        rebuilt_storage.setdefault("storage_accounting_schema", STAGE5_STORAGE_ACCOUNTING_SCHEMA)
        rebuilt_storage.setdefault("token_totals_scope", "all_layers_cumulative_prefix_bytes")
        rebuilt_storage.setdefault("full_reference_scope", "all_layers_exact_full_kv_cumulative_prefix_bytes")
        rebuilt_storage.setdefault("category_sums_scope", "all_layers_cumulative_prefix_bytes")
        rebuilt_storage.setdefault("modified_layer_count", 0)
        rebuilt_storage.setdefault("unmodified_layer_count", int(total_layers))
        rebuilt_storage.setdefault("modified_layers_token_totals_bytes", [0 for _ in range(sequence_length)])
        rebuilt_storage.setdefault("modified_layers_cumulative_total_serialized_bytes", 0)
        rebuilt_storage.setdefault("final_modified_layers_serialized_bytes", 0)
        rebuilt_storage.setdefault("unmodified_exact_token_totals_bytes", token_totals)
        rebuilt_storage.setdefault("unmodified_exact_cumulative_total_bytes", int(cumulative_total))
        rebuilt_storage.setdefault("cumulative_total_serialized_bytes", int(cumulative_total))
        rebuilt_storage.setdefault("cumulative_full_reference_bytes", int(cumulative_full_reference))
        if full_reference_totals:
            rebuilt_storage.setdefault("final_full_reference_bytes", int(full_reference_totals[-1]))
        if token_totals:
            rebuilt_storage.setdefault("final_total_bytes", int(token_totals[-1]))
        rebuilt_storage.setdefault("final_compression_ratio_vs_full_kv", 1.0)
        rebuilt_storage.setdefault("final_memory_saving_fraction_vs_full_kv", 0.0)
        rebuilt_storage.setdefault("mean_compression_ratio_vs_full_kv", 1.0)
        rebuilt_storage.setdefault("mean_memory_saving_fraction_vs_full_kv", 0.0)
        rebuilt_storage["disjoint_category_total_bytes"] = int(sum(category_sums.values()))
        rebuilt_storage["byte_accounting_consistent"] = True
        return rebuilt_storage

    return _method_storage_summary(
        method_name=method_name,
        modified_token_totals=[int(value) for value in payload.get("storage_modified_token_totals", ())],
        category_sums_override={key: int(value) for key, value in payload.get("storage_category_sums", {}).items()},
        total_layers=total_layers,
        modified_layers=modified_layers,
        num_kv_heads=None,
        head_dim=None,
        sequence_length=sequence_length,
        full_layer_exact_per_token=full_layer_exact_per_token,
        recent_window=int(settings.get("recent_window", 16)),
        block_size=int(settings.get("block_size", 8)),
        precision=int(settings.get("precision", 256)),
    )


def _aggregate_storage_summaries(storage_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not storage_summaries:
        raise Stage5ExecutionError("Cannot aggregate an empty Stage 5 storage summary set.")
    category_sums = {key: 0 for key in storage_summaries[0]["category_sums"].keys()}
    final_total_bytes = 0
    final_full_reference_bytes = 0
    cumulative_total_serialized_bytes = 0
    cumulative_full_reference_bytes = 0
    modified_layers_cumulative_total_serialized_bytes = 0
    disjoint_category_total_bytes = 0
    final_modified_layers_serialized_bytes = 0
    for storage in storage_summaries:
        if not bool(storage.get("byte_accounting_consistent", False)):
            raise Stage5ExecutionError("Encountered an inconsistent Stage 5 storage summary during aggregation.")
        final_total_bytes += int(storage["final_total_bytes"])
        final_full_reference_bytes += int(storage["final_full_reference_bytes"])
        cumulative_total_serialized_bytes += int(storage["cumulative_total_serialized_bytes"])
        cumulative_full_reference_bytes += int(storage["cumulative_full_reference_bytes"])
        modified_layers_cumulative_total_serialized_bytes += int(storage["modified_layers_cumulative_total_serialized_bytes"])
        disjoint_category_total_bytes += int(storage["disjoint_category_total_bytes"])
        final_modified_layers_serialized_bytes += int(storage["final_modified_layers_serialized_bytes"])
        for key, value in storage["category_sums"].items():
            category_sums[key] += int(value)
    if sum(category_sums.values()) != disjoint_category_total_bytes:
        raise Stage5ExecutionError(
            "Aggregate Stage 5 storage category sums do not match aggregate disjoint serialized bytes."
        )
    return {
        "prompt_count": int(len(storage_summaries)),
        "final_prefix_total_bytes": int(final_total_bytes),
        "final_prefix_full_reference_bytes": int(final_full_reference_bytes),
        "final_prefix_compression_ratio_vs_full_kv": (
            float(final_full_reference_bytes / final_total_bytes) if final_total_bytes > 0 else math.inf
        ),
        "final_prefix_memory_saving_fraction_vs_full_kv": (
            float(1.0 - (final_total_bytes / final_full_reference_bytes)) if final_full_reference_bytes > 0 else 0.0
        ),
        "cumulative_total_serialized_bytes": int(cumulative_total_serialized_bytes),
        "cumulative_full_reference_bytes": int(cumulative_full_reference_bytes),
        "cumulative_compression_ratio_vs_full_kv": (
            float(cumulative_full_reference_bytes / cumulative_total_serialized_bytes)
            if cumulative_total_serialized_bytes > 0
            else math.inf
        ),
        "cumulative_memory_saving_fraction_vs_full_kv": (
            float(1.0 - (cumulative_total_serialized_bytes / cumulative_full_reference_bytes))
            if cumulative_full_reference_bytes > 0
            else 0.0
        ),
        "modified_layers_cumulative_total_serialized_bytes": int(modified_layers_cumulative_total_serialized_bytes),
        "final_modified_layers_serialized_bytes": int(final_modified_layers_serialized_bytes),
        "disjoint_category_total_bytes": int(disjoint_category_total_bytes),
        "category_sums": category_sums,
        "byte_accounting_consistent": True,
    }


def _stream_provenance_status(method_name: str, payload: Mapping[str, Any]) -> str:
    settings = payload.get("settings", {})
    method_impl = settings.get("method_implementation_version")
    expected_impl = _stage5_method_impl_version(method_name)
    if method_impl is None:
        return "legacy_missing_method_implementation_version"
    if (
        method_impl == expected_impl
        and settings.get("method_version") == STAGE5_METHOD_VERSION
        and settings.get("result_version") == STAGE5_RESULT_VERSION
    ):
        return "versioned_current_checkpoint"
    return "versioned_but_mismatched_checkpoint"


def _audit_required_streams(
    *,
    output_dir: Path,
    prompt_names: Sequence[str],
    required_methods: Sequence[str],
) -> dict[str, Any]:
    checkpoint_root = output_dir / STAGE5_CHECKPOINT_DIR
    metric_root = output_dir / STAGE5_METRIC_DIR
    certificate_root = output_dir / STAGE5_CERTIFICATE_DIR
    prompts: dict[str, Any] = {}
    all_complete = True

    metric_payloads = {prompt_name: _load_prompt_metric_payload(output_dir, prompt_name) for prompt_name in prompt_names}
    certificate_payloads = {
        prompt_name: _load_prompt_certificate_payload(output_dir, prompt_name) for prompt_name in prompt_names
    }

    for prompt_name in prompt_names:
        metadata = _load_prompt_metadata(output_dir, prompt_name)
        metric_payload = metric_payloads[prompt_name]
        cert_payload = certificate_payloads[prompt_name]
        prompt_entry = {
            "token_count": int(metadata["token_count"]),
            "expected_scored_token_count": _expected_scored_token_count(int(metadata["token_count"])),
            "methods": {},
        }
        full_checkpoint_path = checkpoint_root / f"{prompt_name}__full_kv.json"
        if not full_checkpoint_path.exists():
            raise Stage5ExecutionError(f"Missing required full_kv checkpoint for prompt {prompt_name}.")
        full_checkpoint = json.loads(full_checkpoint_path.read_text(encoding="utf-8"))

        for method_name in required_methods:
            checkpoint_path = checkpoint_root / f"{prompt_name}__{method_name}.json"
            if not checkpoint_path.exists():
                all_complete = False
                prompt_entry["methods"][method_name] = {
                    "checkpoint_path": str(checkpoint_path),
                    "status": "missing",
                }
                continue

            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if payload.get("schema") != STAGE5_STREAM_CHECKPOINT_SCHEMA:
                raise Stage5ExecutionError(
                    f"Checkpoint {checkpoint_path} has schema {payload.get('schema')!r}, expected {STAGE5_STREAM_CHECKPOINT_SCHEMA!r}."
                )
            settings = payload.get("settings", {})
            if settings.get("prompt_name") != prompt_name or payload.get("method_name") != method_name:
                raise Stage5ExecutionError(f"Checkpoint identity mismatch for {checkpoint_path}.")

            highest_completed_layer = _highest_completed_layer(dict(payload))
            total_layers = int(settings.get("total_layers_to_run", 0))
            metric_records = metric_payload.get("records_by_method", {}).get(method_name, [])
            certificate_records = list(payload.get("certificate_records", []))
            storage = _rebuild_stream_storage_summary(
                method_name=method_name,
                payload=payload,
                full_checkpoint=full_checkpoint,
            )
            final_logits_persisted_path = None
            if method_name == "full_kv":
                final_logits_persisted_path = payload.get("full_logits_path")
            final_logits_produced = bool(payload.get("metric_records")) and (
                method_name == "full_kv"
                or "final_logits" in payload.get("intermediate_layer_metrics", {})
            )
            token_metrics_exist = len(metric_records) == prompt_entry["expected_scored_token_count"] == len(payload.get("metric_records", []))
            all_layers_completed = payload.get("status") == "complete" and total_layers == 32 and int(payload.get("next_layer_index", 0)) == total_layers
            final_rmsnorm_lm_head_completed = bool(all_layers_completed and final_logits_produced)
            if method_name == "full_kv" and not final_logits_persisted_path:
                final_rmsnorm_lm_head_completed = False
            method_impl_version = settings.get("method_implementation_version")
            source_status = _stream_provenance_status(method_name, payload)
            if source_status != "versioned_current_checkpoint":
                all_complete = False
            if not all_layers_completed or not final_rmsnorm_lm_head_completed or not token_metrics_exist:
                all_complete = False
            if method_name == "rack_kv_certified" and not certificate_records:
                all_complete = False
            if method_name != "rack_kv_certified" and certificate_records:
                raise Stage5ExecutionError(f"Unexpected certificate records stored in non-certified checkpoint {checkpoint_path}.")
            if method_name == "rack_kv_compression_only" and int(storage["category_sums"]["certificate_metadata_bytes"]) != 0:
                raise Stage5ExecutionError("Compression-only storage summary reported nonzero certificate metadata bytes.")
            if prompt_name != cert_payload.get("prompt_name"):
                raise Stage5ExecutionError(f"Certificate prompt mismatch for {prompt_name}.")
            prompt_entry["methods"][method_name] = {
                "checkpoint_path": str(checkpoint_path),
                "status": str(payload.get("status")),
                "next_layer_index": int(payload.get("next_layer_index", 0)),
                "highest_completed_layer": highest_completed_layer,
                "all_32_layers_completed": bool(all_layers_completed),
                "final_rmsnorm_and_lm_head_completed": bool(final_rmsnorm_lm_head_completed),
                "final_logits_exist": bool(final_logits_produced),
                "final_logits_persisted": bool(final_logits_persisted_path),
                "final_logits_persisted_path": final_logits_persisted_path,
                "token_level_metrics_exist": bool(token_metrics_exist),
                "token_level_metric_count": int(len(metric_records)),
                "storage_summary_exists": True,
                "certificate_records_exist": bool(method_name == "rack_kv_certified" and len(certificate_records) > 0),
                "certificate_record_count": int(len(certificate_records)),
                "method_implementation_version": method_impl_version,
                "storage_accounting_schema_version": str(storage["storage_accounting_schema"]),
                "source_provenance_status": source_status,
                "storage_summary": storage,
            }
        prompts[prompt_name] = prompt_entry

    return {
        "schema": "stage5_required_stream_audit_v1",
        "required_prompts": list(prompt_names),
        "required_methods": list(required_methods),
        "all_required_streams_complete": bool(all_complete),
        "prompts": prompts,
    }


def _partial_prompt_audit(
    *,
    output_dir: Path,
    excluded_prompt_names: Sequence[str],
) -> dict[str, Any]:
    checkpoint_root = output_dir / STAGE5_CHECKPOINT_DIR
    entries = []
    for prompt_name in excluded_prompt_names:
        present_methods = {}
        for method_name in STAGE5_METHODS:
            path = checkpoint_root / f"{prompt_name}__{method_name}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            present_methods[method_name] = {
                "checkpoint_path": str(path),
                "status": payload.get("status"),
                "next_layer_index": int(payload.get("next_layer_index", 0)),
                "highest_completed_layer": _highest_completed_layer(payload),
                "method_implementation_version": payload.get("settings", {}).get("method_implementation_version"),
            }
        prompt_paths = _prompt_paths(output_dir, prompt_name)
        entries.append(
            {
                "prompt_name": prompt_name,
                "status": "excluded_partial_prompt",
                "reason": "user_requested_finalization_from_two_completed_prompts_only",
                "present_methods": present_methods,
                "prompt_corpus_files_present": {
                    key: path.exists() for key, path in prompt_paths.items()
                },
            }
        )
    return {
        "schema": "stage5_excluded_partial_prompt_audit_v1",
        "entries": entries,
    }


def _build_final_prompt_result(
    *,
    output_dir: Path,
    prompt_name: str,
    required_methods: Sequence[str],
    stream_audit: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = _load_prompt_metadata(output_dir, prompt_name)
    metric_payload = _load_prompt_metric_payload(output_dir, prompt_name)
    certificate_payload = _load_prompt_certificate_payload(output_dir, prompt_name)
    methods = stream_audit["methods"]
    expected_cases = _expected_scored_token_count(int(metadata["token_count"]))
    if expected_cases <= 0:
        raise Stage5ExecutionError(f"Prompt {prompt_name} does not contain any scored tokens.")

    prompt_result = {
        "name": prompt_name,
        "prompt_category": (
            "natural_language_continuation"
            if prompt_name == "natural_language_128"
            else "passkey_retrieval"
        ),
        "source_sha256": str(metadata["source_sha256"]),
        "token_count": int(metadata["token_count"]),
        "scored_token_count": int(expected_cases),
        "source_path": str(metadata["source_path"]),
        "token_ids_path": str(metadata["token_ids_path"]),
        "metadata_path": str((output_dir / STAGE5_PROMPT_DIR / f"{prompt_name}_metadata.json")),
        "method_aggregates": {},
        "passkey_metrics": {},
        "certificate_summary": None,
        "stream_audit": methods,
    }
    full_records = list(metric_payload["records_by_method"]["full_kv"])
    full_aggregate = _aggregate_metric_records_strict(full_records)
    if full_aggregate["case_count"] != expected_cases:
        raise Stage5ExecutionError(
            f"Prompt {prompt_name} full_kv case count {full_aggregate['case_count']} != expected {expected_cases}."
        )

    cert_records = list(certificate_payload["records"])
    for method_name in required_methods:
        records = list(metric_payload["records_by_method"].get(method_name, ()))
        if len(records) != expected_cases:
            raise Stage5ExecutionError(
                f"Prompt {prompt_name} method {method_name} has {len(records)} token records, expected {expected_cases}."
            )
        aggregate = _aggregate_metric_records_strict(records)
        aggregate["delta_mean_nll_vs_full_kv"] = float(aggregate["mean_nll"] - full_aggregate["mean_nll"])
        aggregate["perplexity_ratio_vs_full_kv"] = float(aggregate["perplexity"] / full_aggregate["perplexity"])
        aggregate["storage"] = methods[method_name]["storage_summary"]
        if method_name == "rack_kv_certified":
            aggregate["certificate"] = _aggregate_certificate_records(
                cert_records,
                num_attention_heads=32,
                num_key_value_heads=8,
            )
            prompt_result["certificate_summary"] = aggregate["certificate"]
        prompt_result["method_aggregates"][method_name] = aggregate

        stream_checkpoint = json.loads(Path(methods[method_name]["checkpoint_path"]).read_text(encoding="utf-8"))
        if prompt_name == "passkey_retrieval_128":
            passkey_metrics = stream_checkpoint.get("passkey_metrics")
            if passkey_metrics is None:
                raise Stage5ExecutionError(
                    f"Passkey prompt stream {prompt_name}/{method_name} is missing passkey_metrics."
                )
            prompt_result["passkey_metrics"][method_name] = passkey_metrics

    if prompt_name != "passkey_retrieval_128":
        prompt_result["passkey_metrics"] = None
    return prompt_result


def _build_two_prompt_report(results: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 5 Full 32-Layer Teacher-Forced Evaluation",
        "",
        results["scientific_scope_statement"],
        "",
        "The evaluation contains one natural-language continuation prompt and one passkey-retrieval prompt.",
        "",
        "## Scope",
        "",
        f"- Experiment scope: `{results['experiment_scope']}`",
        f"- Prompt names: `{results['prompt_names']}`",
        f"- Prompt count: {results['prompt_count']}",
        f"- Required methods: `{results['required_methods']}`",
        f"- Full transformer layers executed: `{results['full_transformer_layers_executed']}`",
        f"- Transformer layer count: {results['transformer_layer_count']}",
        f"- RACK-modified layers: `{results['rack_modified_layers']}`",
        f"- Excluded partial prompts: `{results['excluded_partial_prompts']}`",
        "",
        "## Aggregate Quality Results",
        "",
    ]
    for method_name, aggregate in results["aggregate_method_metrics"].items():
        lines.append(
            f"- {method_name}: cases={aggregate['case_count']}, mean NLL={aggregate['mean_nll']:.6f}, "
            f"perplexity={aggregate['perplexity']:.6f}, mean top-1 agreement={aggregate['mean_top1_agreement']:.6f}, "
            f"mean KL={aggregate['mean_kl_divergence']:.6f}, mean logit L2={aggregate['mean_logit_l2_error']:.6f}, "
            f"max logit L2={aggregate['max_logit_l2_error']:.6f}"
        )
    lines.extend(["", "## Per-Prompt Results", ""])
    for prompt in results["prompts"]:
        lines.append(f"### {prompt['name']}")
        lines.append(f"- Source SHA-256: `{prompt['source_sha256']}`")
        lines.append(f"- Token count: {prompt['token_count']}")
        lines.append(f"- Scored tokens: {prompt['scored_token_count']}")
        for method_name, aggregate in prompt["method_aggregates"].items():
            storage = aggregate["storage"]
            lines.append(
                f"- {method_name}: mean NLL={aggregate['mean_nll']:.6f}, perplexity={aggregate['perplexity']:.6f}, "
                f"top-1 agreement={aggregate['mean_top1_agreement']:.6f}, mean KL={aggregate['mean_kl_divergence']:.6f}, "
                f"final-prefix bytes={storage['final_total_bytes']}, final-prefix saving={storage['final_memory_saving_fraction_vs_full_kv']:.6f}"
            )
        if prompt["passkey_metrics"] is not None:
            lines.append("- Passkey-specific metrics:")
            for method_name, metrics in prompt["passkey_metrics"].items():
                lines.append(
                    f"  - {method_name}: answer log p={metrics['teacher_forced_answer_log_probability']:.6f}, "
                    f"delta vs full={metrics['delta_answer_log_probability_vs_full']:.6f}, "
                    f"first-answer rank={metrics['first_answer_token_rank']}, "
                    f"top1={metrics['first_answer_token_is_top1']}, top5={metrics['first_answer_token_in_top5']}"
                )
        lines.append("")
    lines.extend(
        [
            "## Storage Results",
            "",
        ]
    )
    for method_name, storage in results["aggregate_storage"].items():
        lines.append(
            f"- {method_name}: final-prefix bytes={storage['final_prefix_total_bytes']}, "
            f"final-prefix ratio={storage['final_prefix_compression_ratio_vs_full_kv']:.6f}, "
            f"final-prefix saving={storage['final_prefix_memory_saving_fraction_vs_full_kv']:.6f}, "
            f"cumulative bytes={storage['cumulative_total_serialized_bytes']}, "
            f"disjoint-category bytes={storage['disjoint_category_total_bytes']}"
        )
    lines.extend(["", "## Certificate Results", ""])
    cert = results["aggregate_certificate"]
    lines.extend(
        [
            f"- Eligible query-head/block decisions: {cert['total_eligible_query_head_block_decisions']}",
            f"- FP64 prefilter rejections: {cert['prefilter_rejected_query_head_block_decisions']}",
            f"- MPFR candidates: {cert['candidates_sent_to_mpfr']}",
            f"- MPFR-certified logical skips: {cert['mpfr_certified_query_head_block_skips']}",
            f"- MPFR rejections: {cert['mpfr_rejected_candidates']}",
            f"- Unique tokens with skips: {cert['unique_tokens_with_any_skip']}",
            f"- Unique query heads with skips: {cert['unique_query_heads_with_any_skip']}",
            f"- Unique KV heads with skips: {cert['unique_kv_heads_with_any_skip']}",
            f"- Logical KV-head/block skips: {cert['unique_logical_kv_blocks_with_any_skip']}",
            f"- GQA physical block skips: {cert['gqa_physical_blocks_skippable_by_all_mapped_query_heads']}",
            f"- Physical block decodes avoided: {cert['physical_block_decodes_actually_avoided']}",
            f"- Weighted skip fraction: {cert['weighted_skip_fraction']:.6f}",
            f"- Rigorous interval violations: {cert['rigorous_interval_violations']}",
            f"- Approximate observed violations: {cert['approximate_observed_violations']}",
            f"- False-safe count: {cert['false_safe_count']}",
            f"- Numerical fallbacks: {cert['numerical_fallbacks']}",
            "",
            "## Validation",
            "",
            "- Full KV is the reference stream.",
            "- RACK-KV compression-only and certified streams use the unified reconstructed-attention arithmetic path in the current implementation.",
            "- Certified-mode output must match compression-only when no skip is authorized; this is enforced by the deterministic unit tests included in the package.",
            "- No legacy mixed-arithmetic RACK checkpoints are included in this final package.",
            "- Storage summaries were rebuilt from checkpoint storage totals at a single authoritative scope and exclude the partial third prompt from all scientific aggregates.",
            "",
            "## Limitations",
            "",
            "The full-model evaluation uses two short deterministic prompt categories. Additional prompts, longer contexts, generation experiments, and a second model remain necessary for broader generalization.",
            "- The passkey and natural-language prompts do not provide broad task coverage.",
            "- The current certificate remains local and conservative.",
            "- No physical skipping or runtime acceleration is claimed from this package.",
        ]
    )
    return "\n".join(lines)


def _collect_final_review_files(
    *,
    output_dir: Path,
    included_prompts: Sequence[str],
    excluded_partial_prompts: Sequence[str],
) -> dict[str, Path]:
    file_map: dict[str, Path] = {}
    for relative_name in (
        STAGE5_RESULTS_JSON,
        STAGE5_REPORT_MD,
        STAGE5_TEST_LOG,
        "test_command.txt",
        "test_return_code.txt",
        "stage5_full_run_log.txt",
    ):
        path = output_dir / relative_name
        if path.exists():
            file_map[relative_name] = path
    for relative_source in STAGE5_REVIEW_SOURCE_FILES:
        file_map[relative_source] = Path(relative_source)
    provenance_root = output_dir / STAGE5_PROVENANCE_DIR
    for path in sorted(provenance_root.rglob("*")):
        if path.is_file():
            file_map[str(path.relative_to(output_dir)).replace("\\", "/")] = path
    for prompt_name in included_prompts:
        for kind, path in _prompt_paths(output_dir, prompt_name).items():
            if path.exists():
                file_map[str(path.relative_to(output_dir)).replace("\\", "/")] = path
        prompt_checkpoint = output_dir / STAGE5_CHECKPOINT_DIR / f"{prompt_name}.json"
        if prompt_checkpoint.exists():
            file_map[str(prompt_checkpoint.relative_to(output_dir)).replace("\\", "/")] = prompt_checkpoint
        metric_path = output_dir / STAGE5_METRIC_DIR / f"{prompt_name}.json"
        cert_path = output_dir / STAGE5_CERTIFICATE_DIR / f"{prompt_name}.json"
        if metric_path.exists():
            file_map[str(metric_path.relative_to(output_dir)).replace("\\", "/")] = metric_path
        if cert_path.exists():
            file_map[str(cert_path.relative_to(output_dir)).replace("\\", "/")] = cert_path
        for method_name in STAGE5_METHODS:
            checkpoint_path = output_dir / STAGE5_CHECKPOINT_DIR / f"{prompt_name}__{method_name}.json"
            if checkpoint_path.exists():
                file_map[str(checkpoint_path.relative_to(output_dir)).replace("\\", "/")] = checkpoint_path
    for prompt_name in excluded_partial_prompts:
        for kind, path in _prompt_paths(output_dir, prompt_name).items():
            if path.exists():
                file_map[f"excluded_partial_prompt_provenance/prompt_corpus/{path.name}"] = path
        for method_name in STAGE5_METHODS:
            checkpoint_path = output_dir / STAGE5_CHECKPOINT_DIR / f"{prompt_name}__{method_name}.json"
            if checkpoint_path.exists():
                file_map[f"excluded_partial_prompt_provenance/checkpoints/{checkpoint_path.name}"] = checkpoint_path
    return file_map


def _finalize_from_checkpoints(
    *,
    output_dir: Path,
    review_zip: Path,
    prompt_names: Sequence[str],
    excluded_partial_prompts: Sequence[str],
) -> dict[str, Any]:
    if not prompt_names:
        raise Stage5ExecutionError("Finalize-from-checkpoints requires explicit --prompt-names.")
    required_methods = tuple(STAGE5_METHODS)
    stream_audit = _audit_required_streams(
        output_dir=output_dir,
        prompt_names=prompt_names,
        required_methods=required_methods,
    )
    if not bool(stream_audit["all_required_streams_complete"]):
        return {
            "schema": "stage5_finalize_from_checkpoints_v1",
            "status": "incomplete_required_streams",
            "required_stream_audit": stream_audit,
            "excluded_partial_prompts": list(excluded_partial_prompts),
            "model_forward_calls": 0,
            "hidden_state_recomputations": 0,
            "mpfr_recomputations": 0,
            "remote_tensor_fetches": 0,
        }

    excluded_audit = _partial_prompt_audit(
        output_dir=output_dir,
        excluded_prompt_names=excluded_partial_prompts,
    )
    prompt_results = []
    aggregate_records_by_method = {method_name: [] for method_name in required_methods}
    aggregate_storage_by_method: dict[str, dict[str, Any]] = {}

    for prompt_name in prompt_names:
        prompt_result = _build_final_prompt_result(
            output_dir=output_dir,
            prompt_name=prompt_name,
            required_methods=required_methods,
            stream_audit=stream_audit["prompts"][prompt_name],
        )
        prompt_results.append(prompt_result)
        for method_name in required_methods:
            metric_payload = _load_prompt_metric_payload(output_dir, prompt_name)
            aggregate_records_by_method[method_name].extend(metric_payload["records_by_method"][method_name])

    aggregate_method_metrics: dict[str, dict[str, Any]] = {}
    for method_name, records in aggregate_records_by_method.items():
        aggregate_method_metrics[method_name] = _aggregate_metric_records_strict(records)

    full_reference = aggregate_method_metrics["full_kv"]
    for method_name, aggregate in aggregate_method_metrics.items():
        aggregate["delta_mean_nll_vs_full_kv"] = float(aggregate["mean_nll"] - full_reference["mean_nll"])
        aggregate["perplexity_ratio_vs_full_kv"] = float(aggregate["perplexity"] / full_reference["perplexity"])

    for method_name in required_methods:
        summaries = [
            prompt["method_aggregates"][method_name]["storage"]
            for prompt in prompt_results
        ]
        aggregate_storage_by_method[method_name] = _aggregate_storage_summaries(summaries)

    all_certificate_records: list[dict[str, Any]] = []
    for prompt_name in prompt_names:
        all_certificate_records.extend(_load_prompt_certificate_payload(output_dir, prompt_name)["records"])
    aggregate_certificate = _aggregate_certificate_records(
        all_certificate_records,
        num_attention_heads=32,
        num_key_value_heads=8,
    )

    for method_name in ("rack_kv_compression_only", "rack_kv_certified"):
        impl_versions = {
            stream_audit["prompts"][prompt_name]["methods"][method_name]["method_implementation_version"]
            for prompt_name in prompt_names
        }
        expected_version = _stage5_method_impl_version(method_name)
        if impl_versions != {expected_version}:
            raise Stage5ExecutionError(
                f"Refusing to finalize with non-current {method_name} checkpoints: found {sorted(impl_versions)} expected {expected_version}."
            )

    scientific_scope_statement = (
        "Full 32-layer teacher-forced evaluation on two 128-token prompt categories: "
        "natural-language continuation and passkey retrieval."
    )
    results = {
        "schema": STAGE5_RUN_SCHEMA,
        "result_version": STAGE5_RESULT_VERSION,
        "method_version": STAGE5_METHOD_VERSION,
        "finalization_mode": "finalize_from_checkpoints",
        "reporting_only_rebuild": True,
        "experiment_scope": "full_32_layer_teacher_forced_two_prompt_evaluation",
        "scientific_scope_statement": scientific_scope_statement,
        "prompt_names": list(prompt_names),
        "prompt_count": int(len(prompt_names)),
        "required_methods": list(required_methods),
        "excluded_partial_prompts": list(excluded_partial_prompts),
        "full_transformer_layers_executed": True,
        "transformer_layer_count": 32,
        "rack_modified_layers": list(STAGE5_MODIFIED_LAYERS),
        "configuration": {
            "prompt_names": list(prompt_names),
            "prompt_count": int(len(prompt_names)),
            "required_methods": list(required_methods),
            "full_transformer_layers_executed": True,
            "transformer_layer_count": 32,
            "rack_modified_layers": list(STAGE5_MODIFIED_LAYERS),
        },
        "prompts": prompt_results,
        "aggregate_method_metrics": aggregate_method_metrics,
        "aggregate_storage": aggregate_storage_by_method,
        "aggregate_certificate": aggregate_certificate,
        "required_stream_audit": stream_audit,
        "excluded_partial_prompt_audit": excluded_audit,
        "arithmetic_validation": {
            "full_kv_is_reference_stream": True,
            "unified_reconstructed_attention_required": True,
            "legacy_mixed_attention_checkpoints_used": False,
            "certified_equals_compression_only_when_no_skip_authorized": "validated_by_tests",
            "independent_stream_propagation_evidence": "stored final_hidden_state and final_logits deltas per non-full method",
        },
        "storage_validation": {
            "storage_accounting_schema": STAGE5_STORAGE_ACCOUNTING_SCHEMA,
            "compression_only_certificate_metadata_bytes_zero": True,
            "aggregate_category_sum_matches_total_for_all_methods": True,
        },
        "model_forward_calls": 0,
        "hidden_state_recomputations": 0,
        "mpfr_recomputations": 0,
        "remote_tensor_fetches": 0,
        "dependency_versions": stage5_dependency_versions(),
    }

    provenance_dir = output_dir / STAGE5_PROVENANCE_DIR
    provenance_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(provenance_dir / "checkpoint_audit.json", stream_audit)
    _write_json_atomic(provenance_dir / "prompt_inclusion_exclusion_audit.json", {
        "schema": "stage5_prompt_inclusion_exclusion_v1",
        "included_prompts": list(prompt_names),
        "excluded_partial_prompts": list(excluded_partial_prompts),
        "excluded_partial_prompt_audit": excluded_audit,
    })
    _write_json_atomic(provenance_dir / "storage_summaries.json", {
        "schema": "stage5_storage_summaries_v1",
        "aggregate_storage": aggregate_storage_by_method,
    })
    _write_json_atomic(provenance_dir / "certificate_aggregates.json", {
        "schema": "stage5_certificate_aggregates_v1",
        "aggregate_certificate": aggregate_certificate,
    })
    _write_json_atomic(provenance_dir / "dependency_versions.json", results["dependency_versions"])
    _write_json_atomic(provenance_dir / "experiment_configuration.json", results["configuration"])
    _write_json_atomic(provenance_dir / "manifest_verifier.json", {
        "schema": "stage5_manifest_verifier_v1",
        "verifier_script": "scripts/run_stage5_quality_smoke.py",
        "verifier_function": "_verify_review_zip",
        "notes": "The review ZIP is verified after creation against manifest.json. The manifest does not hash itself.",
    })
    _write_json_atomic(provenance_dir / "excluded_partial_prompt_provenance.json", excluded_audit)

    report = _build_two_prompt_report(results)
    _write_text_atomic(output_dir / STAGE5_REPORT_MD, report)
    _write_json_atomic(output_dir / STAGE5_RESULTS_JSON, results)

    file_map = _collect_final_review_files(
        output_dir=output_dir,
        included_prompts=prompt_names,
        excluded_partial_prompts=excluded_partial_prompts,
    )
    manifest = _build_manifest(
        file_map=file_map,
        source_relative_paths=STAGE5_REVIEW_SOURCE_FILES,
        source_snapshot_base_dir=Path.cwd(),
    )
    _write_json_atomic(output_dir / STAGE5_MANIFEST_JSON, manifest)
    zip_file_map = dict(file_map)
    zip_file_map[STAGE5_MANIFEST_JSON] = output_dir / STAGE5_MANIFEST_JSON
    _create_review_zip(file_map=zip_file_map, zip_path=review_zip)
    verification = _verify_review_zip(
        zip_path=review_zip,
        manifest=manifest,
        extra_files={STAGE5_MANIFEST_JSON: output_dir / STAGE5_MANIFEST_JSON},
    )
    return {
        "schema": "stage5_finalize_from_checkpoints_v1",
        "status": "complete",
        "required_stream_audit": stream_audit,
        "all_required_streams_complete": True,
        "excluded_partial_prompts": list(excluded_partial_prompts),
        "model_forward_calls": 0,
        "hidden_state_recomputations": 0,
        "mpfr_recomputations": 0,
        "remote_tensor_fetches": 0,
        "results_path": str(output_dir / STAGE5_RESULTS_JSON),
        "report_path": str(output_dir / STAGE5_REPORT_MD),
        "manifest_path": str(output_dir / STAGE5_MANIFEST_JSON),
        "review_zip_path": str(review_zip),
        "review_zip_sha256": verification["zip_sha256"],
        "manifest_mismatch_count": verification["manifest_mismatch_count"],
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "manifest_file_count": manifest["file_count"],
        "scientific_scope_statement": scientific_scope_statement,
        "aggregate_method_metrics": aggregate_method_metrics,
        "aggregate_storage": aggregate_storage_by_method,
        "aggregate_certificate": aggregate_certificate,
    }


def _prompt_report_section(results: dict[str, any]) -> str:
    lines = ["## Per-Prompt Results", ""]
    for prompt in results["prompts"]:
        lines.append(f"### {prompt['name']}")
        lines.append(f"- Source SHA-256: `{prompt['source_sha256']}`")
        lines.append(f"- Token count: {prompt['token_count']}")
        lines.append(f"- Scored tokens: {prompt['scored_token_count']}")
        lines.append(f"- Runtime: {prompt['runtime_s']:.2f}s")
        lines.append(f"- Peak RSS: {prompt['peak_rss_bytes'] / (1024 ** 3):.2f} GB")
        for method_name, aggregate in prompt["method_aggregates"].items():
            storage = aggregate.get("storage", {})
            lines.append(
                f"- {method_name}: mean NLL={aggregate['mean_nll']:.6f}, perplexity={aggregate['perplexity']:.6f}, "
                f"top-1 agreement={aggregate['mean_top1_agreement']:.4f}, mean KL={aggregate['mean_kl_divergence']:.6f}, "
                f"final bytes={storage.get('final_total_bytes', 0)}, ratio={storage.get('final_compression_ratio_vs_full_kv', 0.0):.6f}"
            )
        for method_name, metrics in prompt["intermediate_layer_metrics"].items():
            final_hidden = metrics["final_hidden_state"]["max_l2_difference"]
            final_logits = metrics["final_logits"]["max_l2_difference"]
            lines.append(
                f"- propagation/{method_name}: max final hidden-state L2={final_hidden:.6f}, "
                f"max final-logit L2={final_logits:.6f}"
            )
        if prompt["passkey_metrics"] is not None:
            lines.append("- Passkey prompt metrics:")
            for method_name, metrics in prompt["passkey_metrics"].items():
                lines.append(
                    f"  - {method_name}: answer log p={metrics['teacher_forced_answer_log_probability']:.6f}, "
                    f"delta vs full={metrics['delta_answer_log_probability_vs_full']:.6f}, "
                    f"first-answer rank={metrics['first_answer_token_rank']}, top1={metrics['first_answer_token_is_top1']}"
                )
        lines.append("")
    return "\n".join(lines)


def _certificate_summary(results: dict[str, any]) -> dict[str, any]:
    total = {
        "total_eligible_blocks": 0,
        "fast_prefilter_rejected_blocks": 0,
        "potential_skip_candidates_sent_to_mpfr": 0,
        "mpfr_certified_skipped_blocks": 0,
        "mpfr_rejected_skip_candidates": 0,
        "rigorous_interval_violations": 0,
        "approximate_observed_violations": 0,
        "false_safe_count": 0,
        "numerical_fallbacks": 0,
        "tokens_with_any_skip": 0,
        "layers_with_any_skip": set(),
        "heads_with_any_skip": set(),
        "max_per_token_skip_fraction": 0.0,
    }
    for prompt in results["prompts"]:
        cert = prompt["method_aggregates"].get("rack_kv_certified", {}).get("certificate", {})
        total["total_eligible_blocks"] += int(cert.get("total_eligible_blocks", 0))
        total["fast_prefilter_rejected_blocks"] += int(cert.get("fast_prefilter_rejected_blocks", 0))
        total["potential_skip_candidates_sent_to_mpfr"] += int(cert.get("potential_skip_candidates_sent_to_mpfr", 0))
        total["mpfr_certified_skipped_blocks"] += int(cert.get("mpfr_certified_skipped_blocks", 0))
        total["mpfr_rejected_skip_candidates"] += int(cert.get("mpfr_rejected_skip_candidates", 0))
        total["rigorous_interval_violations"] += int(cert.get("rigorous_interval_violations", 0))
        total["approximate_observed_violations"] += int(cert.get("approximate_observed_violations", 0))
        total["false_safe_count"] += int(cert.get("false_safe_count", 0))
        total["numerical_fallbacks"] += int(cert.get("numerical_fallbacks", 0))
        total["tokens_with_any_skip"] += int(cert.get("tokens_with_any_skip", 0))
        total["max_per_token_skip_fraction"] = max(
            total["max_per_token_skip_fraction"],
            float(cert.get("max_per_token_skip_fraction", 0.0)),
        )
        for layer_metrics in prompt["intermediate_layer_metrics"].get("rack_kv_certified", {}).keys():
            if layer_metrics.isdigit():
                pass
    total["weighted_skip_fraction"] = (
        float(total["mpfr_certified_skipped_blocks"] / total["total_eligible_blocks"])
        if total["total_eligible_blocks"] > 0
        else 0.0
    )
    return total


def _build_report(results: dict[str, any], *, output_dir: Path) -> str:
    prompt_records = _aggregate_prompt_method_records(output_dir)
    global_method_aggregates = _aggregate_all_methods(prompt_records)
    rack_cert = _certificate_summary(results)
    has_rack_cert = any("rack_kv_certified" in prompt["method_aggregates"] for prompt in results["prompts"])
    full_kv_equivalence = results.get("full_kv_equivalence_regression")
    lines = [
        "# Stage 5A Teacher-Forced Quality Smoke",
        "",
        "This is a controlled teacher-forced smoke, not yet the full Stage 5 task-quality benchmark.",
        "",
        "## Model And Provenance",
        "",
        f"- Model: `{results['repo_id']}`",
        f"- Revision: `{results['revision']}`",
        f"- TLS verification: `{results['tls_verification']}`",
        f"- Local computation: all forward passes ran locally on CPU; remote access, when used, was limited to tensor byte-range fetching.",
        f"- Tensor cache sources: `{results['tensor_cache_provenance']['sources_count']}`",
        f"- Remote tensor bytes fetched this run: `{results['tensor_cache_provenance']['remote_bytes_fetched']}`",
        "",
        "## Prompt Corpus",
        "",
        f"- Prompt schema: `{STAGE5_PROMPT_CORPUS_VERSION}`",
        f"- Prompt count: {len(results['prompts'])}",
        *[
            f"- {prompt['name']}: SHA-256 `{prompt['source_sha256']}`, token count {prompt['token_count']}, scored tokens {prompt['scored_token_count']}"
            for prompt in results["prompts"]
        ],
        "",
    ]
    if full_kv_equivalence is not None:
        lines.extend(
            [
                "## Full-KV Stock Equivalence",
                "",
                "- The runtime `full_kv` stream is the stock full-model teacher-forced reference.",
                f"- Short custom-vs-stock regression prompt: `{full_kv_equivalence['prompt_name']}` with {full_kv_equivalence['token_count']} tokens.",
                f"- Max absolute logit difference: {full_kv_equivalence['max_abs_logit_difference']:.6f}",
                f"- Mean logit L2 difference: {full_kv_equivalence['mean_logit_l2_difference']:.6f}",
                f"- Max logit L2 difference: {full_kv_equivalence['max_logit_l2_difference']:.6f}",
                f"- Relative logit L2 difference: {full_kv_equivalence['relative_logit_l2_difference']:.6f}",
                f"- Cosine similarity: {full_kv_equivalence['cosine_similarity']:.6f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Modified Layers",
            "",
            f"- Modified layers: `{results['configuration']['modified_layers']}`",
            f"- Executed layers in this run: `{results['configuration'].get('executed_layers', results['configuration']['modified_layers'])}`",
            "- All 32 query heads and all 8 KV heads are processed in every modified layer.",
            "- `full_kv` is the exact reference stream.",
            "- `rack_kv_compression_only` isolates compression loss by reconstructing every compressed block with no skipping.",
            "- `rack_kv_certified` uses the same compression representation and permits skipping only after MPFR certification.",
            "- `uniform_int8_kv` is the transparent quantization control.",
            "",
            "## Aggregate Token-Level Quality",
            "",
        ]
    )
    for method_name, aggregate in global_method_aggregates.items():
        lines.append(
            f"- {method_name}: mean NLL={aggregate['mean_nll']:.6f}, total NLL={aggregate['total_nll']:.6f}, "
            f"perplexity={aggregate['perplexity']:.6f}, top-1 agreement={aggregate['mean_top1_agreement']:.4f}, "
            f"mean KL={aggregate['mean_kl_divergence']:.6f}, max logit L2={aggregate['max_logit_l2_error']:.6f}"
        )
    lines.append("")
    if has_rack_cert:
        lines.extend(
            [
                "## RACK-KV Certificate Summary",
                "",
                f"- Total eligible blocks: {rack_cert['total_eligible_blocks']}",
                f"- Fast-prefilter rejected blocks: {rack_cert['fast_prefilter_rejected_blocks']}",
                f"- Potential skip candidates sent to MPFR: {rack_cert['potential_skip_candidates_sent_to_mpfr']}",
                f"- MPFR-certified skipped blocks: {rack_cert['mpfr_certified_skipped_blocks']}",
                f"- MPFR-rejected skip candidates: {rack_cert['mpfr_rejected_skip_candidates']}",
                f"- Weighted certified skip fraction: {rack_cert['weighted_skip_fraction']:.6f}",
                f"- Tokens with any skip: {rack_cert['tokens_with_any_skip']}",
                f"- Max per-token skip fraction: {rack_cert['max_per_token_skip_fraction']:.6f}",
                f"- Rigorous interval violations: {rack_cert['rigorous_interval_violations']}",
                f"- Approximate observed violations: {rack_cert['approximate_observed_violations']}",
                f"- False-safe count: {rack_cert['false_safe_count']}",
                f"- Numerical fallbacks: {rack_cert['numerical_fallbacks']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Intermediate Propagation",
            "",
            "- The modified-stream hidden states are propagated through all remaining layers before final logits are scored.",
            "- Per-method final hidden-state and final-logit differences are reported in the per-prompt section.",
            "",
            "## Prompt-Level Detail",
            "",
            _prompt_report_section(results),
            "",
            "## Runtime And Memory",
            "",
            f"- Total runtime: {results['runtime_s']:.2f}s",
            f"- Peak RSS: {results['memory_guard']['peak_rss_bytes'] / (1024 ** 3):.2f} GB",
            f"- Minimum available memory: {results['memory_guard']['min_available_bytes'] / (1024 ** 3):.2f} GB",
            "",
            "## Limitations",
            "",
            "- This smoke uses only the selected deterministic prompt subset for the current run.",
            "- It is teacher-forced and does not yet constitute the full Stage 5 quality benchmark.",
            "- Compression error remains empirical.",
            "- Only skipping relative to reconstructed compressed KV is rigorously certified.",
            "- CPU runtime here is experimental and must not be interpreted as production latency or speedup.",
            "- No second model, CUDA/Triton work, or unrestricted free-form generation is included in this stage.",
            "",
            "## Proposed Stage 5B",
            "",
            "- Use the same end-to-end framework on a larger prompt corpus, then add the remaining baselines only after this quality path remains stable.",
        ]
    )
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_roundtrip(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _directory_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _parse_test_count_from_log(path: Path) -> int | None:
    if not path.exists():
        return None
    matches = [int(match.group(1)) for match in re.finditer(r"Ran\s+(\d+)\s+tests", path.read_text(encoding="utf-8", errors="replace"))]
    return int(sum(matches)) if matches else None


def _load_layer0_stream_checkpoints(output_dir: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    checkpoint_root = output_dir / STAGE5_CHECKPOINT_DIR
    checkpoint_paths = sorted(path for path in checkpoint_root.glob("*__*.json") if path.is_file())
    if not checkpoint_paths:
        raise Stage5ExecutionError(f"No Stage 5 stream checkpoints were found under {checkpoint_root}.")
    by_prompt: dict[str, dict[str, dict[str, Any]]] = {}
    for path in checkpoint_paths:
        payload = _load_json(path)
        prompt_name = str(payload.get("settings", {}).get("prompt_name", ""))
        method_name = str(payload.get("method_name", ""))
        if not prompt_name or not method_name:
            continue
        by_prompt.setdefault(prompt_name, {})[method_name] = payload
    if len(by_prompt) != 1:
        raise Stage5ExecutionError(f"Expected exactly one layer-0 prompt checkpoint set, found {sorted(by_prompt)}.")
    prompt_name, method_map = next(iter(by_prompt.items()))
    missing = [method_name for method_name in STAGE5_METHODS if method_name not in method_map]
    if missing:
        raise Stage5ExecutionError(f"Missing completed layer-0 stream checkpoints for methods: {missing}.")
    for method_name, payload in method_map.items():
        if payload.get("status") != "complete":
            raise Stage5ExecutionError(f"Checkpoint {prompt_name}/{method_name} is not complete.")
    return prompt_name, method_map


def _rack_block_lengths(historical_tokens: int, block_size: int) -> list[int]:
    lengths: list[int] = []
    cursor = 0
    while cursor < historical_tokens:
        block_len = min(block_size, historical_tokens - cursor)
        lengths.append(block_len)
        cursor += block_len
    return lengths


def _rack_single_head_storage_categories(
    *,
    visible_length: int,
    recent_window: int,
    block_size: int,
    head_dim: int,
    value_dim: int,
    precision: int,
) -> dict[str, int]:
    recent_tokens = min(int(visible_length), int(recent_window))
    historical_tokens = max(int(visible_length) - recent_tokens, 0)
    block_lengths = _rack_block_lengths(historical_tokens, int(block_size))
    block_count = len(block_lengths)

    anchor_bytes = block_count * (2 * head_dim + 2 * value_dim)
    encoded_key_bytes = sum(max(block_len - 1, 0) * head_dim for block_len in block_lengths)
    encoded_value_bytes = sum(max(block_len - 1, 0) * value_dim for block_len in block_lengths)
    quantization_scale_bytes = block_count * 4
    block_metadata_bytes = block_count * 16
    index_bytes = block_count * 8
    recent_window_bytes = recent_tokens * head_dim * 2 + recent_tokens * value_dim * 2
    historical_container_bytes = 0
    serialized_container_header_bytes = 0
    if block_count > 0:
        serialized_container_header_bytes = 16
        historical_container_bytes = (
            serialized_container_header_bytes
            + index_bytes
            + block_metadata_bytes
            + anchor_bytes
            + quantization_scale_bytes
            + encoded_key_bytes
            + encoded_value_bytes
        )
    total_serialized_bytes, wrapper_header_bytes = _stage5_wrapper_total_bytes(
        recent_keys_shape=(recent_tokens, head_dim),
        recent_values_shape=(recent_tokens, value_dim),
        recent_keys_bytes=recent_tokens * head_dim * 2,
        recent_values_bytes=recent_tokens * value_dim * 2,
        historical_container_bytes=historical_container_bytes,
        recent_window=recent_window,
        block_size=block_size,
        precision=precision,
        historical_tokens=historical_tokens,
    )
    container_header_bytes = wrapper_header_bytes + serialized_container_header_bytes
    certificate_metadata_bytes = 0
    other_serialized_bytes = total_serialized_bytes - (
        anchor_bytes
        + encoded_key_bytes
        + encoded_value_bytes
        + quantization_scale_bytes
        + recent_window_bytes
        + block_metadata_bytes
        + index_bytes
        + certificate_metadata_bytes
        + container_header_bytes
    )
    if other_serialized_bytes < 0:
        raise Stage5ExecutionError(
            f"Negative residual Stage 5 rack accounting remainder for visible length {visible_length}: {other_serialized_bytes}."
        )
    return {
        "anchor_bytes": int(anchor_bytes),
        "encoded_key_bytes": int(encoded_key_bytes),
        "encoded_value_bytes": int(encoded_value_bytes),
        "quantization_scale_bytes": int(quantization_scale_bytes),
        "recent_window_bytes": int(recent_window_bytes),
        "block_metadata_bytes": int(block_metadata_bytes),
        "index_bytes": int(index_bytes),
        "certificate_metadata_bytes": int(certificate_metadata_bytes),
        "container_header_bytes": int(container_header_bytes),
        "other_serialized_bytes": int(other_serialized_bytes),
        "total_serialized_bytes": int(total_serialized_bytes),
    }


def _layer0_storage_summary(
    *,
    method_name: str,
    checkpoint_payload: dict[str, Any],
    checkpoint_summary: dict[str, Any],
) -> dict[str, Any]:
    settings = checkpoint_payload["settings"]
    sequence_length = int(settings["token_count"])
    head_dim = int(checkpoint_summary["head_dim"])
    num_kv_heads = int(checkpoint_summary["num_key_value_heads"])
    value_dim = head_dim
    full_per_token = num_kv_heads * head_dim * 2 * 2

    if method_name == "full_kv":
        token_totals = [(token_index + 1) * full_per_token for token_index in range(sequence_length)]
        category_sums = {
            "anchor_bytes": 0,
            "encoded_key_bytes": int(sum((token_index + 1) * num_kv_heads * head_dim * 2 for token_index in range(sequence_length))),
            "encoded_value_bytes": int(sum((token_index + 1) * num_kv_heads * value_dim * 2 for token_index in range(sequence_length))),
            "quantization_scale_bytes": 0,
            "recent_window_bytes": 0,
            "block_metadata_bytes": 0,
            "index_bytes": 0,
            "certificate_metadata_bytes": 0,
            "container_header_bytes": 0,
            "other_serialized_bytes": 0,
        }
    elif method_name in {"rack_kv_compression_only", "rack_kv_certified"}:
        per_token_categories = []
        for token_index in range(sequence_length):
            single_head = _rack_single_head_storage_categories(
                visible_length=token_index + 1,
                recent_window=int(settings["recent_window"]),
                block_size=int(settings["block_size"]),
                head_dim=head_dim,
                value_dim=value_dim,
                precision=int(settings["precision"]),
            )
            per_token_categories.append({key: int(value * num_kv_heads) for key, value in single_head.items()})
        token_totals = [record["total_serialized_bytes"] for record in per_token_categories]
        category_sums = {
            key: int(sum(record[key] for record in per_token_categories))
            for key in (
                "anchor_bytes",
                "encoded_key_bytes",
                "encoded_value_bytes",
                "quantization_scale_bytes",
                "recent_window_bytes",
                "block_metadata_bytes",
                "index_bytes",
                "certificate_metadata_bytes",
                "container_header_bytes",
                "other_serialized_bytes",
            )
        }
    elif method_name == "uniform_int8_kv":
        legacy_storage = checkpoint_payload["method_aggregate"]["storage"]
        token_totals = [int(value) for value in legacy_storage["token_totals_bytes"]]
        legacy_categories = dict(legacy_storage["category_sums"])
        category_sums = {
            "anchor_bytes": 0,
            "encoded_key_bytes": int(legacy_categories.get("encoded_key_bytes", 0)),
            "encoded_value_bytes": int(legacy_categories.get("encoded_value_bytes", 0)),
            "quantization_scale_bytes": int(legacy_categories.get("scales_bytes", 0)),
            "recent_window_bytes": int(legacy_categories.get("recent_window_bytes", 0)),
            "block_metadata_bytes": int(legacy_categories.get("block_page_metadata_bytes", 0)),
            "index_bytes": int(legacy_categories.get("indices_bytes", 0)),
            "certificate_metadata_bytes": 0,
            "container_header_bytes": int(legacy_categories.get("metadata_bytes", 0)),
            "other_serialized_bytes": 0,
        }
    else:
        raise Stage5ExecutionError(f"Unsupported Stage 5 layer-0 storage method {method_name!r}.")

    total_serialized_bytes = int(sum(token_totals))
    category_sum_total = int(sum(category_sums.values()))
    if category_sum_total != total_serialized_bytes:
        raise Stage5ExecutionError(
            f"Stage 5 layer-0 storage category mismatch for {method_name}: categories={category_sum_total} total={total_serialized_bytes}."
        )
    full_reference_totals = [(token_index + 1) * full_per_token for token_index in range(sequence_length)]
    ratios = [float(full_reference_totals[index] / token_totals[index]) for index in range(sequence_length)]
    savings = [float(1.0 - (token_totals[index] / full_reference_totals[index])) for index in range(sequence_length)]
    return {
        "method_name": method_name,
        "token_totals_bytes": token_totals,
        "full_reference_totals_bytes": full_reference_totals,
        "final_total_bytes": int(token_totals[-1]),
        "final_full_reference_bytes": int(full_reference_totals[-1]),
        "final_compression_ratio_vs_full_kv": float(ratios[-1]),
        "final_memory_saving_fraction_vs_full_kv": float(savings[-1]),
        "mean_compression_ratio_vs_full_kv": float(sum(ratios) / len(ratios)),
        "mean_memory_saving_fraction_vs_full_kv": float(sum(savings) / len(savings)),
        "category_sums": category_sums,
        "byte_accounting_consistent": True,
    }


def _parse_certified_resume_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "provenance_status": "unavailable_from_existing_records",
            "source_path": str(path),
        }
    text = path.read_text(encoding="utf-16")
    runtime_matches = re.findall(r"elapsed=([0-9.]+)s.*peak_rss_gb=([0-9.]+).*free_gb=([0-9.]+)", text)
    cache_matches = re.findall(r"cache_hits=(\d+)\s+remote_fetches=(\d+)", text)
    if not runtime_matches:
        return {
            "provenance_status": "unavailable_from_existing_records",
            "source_path": str(path),
        }
    runtime_s, peak_rss_gb, free_gb = runtime_matches[-1]
    cache_hits, remote_fetches = cache_matches[-1] if cache_matches else ("0", "0")
    return {
        "provenance_status": "partial_from_existing_records",
        "source_path": str(path),
        "runtime_s": float(runtime_s),
        "peak_rss_bytes": int(float(peak_rss_gb) * (1024 ** 3)),
        "min_available_bytes": int(float(free_gb) * (1024 ** 3)),
        "cache_hits": int(cache_hits),
        "remote_fetches": int(remote_fetches),
    }


def _checkpoint_provenance_audit(
    *,
    output_dir: Path,
    checkpoint_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    audit_entries = []
    for method_name, payload in sorted(checkpoint_map.items()):
        path = output_dir / STAGE5_CHECKPOINT_DIR / f"{payload['settings']['prompt_name']}__{method_name}.json"
        implementation_version = payload.get("settings", {}).get("method_implementation_version")
        legacy_compatibility = implementation_version is None
        audit_entries.append(
            {
                "relative_path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "sha256": _sha256_file(path),
                "schema_version": payload.get("schema"),
                "method_name": method_name,
                "method_implementation_version": implementation_version,
                "legacy_compatibility_status": "legacy_missing_method_implementation_version" if legacy_compatibility else "explicit_version_present",
                "exact_source_code_linkage_cryptographically_proven": False,
                "allowed_for_historical_layer0_package": True,
                "allowed_for_future_full_stage5_execution": False,
                "future_rejection_reason": (
                    "missing_method_implementation_version"
                    if legacy_compatibility
                    else "historical_layer0_validation_checkpoint_not_approved_for_future_full_stage5_reuse"
                ),
            }
        )
    return {
        "schema": "stage5_layer0_checkpoint_provenance_audit_v1",
        "entries": audit_entries,
    }


def _build_checkpoint_creation_runs(
    *,
    output_dir: Path,
    checkpoint_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    compression_profile_path = output_dir / "compression_only_layer0_128.json"
    compression_profile = _load_json(compression_profile_path) if compression_profile_path.exists() else None
    certified_profile = _parse_certified_resume_log(output_dir / "rack_certified_resume_log.txt")
    runs: list[dict[str, Any]] = []
    for method_name, payload in sorted(checkpoint_map.items()):
        run_record: dict[str, Any] = {
            "method_name": method_name,
            "checkpoint_relative_path": f"{STAGE5_CHECKPOINT_DIR}/{payload['settings']['prompt_name']}__{method_name}.json",
            "checkpoint_creation_timestamp": None,
            "runtime_s": float(payload.get("runtime_s", 0.0)),
            "peak_rss_bytes": None,
            "min_available_bytes": None,
            "cache_hits": None,
            "remote_fetches": None,
            "tensor_cache_disk_size_bytes": None,
            "method_implementation_version": payload.get("settings", {}).get("method_implementation_version"),
            "log_source": None,
            "provenance_status": "unavailable_from_existing_records",
        }
        checkpoint_path = output_dir / STAGE5_CHECKPOINT_DIR / f"{payload['settings']['prompt_name']}__{method_name}.json"
        if checkpoint_path.exists():
            run_record["checkpoint_creation_timestamp"] = checkpoint_path.stat().st_mtime
        if method_name == "rack_kv_compression_only" and compression_profile is not None:
            run_record.update(
                {
                    "peak_rss_bytes": int(compression_profile.get("peak_rss_bytes", 0)),
                    "min_available_bytes": int(compression_profile.get("min_available_bytes", 0)),
                    "cache_hits": int(compression_profile.get("tensor_cache_sources", {}).get("project_local_persistent_cache", 0)),
                    "remote_fetches": int(compression_profile.get("tensor_cache_sources", {}).get("newly_fetched_remote_range", 0)),
                    "tensor_cache_disk_size_bytes": None,
                    "log_source": str(compression_profile_path),
                    "provenance_status": "partial_from_existing_records",
                }
            )
        elif method_name == "rack_kv_certified" and certified_profile.get("provenance_status") != "unavailable_from_existing_records":
            run_record.update(
                {
                    "peak_rss_bytes": certified_profile.get("peak_rss_bytes"),
                    "min_available_bytes": certified_profile.get("min_available_bytes"),
                    "cache_hits": certified_profile.get("cache_hits"),
                    "remote_fetches": certified_profile.get("remote_fetches"),
                    "log_source": certified_profile.get("source_path"),
                    "provenance_status": certified_profile.get("provenance_status"),
                }
            )
        runs.append(run_record)
    return runs


def _build_layer0_safety_validation_report(results: dict[str, Any]) -> str:
    prompt = results["prompts"][0]
    rack_cert = prompt["method_aggregates"]["rack_kv_certified"]["certificate"]
    storage_records = results["storage_accounting_records"]
    repair_tests = results["repair_test_evidence"]
    provenance = results["aggregate_experiment_provenance"]
    lines = [
        "# Stage 5 Layer-0 Memory And Correctness Safety Validation",
        "",
        "> This artifact is a layer-0 memory and correctness safety validation. It is not a full 32-layer quality benchmark. Its diagnostic logits, NLL, and perplexity must not be interpreted as final Llama-3.1-8B model-quality results.",
        "",
        "## Experiment Scope",
        "",
        "- `experiment_scope = layer_0_safety_validation`",
        f"- Executed layers: `{results['executed_layer_indices']}`",
        f"- Full transformer layers executed: `{results['full_transformer_layers_executed']}`",
        f"- Diagnostic logits only: `{results['diagnostic_logits_only']}`",
        "- Layers 1 through 31 were not executed.",
        "- The final RMSNorm and LM head were applied after layer 0 only.",
        "- Reported logits, NLL, and perplexity are diagnostic pseudo-logit metrics only.",
        "- This artifact cannot prove end-to-end quality preservation.",
        "",
        "## Provenance",
        "",
        f"- Model: `{results['repo_id']}`",
        f"- Revision: `{results['revision']}`",
        f"- Prompt: `{prompt['name']}`",
        f"- Prompt SHA-256: `{prompt['source_sha256']}`",
        f"- Token count: {prompt['token_count']}",
        f"- Scored token count: {prompt['scored_token_count']}",
        f"- Legacy mixed attention arithmetic: `{results['legacy_mixed_attention_arithmetic']}`",
        "",
        "## Reused Model Computations",
        "",
        "- No real model stream was rerun for this repair.",
        f"- Reused checkpoints: `{', '.join(results['reused_checkpoint_methods'])}`",
        f"- Repair test command: `{repair_tests['test_command']}`",
        f"- Repair test count: {repair_tests['test_count']}",
        f"- Repair test return code: {repair_tests['test_return_code']}",
        "",
        "## Diagnostic Pseudo-Logit Metrics",
        "",
    ]
    for method_name, aggregate in prompt["method_aggregates"].items():
        lines.append(
            f"- {method_name}: mean NLL={aggregate['mean_nll']:.6f}, total NLL={aggregate['total_nll']:.6f}, "
            f"perplexity={aggregate['perplexity']:.6f}, top-1 agreement={aggregate['mean_top1_agreement']:.6f}, "
            f"mean KL={aggregate['mean_kl_divergence']:.6f}, max logit L2={aggregate['max_logit_l2_error']:.6f}"
        )
    lines.extend(
        [
            "",
            "## Corrected Certificate Aggregation",
            "",
            f"- Total certificate records: {rack_cert['total_certificate_records']}",
            f"- Total eligible query-head/block decisions: {rack_cert['total_eligible_query_head_block_decisions']}",
            f"- Prefilter-rejected query-head/block decisions: {rack_cert['prefilter_rejected_query_head_block_decisions']}",
            f"- Candidates sent to MPFR: {rack_cert['candidates_sent_to_mpfr']}",
            f"- MPFR-certified query-head/block skips: {rack_cert['mpfr_certified_query_head_block_skips']}",
            f"- MPFR-rejected candidates: {rack_cert['mpfr_rejected_candidates']}",
            f"- Unique tokens with any certified skip: {rack_cert['unique_tokens_with_any_skip']}",
            f"- Unique layers with any certified skip: {rack_cert['unique_layers_with_any_skip']}",
            f"- Unique query heads with any certified skip: {rack_cert['unique_query_heads_with_any_skip']}",
            f"- Unique KV heads with any certified skip: {rack_cert['unique_kv_heads_with_any_skip']}",
            f"- Unique logical KV-head/block combinations skipped: {rack_cert['unique_logical_kv_blocks_with_any_skip']}",
            f"- GQA physical blocks skippable by all mapped query heads: {rack_cert['gqa_physical_blocks_skippable_by_all_mapped_query_heads']}",
            f"- Physical block decodes actually avoided: {rack_cert['physical_block_decodes_actually_avoided']}",
            f"- Weighted certified skip fraction: {rack_cert['weighted_skip_fraction']:.9f}",
            f"- Skips lacking MPFR proof records: {rack_cert['skips_lacking_proof_records']}",
            "",
            "## Logical Versus Physical Skipping",
            "",
            "- `query_head_skip_decision`: one query head omits one block contribution.",
            "- `logical_kv_block_skip`: a unique KV-head/block combination skipped by at least one mapped query head.",
            "- `gqa_physical_block_skip`: a KV-head/block skipped by every query head mapped to that KV head.",
            "- `physical_decode_avoided`: a block that was never reconstructed or decoded.",
            "- This artifact validates certificate mechanics for query-head skip decisions only.",
            "- It does not prove physical random-access decode avoidance, bandwidth reduction, latency speedup, or production acceleration.",
            "",
            "## Storage Accounting",
            "",
        ]
    )
    for method_name, storage in storage_records.items():
        category_sums = storage["category_sums"]
        lines.append(
            f"- {method_name}: final bytes={storage['final_total_bytes']}, ratio={storage['final_compression_ratio_vs_full_kv']:.6f}, "
            f"byte-accounting-consistent={storage['byte_accounting_consistent']}"
        )
        lines.append(
            "  "
            f"anchors={category_sums['anchor_bytes']}, encoded_k={category_sums['encoded_key_bytes']}, "
            f"encoded_v={category_sums['encoded_value_bytes']}, scales={category_sums['quantization_scale_bytes']}, "
            f"recent={category_sums['recent_window_bytes']}, block_metadata={category_sums['block_metadata_bytes']}, "
            f"indices={category_sums['index_bytes']}, certificate_metadata={category_sums['certificate_metadata_bytes']}, "
            f"container_headers={category_sums['container_header_bytes']}, other={category_sums['other_serialized_bytes']}"
        )
    lines.extend(
        [
            "",
            "## Computational Provenance",
            "",
            f"- Maximum peak RSS across recovered computation runs: {provenance['max_peak_rss_bytes_from_actual_computation_runs']}",
            f"- Minimum free RAM across recovered computation runs: {provenance['min_available_bytes_from_actual_computation_runs']}",
            f"- Total cache hits across recovered computation runs: {provenance['total_cache_hits_from_actual_computation_runs']}",
            f"- Total remote fetches across recovered computation runs: {provenance['total_remote_fetches_from_actual_computation_runs']}",
            f"- Tensor-cache size at packaging time: {provenance['tensor_cache_disk_size_bytes_at_packaging']}",
            f"- Provenance completeness: `{provenance['provenance_status']}`",
            "",
            "## Independent Review Corrections",
            "",
            "1. Scope corrected to layer-0-only safety validation.",
            "2. Legacy mixed arithmetic flagged explicitly; future Stage 5 runs now require unified reconstructed-attention arithmetic.",
            "3. Certificate aggregation rebuilt from source records with legacy-field interpretation.",
            "4. Logical skipping separated from physical skipping.",
            "5. Storage categories rebuilt to avoid metadata double counting.",
            "6. Computation-time provenance separated from packaging-time provenance.",
            "7. Repair-test evidence included in the package.",
            "8. Checkpoint provenance audit added instead of retrofitting legacy files.",
            "9. Results, report, aggregates, and provenance rebuilt from checkpoints and source records.",
            "10. Packaging finalized after all rebuilt artifacts, with manifest self-reference excluded.",
            "",
            "## Limitations",
            "",
            "- Only layer 0 was executed.",
            "- The final RMSNorm and LM head were applied after layer 0 only.",
            "- Diagnostic logits, NLL, and perplexity are not full-model Llama-3.1-8B quality metrics.",
            "- The historical layer-0 checkpoints predate the unified reconstructed-attention arithmetic path.",
            "- Compression-only and certified arithmetic-path parity is validated by new cheap tests, not by rerunning the historical 128-token validation.",
            "- Physical decode avoidance remains unproven in this historical artifact because blocks were reconstructed before certificate decisions.",
        ]
    )
    return "\n".join(lines)


def _rebuild_layer0_safety_package(*, output_dir: Path, review_zip: Path, tensor_cache_dir: Path) -> dict[str, Any]:
    packaging_start = time.perf_counter()
    process = psutil.Process()
    packaging_start_rss = process.memory_info().rss
    legacy_results = _load_json(output_dir / STAGE5_RESULTS_JSON)
    prompt_name, checkpoint_map = _load_layer0_stream_checkpoints(output_dir)

    prompt_source = next((prompt for prompt in legacy_results["prompts"] if prompt["name"] == prompt_name), None)
    if prompt_source is None:
        raise Stage5ExecutionError(f"Prompt {prompt_name!r} was not present in the existing Stage 5 results source.")

    metric_records_by_method = {
        method_name: list(payload["metric_records"])
        for method_name, payload in checkpoint_map.items()
    }
    certificate_records = list(checkpoint_map["rack_kv_certified"].get("certificate_records", []))
    metric_path = _write_json(
        output_dir / STAGE5_METRIC_DIR / f"{prompt_name}.json",
        {
            "schema": "stage5_metric_record_v1",
            "prompt_name": prompt_name,
            "records_by_method": metric_records_by_method,
        },
    )
    cert_path = _write_json(
        output_dir / STAGE5_CERTIFICATE_DIR / f"{prompt_name}.json",
        {
            "schema": "stage5_certificate_record_v1_legacy_source",
            "prompt_name": prompt_name,
            "records": certificate_records,
        },
    )

    checkpoint_summary = dict(legacy_results["checkpoint_summary"])
    method_aggregates: dict[str, dict[str, Any]] = {}
    passkey_metrics_by_method: dict[str, Any] = {}
    for method_name in STAGE5_METHODS:
        checkpoint_payload = checkpoint_map[method_name]
        aggregate = _aggregate_metric_records(checkpoint_payload["metric_records"])
        if method_name == "full_kv":
            aggregate["delta_mean_nll_vs_full_kv"] = 0.0
            aggregate["perplexity_ratio_vs_full_kv"] = 1.0
        else:
            full_mean_nll = float(_aggregate_metric_records(checkpoint_map["full_kv"]["metric_records"])["mean_nll"])
            full_perplexity = float(_aggregate_metric_records(checkpoint_map["full_kv"]["metric_records"])["perplexity"])
            aggregate["delta_mean_nll_vs_full_kv"] = float(aggregate["mean_nll"] - full_mean_nll)
            aggregate["perplexity_ratio_vs_full_kv"] = float(aggregate["perplexity"] / full_perplexity)
        aggregate["storage"] = _layer0_storage_summary(
            method_name=method_name,
            checkpoint_payload=checkpoint_payload,
            checkpoint_summary=checkpoint_summary,
        )
        aggregate["legacy_mixed_attention_arithmetic"] = method_name in {"rack_kv_compression_only", "rack_kv_certified"}
        if method_name == "rack_kv_certified":
            aggregate["certificate"] = _aggregate_certificate_records(
                certificate_records,
                num_attention_heads=int(checkpoint_summary["num_attention_heads"]),
                num_key_value_heads=int(checkpoint_summary["num_key_value_heads"]),
            )
        method_aggregates[method_name] = aggregate
        if checkpoint_payload.get("passkey_metrics") is not None:
            passkey_metrics_by_method[method_name] = checkpoint_payload["passkey_metrics"]

    expected_scored_tokens = int(prompt_source["token_count"]) - int(STAGE5_SCORE_START) - 1
    for method_name, records in metric_records_by_method.items():
        if len(records) != expected_scored_tokens:
            raise Stage5ExecutionError(
                f"Metric count mismatch for {method_name}: expected {expected_scored_tokens}, found {len(records)}."
            )

    certificate_aggregate = method_aggregates["rack_kv_certified"]["certificate"]
    _write_json(
        output_dir / STAGE5_CERTIFICATE_DIR / f"{prompt_name}_aggregate.json",
        {
            "schema": "stage5_layer0_certificate_aggregate_v1",
            "prompt_name": prompt_name,
            "aggregate": certificate_aggregate,
        },
    )
    storage_accounting_records = {
        method_name: method_aggregates[method_name]["storage"]
        for method_name in STAGE5_METHODS
    }
    _write_json(
        output_dir / STAGE5_PROVENANCE_DIR / "storage_accounting_records.json",
        {
            "schema": "stage5_layer0_storage_accounting_v1",
            "prompt_name": prompt_name,
            "records_by_method": storage_accounting_records,
        },
    )

    checkpoint_runs = _build_checkpoint_creation_runs(output_dir=output_dir, checkpoint_map=checkpoint_map)
    packaging_end_rss = process.memory_info().rss
    packaging_run = {
        "schema": "stage5_layer0_packaging_run_v1",
        "runtime_s": time.perf_counter() - packaging_start,
        "start_rss_bytes": packaging_start_rss,
        "end_rss_bytes": packaging_end_rss,
        "peak_rss_bytes_observed": max(packaging_start_rss, packaging_end_rss),
        "tensor_cache_disk_size_bytes": _directory_size_bytes(tensor_cache_dir),
        "provenance_status": "packaging_only",
    }
    peak_values = [entry["peak_rss_bytes"] for entry in checkpoint_runs if entry.get("peak_rss_bytes") is not None]
    free_values = [entry["min_available_bytes"] for entry in checkpoint_runs if entry.get("min_available_bytes") is not None]
    cache_hits = [entry["cache_hits"] for entry in checkpoint_runs if entry.get("cache_hits") is not None]
    remote_fetches = [entry["remote_fetches"] for entry in checkpoint_runs if entry.get("remote_fetches") is not None]
    aggregate_provenance = {
        "schema": "stage5_layer0_aggregate_experiment_provenance_v1",
        "max_peak_rss_bytes_from_actual_computation_runs": max(peak_values) if peak_values else None,
        "min_available_bytes_from_actual_computation_runs": min(free_values) if free_values else None,
        "total_cache_hits_from_actual_computation_runs": int(sum(cache_hits)) if cache_hits else None,
        "total_remote_fetches_from_actual_computation_runs": int(sum(remote_fetches)) if remote_fetches else None,
        "tensor_cache_disk_size_bytes_at_packaging": packaging_run["tensor_cache_disk_size_bytes"],
        "provenance_status": (
            "partial_from_existing_records"
            if peak_values or free_values or cache_hits or remote_fetches
            else "unavailable_from_existing_records"
        ),
    }
    _write_json(
        output_dir / STAGE5_PROVENANCE_DIR / "checkpoint_creation_runs.json",
        {"schema": "stage5_layer0_checkpoint_creation_runs_v1", "runs": checkpoint_runs},
    )
    _write_json(output_dir / STAGE5_PROVENANCE_DIR / "packaging_run.json", packaging_run)
    _write_json(output_dir / STAGE5_PROVENANCE_DIR / "aggregate_experiment_provenance.json", aggregate_provenance)
    _write_json(
        output_dir / STAGE5_PROVENANCE_DIR / "dependency_versions.json",
        {
            "schema": "stage5_dependency_versions_v1",
            "dependency_versions": stage5_dependency_versions(),
        },
    )

    checkpoint_audit = _checkpoint_provenance_audit(output_dir=output_dir, checkpoint_map=checkpoint_map)
    checkpoint_audit_path = _write_json(output_dir / "checkpoint_provenance_audit.json", checkpoint_audit)

    test_command_path = output_dir / "test_command.txt"
    test_return_code_path = output_dir / "test_return_code.txt"
    repair_test_evidence = {
        "test_command": test_command_path.read_text(encoding="utf-8").strip() if test_command_path.exists() else "",
        "test_return_code": int(test_return_code_path.read_text(encoding="utf-8").strip()) if test_return_code_path.exists() else None,
        "test_count": _parse_test_count_from_log(output_dir / STAGE5_TEST_LOG),
    }
    _write_json(
        output_dir / STAGE5_PROVENANCE_DIR / "repair_test_evidence.json",
        {"schema": "stage5_layer0_repair_test_evidence_v1", **repair_test_evidence},
    )

    results = {
        "schema": STAGE5_RUN_SCHEMA,
        "result_version": STAGE5_RESULT_VERSION,
        "method_version": STAGE5_METHOD_VERSION,
        "repo_id": legacy_results["repo_id"],
        "revision": legacy_results["revision"],
        "tls_verification": legacy_results.get("tls_verification"),
        "experiment_scope": STAGE5_EXPERIMENT_SCOPE_LAYER0_SAFETY_VALIDATION,
        "full_transformer_layers_executed": False,
        "executed_layer_indices": [0],
        "diagnostic_logits_only": True,
        "legacy_mixed_attention_arithmetic": True,
        "model_streams_rerun": False,
        "reused_checkpoint_methods": list(STAGE5_METHODS),
        "configuration": {
            "recent_window": int(checkpoint_map["full_kv"]["settings"]["recent_window"]),
            "block_size": int(checkpoint_map["full_kv"]["settings"]["block_size"]),
            "tolerance": float(checkpoint_map["full_kv"]["settings"]["tolerance"]),
            "precision": int(checkpoint_map["full_kv"]["settings"]["precision"]),
            "seed": int(checkpoint_map["full_kv"]["settings"]["seed"]),
            "modified_layers": list(legacy_results["configuration"].get("modified_layers", [0])),
            "executed_layers": [0],
            "score_start_position": int(legacy_results["configuration"].get("score_start_position", STAGE5_SCORE_START)),
            "methods": list(STAGE5_METHODS),
            "selected_methods": list(STAGE5_METHODS),
            "prompt_token_count": int(prompt_source["token_count"]),
        },
        "local_model_state": legacy_results.get("local_model_state"),
        "asset_hashes": legacy_results.get("asset_hashes"),
        "asset_sources": legacy_results.get("asset_sources"),
        "model_index_sha256": legacy_results.get("model_index_sha256"),
        "model_index_source": legacy_results.get("model_index_source"),
        "checkpoint_summary": checkpoint_summary,
        "dependency_versions": stage5_dependency_versions(),
        "full_kv_equivalence_regression": legacy_results.get("full_kv_equivalence_regression"),
        "prompts": [
            {
                "name": prompt_name,
                "source_sha256": prompt_source["source_sha256"],
                "token_count": int(prompt_source["token_count"]),
                "token_ids_path": str(prompt_source["token_ids_path"]),
                "source_path": str(prompt_source["source_path"]),
                "metadata_path": str(prompt_source["metadata_path"]),
                "scored_token_count": expected_scored_tokens,
                "runtime_s": float(sum(float(checkpoint_map[method_name].get("runtime_s", 0.0)) for method_name in STAGE5_METHODS)),
                "peak_rss_bytes": aggregate_provenance["max_peak_rss_bytes_from_actual_computation_runs"],
                "method_aggregates": method_aggregates,
                "intermediate_layer_metrics": {
                    method_name: dict(checkpoint_map[method_name].get("intermediate_layer_metrics", {}))
                    for method_name in STAGE5_METHODS
                    if method_name != "full_kv"
                },
                "passkey_metrics": passkey_metrics_by_method or None,
            }
        ],
        "tensor_cache_provenance": {
            "historical_records_unavailable": True,
            "sources_count": legacy_results.get("tensor_cache_provenance", {}).get("sources_count", {}),
            "remote_bytes_fetched": legacy_results.get("tensor_cache_provenance", {}).get("remote_bytes_fetched", 0),
        },
        "storage_accounting_records": storage_accounting_records,
        "memory_guard": {
            "start_rss_bytes": None,
            "peak_rss_bytes": aggregate_provenance["max_peak_rss_bytes_from_actual_computation_runs"],
            "end_rss_bytes": None,
            "min_available_bytes": aggregate_provenance["min_available_bytes_from_actual_computation_runs"],
            "max_rss_bytes": legacy_results.get("memory_guard", {}).get("max_rss_bytes"),
            "min_free_bytes": legacy_results.get("memory_guard", {}).get("min_free_bytes"),
        },
        "runtime_s": float(sum(float(checkpoint_map[method_name].get("runtime_s", 0.0)) for method_name in STAGE5_METHODS)),
        "checkpoint_creation_runs": checkpoint_runs,
        "packaging_run": packaging_run,
        "aggregate_experiment_provenance": aggregate_provenance,
        "repair_test_evidence": repair_test_evidence,
        "scientific_scope": {
            "scope_statement": "Layer-0 memory and correctness safety validation only.",
            "diagnostic_logits": "Final RMSNorm and LM head were applied after layer 0 only.",
            "non_end_to_end": "Layers 1 through 31 were not executed in this artifact.",
            "certificate_scope": "Only skip authorization mechanics are validated here; this artifact does not prove physical decode avoidance or full-model quality preservation.",
        },
    }

    if results["prompts"][0]["method_aggregates"]["rack_kv_certified"]["certificate"]["total_certificate_records"] != len(certificate_records):
        raise Stage5ExecutionError("Certificate aggregate record count does not match the underlying certificate records.")
    if results["experiment_scope"] != STAGE5_EXPERIMENT_SCOPE_LAYER0_SAFETY_VALIDATION:
        raise Stage5ExecutionError("Layer-0 safety rebuild produced the wrong experiment scope.")

    results_path = output_dir / STAGE5_RESULTS_JSON
    _write_json(results_path, results)
    report = _build_layer0_safety_validation_report(results)
    report_path = output_dir / STAGE5_REPORT_MD
    report_path.write_text(report, encoding="utf-8")

    file_map = _collect_review_files(output_dir, include_manifest=False)
    file_map[str(checkpoint_audit_path.relative_to(output_dir)).replace("\\", "/")] = checkpoint_audit_path
    manifest = _build_manifest(
        file_map=file_map,
        source_relative_paths=STAGE5_REVIEW_SOURCE_FILES,
        source_snapshot_base_dir=Path.cwd(),
    )
    manifest_path = output_dir / STAGE5_MANIFEST_JSON
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    zip_file_map = dict(file_map)
    zip_file_map[STAGE5_MANIFEST_JSON] = manifest_path
    created_zip = _create_review_zip(file_map=zip_file_map, zip_path=review_zip)
    verification = _verify_review_zip(
        zip_path=created_zip,
        manifest=manifest,
        extra_files={STAGE5_MANIFEST_JSON: manifest_path},
    )
    return {
        "results_path": str(results_path),
        "report_path": str(report_path),
        "manifest_path": str(manifest_path),
        "checkpoint_provenance_audit_path": str(checkpoint_audit_path),
        "review_zip_path": str(created_zip),
        "review_zip_sha256": verification["zip_sha256"],
        "manifest_mismatch_count": verification["manifest_mismatch_count"],
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "dependency_versions": stage5_dependency_versions(),
    }


def _build_manifest(
    *,
    file_map: dict[str, Path],
    source_relative_paths: Sequence[str],
    source_snapshot_base_dir: Path,
) -> dict[str, any]:
    duplicates = len(file_map) != len(set(file_map.keys()))
    if duplicates:
        raise Stage5ExecutionError("Duplicate relative paths were supplied to the Stage 5 manifest.")
    for relative_path, path in file_map.items():
        if not path.exists():
            raise Stage5ExecutionError(f"Manifest file is missing: {relative_path} -> {path}")
    files = []
    for relative_path in sorted(file_map):
        path = file_map[relative_path]
        files.append(
            {
                "path": relative_path.replace("\\", "/"),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schema": "stage5_review_manifest_v1",
        "source_snapshot_sha256": _source_snapshot_sha256(
            base_dir=source_snapshot_base_dir,
            relative_paths=source_relative_paths,
        ),
        "file_count": len(files),
        "files": files,
    }


def _collect_review_files(output_dir: Path, *, include_manifest: bool = False) -> dict[str, Path]:
    file_map: dict[str, Path] = {
        STAGE5_RESULTS_JSON: output_dir / STAGE5_RESULTS_JSON,
        STAGE5_REPORT_MD: output_dir / STAGE5_REPORT_MD,
    }
    manifest_path = output_dir / STAGE5_MANIFEST_JSON
    if include_manifest and manifest_path.exists():
        file_map[STAGE5_MANIFEST_JSON] = manifest_path
    test_log_path = output_dir / STAGE5_TEST_LOG
    if test_log_path.exists():
        file_map[STAGE5_TEST_LOG] = test_log_path
    for extra_name in ("test_command.txt", "test_return_code.txt"):
        extra_path = output_dir / extra_name
        if extra_path.exists():
            file_map[extra_name] = extra_path
    for relative_source in STAGE5_REVIEW_SOURCE_FILES:
        file_map[relative_source] = Path(relative_source)
    for root_name in (STAGE5_PROMPT_DIR, STAGE5_CHECKPOINT_DIR, STAGE5_METRIC_DIR, STAGE5_CERTIFICATE_DIR, STAGE5_PROVENANCE_DIR):
        root = output_dir / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                file_map[str(path.relative_to(output_dir)).replace("\\", "/")] = path
    return file_map


def _create_review_zip(*, file_map: dict[str, Path], zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for relative_path, path in sorted(file_map.items()):
            handle.write(path, arcname=relative_path.replace("\\", "/"))
    return zip_path


def _verify_review_zip(
    *,
    zip_path: Path,
    manifest: dict[str, any],
    extra_files: dict[str, Path] | None = None,
) -> dict[str, any]:
    mismatch_count = 0
    manifest_map = {entry["path"]: entry for entry in manifest["files"]}
    extra_map = {
        relative_path.replace("\\", "/"): path
        for relative_path, path in (extra_files or {}).items()
    }
    with zipfile.ZipFile(zip_path, "r") as handle:
        seen = set()
        for info in handle.infolist():
            seen.add(info.filename)
            if info.filename not in manifest_map:
                if info.filename not in extra_map:
                    mismatch_count += 1
                continue
            payload = handle.read(info.filename)
            if hashlib.sha256(payload).hexdigest() != manifest_map[info.filename]["sha256"]:
                mismatch_count += 1
            if len(payload) != int(manifest_map[info.filename]["bytes"]):
                mismatch_count += 1
        for relative_path in manifest_map:
            if relative_path not in seen:
                mismatch_count += 1
        if extra_map:
            for normalized, path in sorted(extra_map.items()):
                if normalized not in seen:
                    mismatch_count += 1
                    continue
                payload = handle.read(normalized)
                if hashlib.sha256(payload).hexdigest() != _sha256_file(path):
                    mismatch_count += 1
                if len(payload) != path.stat().st_size:
                    mismatch_count += 1
    return {
        "zip_sha256": _sha256_file(zip_path),
        "manifest_mismatch_count": mismatch_count,
    }


def _backup_metadata_file(*, output_dir: Path, path: Path, backup_root: Path) -> Path:
    relative_path = path.relative_to(output_dir)
    backup_path = backup_root / relative_path
    if not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)
    return backup_path


def _highest_completed_layer(payload: dict[str, Any]) -> int | None:
    next_layer_index = int(payload.get("next_layer_index", 0))
    if next_layer_index <= 0:
        return None
    return next_layer_index - 1


def _full_layer_exact_per_token_from_checkpoint(payload: dict[str, Any]) -> int:
    storage = payload.get("method_aggregate", {}).get("storage")
    settings = payload.get("settings", {})
    total_layers = int(settings.get("total_layers_to_run", 0))
    if not isinstance(storage, dict):
        raise Stage5ExecutionError("Full-KV checkpoint is missing method_aggregate.storage.")
    token_totals = [int(value) for value in storage.get("token_totals_bytes", [])]
    if total_layers <= 0 or not token_totals:
        raise Stage5ExecutionError("Full-KV checkpoint does not contain enough storage data to infer exact per-layer bytes.")
    first_total = int(token_totals[0])
    if first_total % total_layers != 0:
        raise Stage5ExecutionError(
            f"Full-KV storage total {first_total} is not divisible by total_layers={total_layers}."
        )
    return int(first_total // total_layers)


def _legacy_buggy_storage_total(
    *,
    modified_token_totals: list[int],
    total_layers: int,
    modified_layers: list[int],
    full_layer_exact_per_token: int,
) -> dict[str, Any]:
    modified_set = set(int(value) for value in modified_layers)
    sequence_length = len(modified_token_totals)
    token_totals: list[int] = []
    for token_index in range(sequence_length):
        token_total = 0
        for layer_index in range(total_layers):
            if layer_index in modified_set:
                token_total += int(modified_token_totals[token_index])
            else:
                token_total += (token_index + 1) * int(full_layer_exact_per_token)
        token_totals.append(int(token_total))
    return {
        "token_totals_bytes": token_totals,
        "total_bytes": int(sum(token_totals)),
        "record_count": int(sequence_length),
        "scope": "whole_method_all_layers_cumulative_prefix_bytes_built_by_legacy_summary_loop",
    }


def _repair_storage_summary_from_checkpoints(*, output_dir: Path) -> dict[str, Any]:
    checkpoint_root = output_dir / STAGE5_CHECKPOINT_DIR
    if not checkpoint_root.exists():
        raise Stage5ExecutionError(f"Stage 5 checkpoint directory is missing: {checkpoint_root}")
    backup_root = output_dir / "accounting_repair_backup"
    checkpoint_paths = sorted(path for path in checkpoint_root.glob("*__*.json") if path.is_file())
    if not checkpoint_paths:
        raise Stage5ExecutionError(f"No Stage 5 stream checkpoints were found under {checkpoint_root}.")

    checkpoints_by_prompt_method: dict[str, dict[str, dict[str, Any]]] = {}
    highest_completed_layer_by_prompt_method: dict[str, dict[str, int | None]] = {}
    for path in checkpoint_paths:
        payload = _load_json(path)
        settings = payload.get("settings", {})
        prompt_name = str(settings.get("prompt_name", ""))
        method_name = str(payload.get("method_name", ""))
        if not prompt_name or not method_name:
            continue
        checkpoints_by_prompt_method.setdefault(prompt_name, {})[method_name] = payload
        highest_completed_layer_by_prompt_method.setdefault(prompt_name, {})[method_name] = _highest_completed_layer(payload)

    if not checkpoints_by_prompt_method:
        raise Stage5ExecutionError(f"No valid Stage 5 prompt/method checkpoints were found under {checkpoint_root}.")

    preserved_checkpoints: list[str] = []
    repaired_checkpoints: list[str] = []
    records_inspected = 0
    repair_entries: list[dict[str, Any]] = []

    for prompt_name, method_map in sorted(checkpoints_by_prompt_method.items()):
        full_checkpoint = method_map.get("full_kv")
        if full_checkpoint is None:
            raise Stage5ExecutionError(f"Prompt {prompt_name!r} is missing the full_kv checkpoint required for storage repair.")
        full_layer_exact_per_token = _full_layer_exact_per_token_from_checkpoint(full_checkpoint)
        total_layers = int(full_checkpoint.get("settings", {}).get("total_layers_to_run", 0))
        modified_layers = [int(value) for value in full_checkpoint.get("settings", {}).get("modified_layers", STAGE5_MODIFIED_LAYERS)]
        if not modified_layers:
            modified_layers = list(STAGE5_MODIFIED_LAYERS)

        for method_name, payload in sorted(method_map.items()):
            relative_path = str((checkpoint_root / f"{prompt_name}__{method_name}.json").relative_to(output_dir)).replace("\\", "/")
            preserved_checkpoints.append(relative_path)
            if method_name == "full_kv":
                continue
            if "storage_modified_token_totals" not in payload or "storage_category_sums" not in payload:
                continue

            modified_token_totals = [int(value) for value in payload.get("storage_modified_token_totals", [])]
            category_sums = {key: int(value) for key, value in payload.get("storage_category_sums", {}).items()}
            records_inspected += len(modified_token_totals)
            repaired_storage = _method_storage_summary(
                method_name=method_name,
                modified_token_totals=modified_token_totals,
                category_sums_override=category_sums,
                total_layers=total_layers,
                modified_layers=modified_layers,
                num_kv_heads=None,
                head_dim=None,
                sequence_length=int(payload["settings"]["token_count"]),
                full_layer_exact_per_token=full_layer_exact_per_token,
            )
            legacy_buggy = _legacy_buggy_storage_total(
                modified_token_totals=modified_token_totals,
                total_layers=total_layers,
                modified_layers=modified_layers,
                full_layer_exact_per_token=full_layer_exact_per_token,
            )
            repair_entry = {
                "prompt_name": prompt_name,
                "method_name": method_name,
                "checkpoint_relative_path": relative_path,
                "checkpoint_status": payload.get("status"),
                "next_layer_index": int(payload.get("next_layer_index", 0)),
                "highest_completed_layer": _highest_completed_layer(payload),
                "storage_accounting_schema": STAGE5_STORAGE_ACCOUNTING_SCHEMA,
                "legacy_mismatch_category_total_bytes": int(sum(category_sums.values())),
                "legacy_buggy_whole_method_total_bytes": int(legacy_buggy["total_bytes"]),
                "legacy_modified_token_total_record_count": int(len(modified_token_totals)),
                "legacy_buggy_token_total_record_count": int(legacy_buggy["record_count"]),
                "legacy_category_scope": "modified_layers_cumulative_serialized_bytes",
                "legacy_buggy_scope": legacy_buggy["scope"],
                "modified_layers_cumulative_total_serialized_bytes": int(repaired_storage["modified_layers_cumulative_total_serialized_bytes"]),
                "aggregate_total_serialized_bytes": int(repaired_storage["modified_layers_cumulative_total_serialized_bytes"]),
                "disjoint_category_total_bytes": int(repaired_storage["disjoint_category_total_bytes"]),
                "aggregate_category_sum_matches_total": bool(
                    int(repaired_storage["modified_layers_cumulative_total_serialized_bytes"])
                    == int(repaired_storage["disjoint_category_total_bytes"])
                ),
                "final_total_bytes_all_layers": int(repaired_storage["final_total_bytes"]),
                "final_modified_layers_serialized_bytes": int(repaired_storage["final_modified_layers_serialized_bytes"]),
                "final_unmodified_exact_bytes": int(repaired_storage["unmodified_exact_token_totals_bytes"][-1]),
                "byte_accounting_consistent": bool(repaired_storage["byte_accounting_consistent"]),
                "certificate_metadata_bytes": int(repaired_storage["category_sums"]["certificate_metadata_bytes"]),
            }
            repair_entries.append(repair_entry)

            checkpoint_path = checkpoint_root / f"{prompt_name}__{method_name}.json"
            _backup_metadata_file(output_dir=output_dir, path=checkpoint_path, backup_root=backup_root)
            payload["storage_accounting_schema"] = STAGE5_STORAGE_ACCOUNTING_SCHEMA
            payload["reporting_only_repair"] = {
                "schema": "stage5_storage_accounting_repair_checkpoint_v1",
                "repair_kind": "reporting_only",
                "model_forward_calls": 0,
                "remote_tensor_fetches": 0,
                "hidden_state_recomputations": 0,
                "mpfr_recomputations": 0,
                "legacy_mismatch_category_total_bytes": int(repair_entry["legacy_mismatch_category_total_bytes"]),
                "legacy_buggy_whole_method_total_bytes": int(repair_entry["legacy_buggy_whole_method_total_bytes"]),
                "repaired_storage_summary": repaired_storage,
            }
            if isinstance(payload.get("method_aggregate"), dict):
                payload["method_aggregate"]["storage"] = repaired_storage
            else:
                payload["repaired_storage_summary"] = repaired_storage
            checkpoint_path.write_text(json.dumps(_json_roundtrip(payload), indent=2, sort_keys=True), encoding="utf-8")
            repaired_checkpoints.append(relative_path)

    repair_report = {
        "schema": "stage5_storage_accounting_repair_v1",
        "repair_kind": "reporting_only",
        "storage_accounting_schema": STAGE5_STORAGE_ACCOUNTING_SCHEMA,
        "model_forward_calls": 0,
        "remote_tensor_fetches": 0,
        "hidden_state_recomputations": 0,
        "mpfr_recomputations": 0,
        "records_inspected": int(records_inspected),
        "prompts_found": sorted(checkpoints_by_prompt_method.keys()),
        "highest_completed_layer_by_prompt_method": highest_completed_layer_by_prompt_method,
        "preserved_checkpoints": preserved_checkpoints,
        "repaired_checkpoints": repaired_checkpoints,
        "entries": repair_entries,
    }
    report_path = output_dir / STAGE5_PROVENANCE_DIR / "storage_accounting_repair.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_json_roundtrip(repair_report), indent=2, sort_keys=True), encoding="utf-8")

    return {
        "repair_report_path": str(report_path),
        "backup_root": str(backup_root),
        "storage_accounting_schema": STAGE5_STORAGE_ACCOUNTING_SCHEMA,
        "records_inspected": int(records_inspected),
        "highest_completed_layer_by_prompt_method": highest_completed_layer_by_prompt_method,
        "preserved_checkpoints": preserved_checkpoints,
        "repaired_checkpoints": repaired_checkpoints,
        "entries": repair_entries,
        "model_forward_calls": 0,
        "remote_tensor_fetches": 0,
        "hidden_state_recomputations": 0,
        "mpfr_recomputations": 0,
    }


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    review_zip = Path(args.review_zip)
    if args.rebuild_layer0_safety_package_from_checkpoints:
        summary = _rebuild_layer0_safety_package(
            output_dir=output_dir,
            review_zip=review_zip,
            tensor_cache_dir=Path(args.tensor_cache_dir),
        )
    elif args.finalize_from_checkpoints:
        summary = _finalize_from_checkpoints(
            output_dir=output_dir,
            review_zip=review_zip,
            prompt_names=_parse_csv_names(args.prompt_names),
            excluded_partial_prompts=_parse_csv_names(args.exclude_partial_prompts),
        )
    elif args.repair_storage_summary_from_checkpoints:
        summary = _repair_storage_summary_from_checkpoints(output_dir=output_dir)
    else:
        results = run_stage5_quality_smoke(
            output_dir=output_dir,
            review_zip_path=review_zip,
            capture_dir=Path(args.capture_dir),
            tensor_cache_dir=Path(args.tensor_cache_dir),
            repo_id=args.repo_id,
            repo_revision=args.repo_revision,
            allow_insecure_tls=args.allow_insecure_tls,
            max_rss_bytes=int(args.max_rss_gb * (1024 ** 3)),
            min_free_bytes=int(args.min_free_gb * (1024 ** 3)),
            prompt_names=tuple(part.strip() for part in args.prompt_names.split(",") if part.strip()) if args.prompt_names else None,
            method_names=tuple(part.strip() for part in args.method_names.split(",") if part.strip()) if args.method_names else None,
            prompt_token_count=args.prompt_token_count if args.prompt_token_count is not None else 128,
            profile_progress_every_tokens=args.profile_progress_every,
            max_layer_index=args.max_layer_index,
            run_full_kv_equivalence_regression=not args.skip_full_kv_equivalence,
        )
        report = _build_report(results, output_dir=output_dir)
        report_path = output_dir / STAGE5_REPORT_MD
        report_path.write_text(report, encoding="utf-8")
        file_map = _collect_review_files(output_dir, include_manifest=False)
        manifest = _build_manifest(
            file_map=file_map,
            source_relative_paths=STAGE5_REVIEW_SOURCE_FILES,
            source_snapshot_base_dir=Path.cwd(),
        )
        manifest_path = output_dir / STAGE5_MANIFEST_JSON
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        zip_file_map = dict(file_map)
        zip_file_map[STAGE5_MANIFEST_JSON] = manifest_path
        review_zip = _create_review_zip(file_map=zip_file_map, zip_path=review_zip)
        verification = _verify_review_zip(
            zip_path=review_zip,
            manifest=manifest,
            extra_files={STAGE5_MANIFEST_JSON: manifest_path},
        )
        summary = {
            "results_path": str(output_dir / STAGE5_RESULTS_JSON),
            "report_path": str(report_path),
            "manifest_path": str(manifest_path),
            "review_zip_path": str(review_zip),
            "review_zip_sha256": verification["zip_sha256"],
            "manifest_mismatch_count": verification["manifest_mismatch_count"],
            "source_snapshot_sha256": manifest["source_snapshot_sha256"],
            "dependency_versions": stage5_dependency_versions(),
        }
    print(json.dumps(_json_roundtrip(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
