from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable
import zipfile

import gmpy2
import numpy as np
import psutil
import safetensors
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rack_kv.stage4 import (  # noqa: E402
    BASELINE_DEFINITIONS,
    STAGE4_BLOCK_SIZE,
    STAGE4_BUDGET_MATCHING_POLICY_VERSION,
    STAGE4_METHOD_VERSION,
    STAGE4_PRECISION,
    STAGE4_RACK_TOLERANCE,
    STAGE4_RECENT_WINDOW,
    STAGE4_RESULT_VERSION,
    STAGE4_SEED,
    BaselineCaseResult,
    build_stage4_case_context,
    run_baseline_case,
    stage4_case_result_to_dict,
    validate_stage4_capture_inputs,
    aggregate_baseline_results,
)


DEFAULT_CAPTURE_DIR = Path(".tmp/stage3_multilayer_full/capture")
DEFAULT_OUTPUT_DIR = Path(".tmp/stage4_baselines_smoke")
DEFAULT_SMOKE_RESULTS_FILENAME = "stage4_baseline_smoke_results.json"
DEFAULT_SMOKE_REPORT_FILENAME = "stage4_baseline_smoke_report.md"
DEFAULT_FULL_RESULTS_FILENAME = "stage4_baseline_full_results.json"
DEFAULT_FULL_REPORT_FILENAME = "stage4_baseline_full_report.md"
DEFAULT_GENERIC_RESULTS_FILENAME = "stage4_baseline_results.json"
DEFAULT_GENERIC_REPORT_FILENAME = "stage4_baseline_report.md"
DEFAULT_MANIFEST_FILENAME = "manifest.json"
DEFAULT_CHECKPOINT_DIRNAME = "checkpoints"
DEFAULT_SERIALIZED_DIRNAME = "serialized_cases"
DEFAULT_TEST_LOG = "stage4_test_log.txt"
DEFAULT_TEST_COMMAND = "test_command.txt"
DEFAULT_TEST_RETURN_CODE = "test_return_code.txt"
DEFAULT_CASE_CHECKPOINT_SCHEMA = "stage4_baseline_case_checkpoint_v2"
DEFAULT_RUN_SCHEMA = "stage4_baseline_run_v2"
DEFAULT_MAX_RSS_GB = 8.0
DEFAULT_MIN_FREE_GB = 2.0
DEFAULT_RUN_LABEL_SMOKE = "Stage 4A Baseline Smoke"
DEFAULT_RUN_LABEL_FULL = "Stage 4B Representative-Layer Benchmark"
DEFAULT_RUN_LABEL_GENERIC = "Stage 4 Baseline Benchmark"
DEFAULT_METHOD_MODE_PLAN: tuple[tuple[str, str], ...] = (
    ("rack_kv", "native"),
    ("full_kv", "native"),
    ("uniform_int8_kv", "native"),
    ("kivi_style", "native"),
    ("snapkv_style", "native"),
    ("snapkv_style", "matched_budget"),
    ("quest_style", "native"),
    ("quest_style", "matched_budget"),
)
REVIEW_SOURCE_FILES = (
    "rack_kv/stage4.py",
    "rack_kv/stage3.py",
    "rack_kv/stage2.py",
    "rack_kv/codec.py",
    "rack_kv/certificate.py",
    "rack_kv/rigorous.py",
    "rack_kv/llama_trace.py",
    "scripts/run_stage4_baselines_smoke.py",
    "tests/test_stage4_baselines.py",
    "tests/test_stage3_multilayer.py",
    "tests/test_stage2_integration.py",
    "tests/test_stage2b_resumable.py",
    "tests/test_codec_and_accounting.py",
    "tests/test_rigorous_certificate.py",
    "tests/test_llama_trace.py",
)


class Stage4ExecutionError(RuntimeError):
    pass


class MemoryGuardExceeded(Stage4ExecutionError):
    pass


class MemoryGuard:
    def __init__(self, *, max_rss_bytes: int, min_free_bytes: int) -> None:
        self.max_rss_bytes = int(max_rss_bytes)
        self.min_free_bytes = int(min_free_bytes)
        self.start_rss_bytes = self._rss_bytes()
        self.peak_rss_bytes = self.start_rss_bytes
        self.end_rss_bytes = self.start_rss_bytes
        self.min_available_bytes = self._available_bytes()

    @staticmethod
    def _rss_bytes() -> int:
        return int(psutil.Process().memory_info().rss)

    @staticmethod
    def _available_bytes() -> int:
        return int(psutil.virtual_memory().available)

    def check(self, stage: str) -> tuple[int, int]:
        rss = self._rss_bytes()
        free = self._available_bytes()
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self.end_rss_bytes = rss
        self.min_available_bytes = min(self.min_available_bytes, free)
        if rss > self.max_rss_bytes:
            raise MemoryGuardExceeded(
                f"Stage 4 memory guard exceeded at {stage}: RSS {rss} > hard limit {self.max_rss_bytes}."
            )
        if free < self.min_free_bytes:
            raise MemoryGuardExceeded(
                f"Stage 4 memory guard exceeded at {stage}: available memory {free} < floor {self.min_free_bytes}."
            )
        return rss, free


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonicalize_int_sequence(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values}))


def _parse_csv_ints(text: str, name: str) -> tuple[int, ...]:
    values = [piece.strip() for piece in text.split(",") if piece.strip()]
    if not values:
        raise Stage4ExecutionError(f"{name} must not be empty.")
    try:
        return _canonicalize_int_sequence(int(piece) for piece in values)
    except ValueError as exc:
        raise Stage4ExecutionError(f"{name} must contain only integers: {text}") from exc


def _default_results_filename(output_dir: Path) -> str:
    name = output_dir.name.lower()
    if "full" in name:
        return DEFAULT_FULL_RESULTS_FILENAME
    if "smoke" in name:
        return DEFAULT_SMOKE_RESULTS_FILENAME
    return DEFAULT_GENERIC_RESULTS_FILENAME


def _default_report_filename(output_dir: Path) -> str:
    name = output_dir.name.lower()
    if "full" in name:
        return DEFAULT_FULL_REPORT_FILENAME
    if "smoke" in name:
        return DEFAULT_SMOKE_REPORT_FILENAME
    return DEFAULT_GENERIC_REPORT_FILENAME


def _default_run_label(output_dir: Path) -> str:
    name = output_dir.name.lower()
    if "full" in name:
        return DEFAULT_RUN_LABEL_FULL
    if "smoke" in name:
        return DEFAULT_RUN_LABEL_SMOKE
    return DEFAULT_RUN_LABEL_GENERIC


def _default_review_zip(output_dir: Path) -> Path:
    return output_dir.parent / f"{output_dir.name}_review.zip"


def _write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
    return path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    return _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True))


def _json_normalize(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _dependency_versions() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "gmpy2_version": gmpy2.version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "safetensors_version": safetensors.__version__,
        "device": "cpu",
    }


def _source_path_map() -> dict[str, Path]:
    return {relative_path: Path(relative_path) for relative_path in REVIEW_SOURCE_FILES}


def _source_snapshot_sha256(*, base_dir: Path, relative_paths: Sequence[str]) -> str:
    lines: list[str] = []
    for relative_path in sorted(relative_paths):
        full_path = base_dir / relative_path
        if not full_path.exists():
            raise Stage4ExecutionError(f"Missing source file for source snapshot: {full_path}")
        lines.append(f"{relative_path}\t{_sha256_file(full_path)}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _result_checkpoint_path(
    *,
    output_dir: Path,
    case_key: str,
    method_name: str,
    mode: str,
) -> Path:
    return output_dir / DEFAULT_CHECKPOINT_DIRNAME / f"{case_key}__{method_name}__{mode}.json"


def _payload_path(
    *,
    output_dir: Path,
    case_key: str,
    method_name: str,
    mode: str,
) -> Path:
    return output_dir / DEFAULT_SERIALIZED_DIRNAME / case_key / f"{method_name}__{mode}.bin"


def _base_case_settings(
    *,
    context,
    method_name: str,
    mode: str,
    recent_window: int,
    block_size: int,
    tolerance: float,
    precision: int,
    seed: int,
    rack_target_bytes: int | None = None,
) -> dict[str, Any]:
    settings = {
        "schema": DEFAULT_CASE_CHECKPOINT_SCHEMA,
        "run_schema": DEFAULT_RUN_SCHEMA,
        "result_version": STAGE4_RESULT_VERSION,
        "method_version": STAGE4_METHOD_VERSION,
        "trace_path": str(context.trace_path),
        "trace_sha256": context.trace_sha256,
        "layer_index": context.layer_index,
        "record_index": context.record_index,
        "query_position": context.query_position,
        "query_local_index": context.query_local_index,
        "query_head_global": context.query_head_global,
        "kv_head_global": context.kv_head_global,
        "visible_length": context.visible_length,
        "historical_tokens": context.historical_tokens,
        "recent_window": recent_window,
        "block_size": block_size,
        "tolerance": tolerance,
        "precision": precision,
        "seed": seed,
        "method_name": method_name,
        "mode": mode,
    }
    if mode == "matched_budget":
        settings["rack_target_bytes"] = int(rack_target_bytes) if rack_target_bytes is not None else None
        settings["budget_matching_policy_version"] = STAGE4_BUDGET_MATCHING_POLICY_VERSION
    return settings


def _checkpoint_settings_with_result(
    *,
    base_settings: dict[str, Any],
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    settings = dict(base_settings)
    settings["budget_tunable"] = bool(result_payload["budget_tunable"])
    settings["budget_match_attempted"] = bool(result_payload["budget_match_attempted"])
    if settings["mode"] == "matched_budget":
        parameters = dict(result_payload.get("parameters", {}))
        settings["rack_target_bytes"] = result_payload.get("budget_target_bytes")
        settings["budget_matching_policy_version"] = parameters.get("budget_matching_policy_version")
        if settings["method_name"] == "snapkv_style":
            settings["selected_keep_count"] = parameters.get("keep_count")
        elif settings["method_name"] == "quest_style":
            settings["selected_keep_block_count"] = parameters.get("keep_block_count")
    return settings


def _validate_checkpoint_payload(
    *,
    checkpoint_path: Path,
    payload_path: Path,
    payload: dict[str, Any],
    expected_settings: dict[str, Any],
) -> dict[str, Any]:
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise Stage4ExecutionError(f"Malformed Stage 4 checkpoint settings: {checkpoint_path}")
    for key, expected_value in expected_settings.items():
        if settings.get(key) != expected_value:
            raise Stage4ExecutionError(
                f"Stage 4 checkpoint settings mismatch for {checkpoint_path}: {key} expected {expected_value!r}, got {settings.get(key)!r}."
            )
    result = payload.get("result")
    if not isinstance(result, dict):
        raise Stage4ExecutionError(f"Malformed Stage 4 checkpoint result: {checkpoint_path}")
    if settings.get("mode") == "matched_budget":
        parameters = dict(result.get("parameters", {}))
        if settings.get("method_name") == "snapkv_style":
            if settings.get("selected_keep_count") != parameters.get("keep_count"):
                raise Stage4ExecutionError(f"SnapKV-style matched-budget checkpoint keep_count mismatch: {checkpoint_path}")
        elif settings.get("method_name") == "quest_style":
            if settings.get("selected_keep_block_count") != parameters.get("keep_block_count"):
                raise Stage4ExecutionError(f"Quest-style matched-budget checkpoint keep_block_count mismatch: {checkpoint_path}")
    expected_sha = str(result["payload_sha256"])
    if not payload_path.exists():
        raise Stage4ExecutionError(f"Serialized payload file is missing for checkpoint: {payload_path}")
    actual_sha = _sha256_file(payload_path)
    if actual_sha != expected_sha:
        raise Stage4ExecutionError(
            f"Serialized payload SHA-256 mismatch for {payload_path}: expected {expected_sha}, got {actual_sha}."
        )
    return payload


def _load_checkpoint(
    *,
    checkpoint_path: Path,
    payload_path: Path,
    expected_settings: dict[str, Any],
) -> dict[str, Any] | None:
    temp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    if temp_path.exists():
        try:
            temp_path.unlink()
        except OSError:
            pass
    if not checkpoint_path.exists():
        return None
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return _validate_checkpoint_payload(
        checkpoint_path=checkpoint_path,
        payload_path=payload_path,
        payload=payload,
        expected_settings=expected_settings,
    )


def _save_checkpoint(
    *,
    checkpoint_path: Path,
    payload_path: Path,
    settings: dict[str, Any],
    result_payload: dict[str, Any],
    serialized_payload: bytes,
) -> tuple[Path, Path]:
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_temp = payload_path.with_name(payload_path.name + ".tmp")
    try:
        payload_temp.write_bytes(serialized_payload)
        os.replace(payload_temp, payload_path)
    finally:
        if payload_temp.exists():
            try:
                payload_temp.unlink()
            except OSError:
                pass
    checkpoint_payload = {
        "settings": settings,
        "result": result_payload,
    }
    _write_json_atomic(checkpoint_path, checkpoint_payload)
    return checkpoint_path, payload_path


def _gc_cleanup() -> None:
    gc.collect()


def _method_mode_plan() -> tuple[tuple[str, str], ...]:
    return DEFAULT_METHOD_MODE_PLAN


def _stage4_case_order(
    *,
    layer_indices: tuple[int, ...],
    query_positions: tuple[int, ...],
    query_local_indices: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (layer_index, query_position, query_local_index)
        for layer_index in layer_indices
        for query_position in query_positions
        for query_local_index in query_local_indices
    )


def _validate_requested_selection(
    *,
    traces: dict[int, Any],
    layer_indices: tuple[int, ...],
    query_positions: tuple[int, ...],
    query_local_indices: tuple[int, ...],
) -> dict[str, Any]:
    if not layer_indices:
        raise Stage4ExecutionError("At least one layer must be requested.")
    reference_trace = traces[layer_indices[0]]
    max_query_local_index = reference_trace.query_head_count - 1
    for query_local_index in query_local_indices:
        if query_local_index < 0 or query_local_index > max_query_local_index:
            raise Stage4ExecutionError(
                f"Query local index {query_local_index} is out of range for trace query_head_count {reference_trace.query_head_count}."
            )
    for layer_index in layer_indices:
        trace = traces[layer_index]
        if trace.selected_query_heads != reference_trace.selected_query_heads:
            raise Stage4ExecutionError(f"Selected query heads differ across traces; layer {layer_index} does not match the reference trace.")
        if trace.selected_kv_heads != reference_trace.selected_kv_heads:
            raise Stage4ExecutionError(f"Selected KV heads differ across traces; layer {layer_index} does not match the reference trace.")
        if trace.query_to_kv_heads != reference_trace.query_to_kv_heads:
            raise Stage4ExecutionError(f"Query-to-KV mapping differs across traces; layer {layer_index} does not match the reference trace.")
        for query_position in query_positions:
            if query_position not in trace.query_positions:
                raise Stage4ExecutionError(f"Query position {query_position} is not present in layer {layer_index} trace.")
    return {
        "selected_query_heads": list(reference_trace.selected_query_heads),
        "selected_kv_heads": list(reference_trace.selected_kv_heads),
        "query_to_kv_heads": list(reference_trace.query_to_kv_heads),
    }


def _layer_report_entries(*, capture_report: dict[str, Any], layer_indices: tuple[int, ...]) -> list[dict[str, Any]]:
    entries = []
    for entry in capture_report.get("layers", []):
        layer_index = int(entry["layer_index"])
        if layer_index not in layer_indices:
            continue
        entries.append(
            {
                "layer_index": layer_index,
                "trace_path": str(entry["trace_path"]),
                "trace_sha256": str(entry["trace_sha256"]),
                "queries_shape": list(entry["queries_shape"]),
                "final_keys_shape": list(entry["final_keys_shape"]),
                "final_values_shape": list(entry["final_values_shape"]),
                "model_head_outputs_shape": list(entry["model_head_outputs_shape"]),
                "selected_query_heads": list(entry["selected_query_heads"]),
                "selected_kv_heads": list(entry["selected_kv_heads"]),
                "query_to_kv_heads": list(entry["query_to_kv_heads"]),
                "stock_projected_output_max_abs_diff": entry["stock_forward_comparison"]["projected_output_max_abs_diff"],
                "stock_decoder_output_max_abs_diff": entry["stock_forward_comparison"]["decoder_output_max_abs_diff"],
                "stock_cache_key_max_abs_diff": entry["cache_comparison"]["key_max_abs_diff"],
                "stock_cache_value_max_abs_diff": entry["cache_comparison"]["value_max_abs_diff"],
                "compact_replay_max_abs_diff": entry["compact_replay"]["max_abs_diff"],
            }
        )
    return sorted(entries, key=lambda item: item["layer_index"])


def _aggregate_by_layer(case_results: list[BaselineCaseResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for layer_index in sorted({result.layer_index for result in case_results}):
        layer_results = [result for result in case_results if result.layer_index == layer_index]
        summary[str(layer_index)] = aggregate_baseline_results(layer_results)
    return summary


def _build_case_descriptor(
    *,
    trace,
    query_position: int,
    query_local_index: int,
    recent_window: int,
):
    record_index = trace.query_positions.index(query_position)
    visible_length = int(trace.visible_lengths[record_index])
    recent_exact_tokens = min(int(recent_window), visible_length)
    historical_tokens = max(0, visible_length - recent_exact_tokens)
    query_head_global = int(trace.selected_query_heads[query_local_index])
    kv_local_index = int(trace.query_to_kv_heads[query_local_index])
    kv_head_global = int(trace.selected_kv_heads[kv_local_index])
    return SimpleNamespace(
        trace_path=trace.trace_path,
        trace_sha256=trace.trace_sha256,
        layer_index=int(trace.layer_index),
        record_index=int(record_index),
        query_position=int(query_position),
        query_local_index=int(query_local_index),
        query_head_global=query_head_global,
        kv_head_global=kv_head_global,
        visible_length=visible_length,
        historical_tokens=historical_tokens,
    )


def _load_complete_case_results(
    *,
    trace,
    query_position: int,
    query_local_index: int,
    output_dir: Path,
    recent_window: int,
    block_size: int,
    tolerance: float,
    precision: int,
    seed: int,
) -> list[dict[str, Any]] | None:
    descriptor = _build_case_descriptor(
        trace=trace,
        query_position=query_position,
        query_local_index=query_local_index,
        recent_window=recent_window,
    )
    case_key = f"{descriptor.layer_index}_pos{descriptor.query_position}_ql{descriptor.query_local_index}"
    rack_target_bytes: int | None = None
    results: list[dict[str, Any]] = []
    for method_name, mode in _method_mode_plan():
        payload_path = _payload_path(output_dir=output_dir, case_key=case_key, method_name=method_name, mode=mode)
        checkpoint_path = _result_checkpoint_path(output_dir=output_dir, case_key=case_key, method_name=method_name, mode=mode)
        if not checkpoint_path.exists():
            return None
        expected_settings = _base_case_settings(
            context=descriptor,
            method_name=method_name,
            mode=mode,
            recent_window=recent_window,
            block_size=block_size,
            tolerance=tolerance,
            precision=precision,
            seed=seed,
            rack_target_bytes=rack_target_bytes if mode == "matched_budget" else None,
        )
        reused = _load_checkpoint(
            checkpoint_path=checkpoint_path,
            payload_path=payload_path,
            expected_settings=expected_settings,
        )
        if reused is None:
            return None
        result_payload = reused["result"]
        if method_name == "rack_kv" and mode == "native":
            rack_target_bytes = int(result_payload["total_serialized_bytes"])
        results.append(result_payload)
    return results


def _load_all_case_results_from_checkpoints(
    *,
    traces: dict[int, Any],
    case_order: Sequence[tuple[int, int, int]],
    output_dir: Path,
    recent_window: int,
    block_size: int,
    tolerance: float,
    precision: int,
    seed: int,
) -> list[dict[str, Any]]:
    all_case_payloads: list[dict[str, Any]] = []
    for layer_index, query_position, query_local_index in case_order:
        trace = traces[layer_index]
        case_results = _load_complete_case_results(
            trace=trace,
            query_position=query_position,
            query_local_index=query_local_index,
            output_dir=output_dir,
            recent_window=recent_window,
            block_size=block_size,
            tolerance=tolerance,
            precision=precision,
            seed=seed,
        )
        if case_results is None or len(case_results) != len(_method_mode_plan()):
            raise Stage4ExecutionError(
                f"Stage 4 checkpoint collection is incomplete for layer={layer_index} position={query_position} query_local_index={query_local_index}."
            )
        all_case_payloads.extend(case_results)
    return all_case_payloads


def _run_one_case(
    *,
    traces: dict[int, Any],
    layer_index: int,
    query_position: int,
    query_local_index: int,
    output_dir: Path,
    recent_window: int,
    block_size: int,
    tolerance: float,
    precision: int,
    seed: int,
    guard: MemoryGuard,
) -> list[dict[str, Any]]:
    trace = traces[layer_index]
    descriptor = _build_case_descriptor(
        trace=trace,
        query_position=query_position,
        query_local_index=query_local_index,
        recent_window=recent_window,
    )
    case_key = f"{descriptor.layer_index}_pos{descriptor.query_position}_ql{descriptor.query_local_index}"

    guard.check(f"before_case:{case_key}")

    reused_case = _load_complete_case_results(
        trace=trace,
        query_position=query_position,
        query_local_index=query_local_index,
        output_dir=output_dir,
        recent_window=recent_window,
        block_size=block_size,
        tolerance=tolerance,
        precision=precision,
        seed=seed,
    )
    if reused_case is not None:
        _gc_cleanup()
        guard.check(f"after_cleanup:{case_key}")
        return reused_case

    context = build_stage4_case_context(
        trace=trace,
        record_index=descriptor.record_index,
        query_local_index=query_local_index,
        recent_window=recent_window,
        precision=precision,
    )
    results: list[dict[str, Any]] = []
    rack_target_bytes: int | None = None
    for method_name, mode in _method_mode_plan():
        payload_path = _payload_path(output_dir=output_dir, case_key=case_key, method_name=method_name, mode=mode)
        checkpoint_path = _result_checkpoint_path(output_dir=output_dir, case_key=case_key, method_name=method_name, mode=mode)
        expected_settings = _base_case_settings(
            context=context,
            method_name=method_name,
            mode=mode,
            recent_window=recent_window,
            block_size=block_size,
            tolerance=tolerance,
            precision=precision,
            seed=seed,
            rack_target_bytes=rack_target_bytes if mode == "matched_budget" else None,
        )
        reused = _load_checkpoint(
            checkpoint_path=checkpoint_path,
            payload_path=payload_path,
            expected_settings=expected_settings,
        ) if checkpoint_path.exists() else None
        if reused is None:
            result, payload_bytes = run_baseline_case(
                context=context,
                method_name=method_name,
                mode=mode,
                trace=trace if method_name == "rack_kv" else None,
                recent_window=recent_window,
                block_size=block_size,
                tolerance=tolerance,
                precision=precision,
                payload_relative_path=str(payload_path.relative_to(output_dir)),
                rack_target_bytes=rack_target_bytes if mode == "matched_budget" else None,
            )
            result_payload = _json_normalize(stage4_case_result_to_dict(result))
            checkpoint_settings = _checkpoint_settings_with_result(
                base_settings=expected_settings,
                result_payload=result_payload,
            )
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                payload_path=payload_path,
                settings=checkpoint_settings,
                result_payload=result_payload,
                serialized_payload=payload_bytes,
            )
        else:
            result_payload = reused["result"]
        if method_name == "rack_kv" and mode == "native":
            rack_target_bytes = int(result_payload["total_serialized_bytes"])
        results.append(result_payload)
        del result_payload
        if reused is None:
            del result
            del payload_bytes
            del checkpoint_settings
        else:
            del reused
        _gc_cleanup()
        guard.check(f"after_{method_name}_{mode}:{case_key}")

    del context
    _gc_cleanup()
    guard.check(f"after_cleanup:{case_key}")
    return results


def _run_benchmark(
    *,
    capture_dir: Path,
    output_dir: Path,
    run_label: str,
    layer_indices: tuple[int, ...],
    query_positions: tuple[int, ...],
    query_local_indices: tuple[int, ...],
    recent_window: int,
    block_size: int,
    tolerance: float,
    precision: int,
    seed: int,
    max_rss_bytes: int,
    min_free_bytes: int,
) -> dict[str, Any]:
    traces, capture_report = validate_stage4_capture_inputs(
        capture_dir=capture_dir,
        layer_indices=layer_indices,
    )
    selection_info = _validate_requested_selection(
        traces=traces,
        layer_indices=layer_indices,
        query_positions=query_positions,
        query_local_indices=query_local_indices,
    )
    case_order = _stage4_case_order(
        layer_indices=layer_indices,
        query_positions=query_positions,
        query_local_indices=query_local_indices,
    )
    guard = MemoryGuard(max_rss_bytes=max_rss_bytes, min_free_bytes=min_free_bytes)
    guard.check("before_stage4_benchmark")
    total_cases = len(case_order)
    start_time = time.perf_counter()
    for index, (layer_index, query_position, query_local_index) in enumerate(case_order, start=1):
        case_payloads = _run_one_case(
            traces=traces,
            layer_index=layer_index,
            query_position=query_position,
            query_local_index=query_local_index,
            output_dir=output_dir,
            recent_window=recent_window,
            block_size=block_size,
            tolerance=tolerance,
            precision=precision,
            seed=seed,
            guard=guard,
        )
        if len(case_payloads) != len(_method_mode_plan()):
            raise Stage4ExecutionError(
                f"Stage 4 case {layer_index}:{query_position}:{query_local_index} produced {len(case_payloads)} results; expected {len(_method_mode_plan())}."
            )
        elapsed = time.perf_counter() - start_time
        average_per_case = elapsed / index if index > 0 else 0.0
        remaining = max(total_cases - index, 0)
        eta_seconds = average_per_case * remaining
        query_head_global = selection_info["selected_query_heads"][query_local_index]
        print(
            f"{index}/{total_cases} layer={layer_index} position={query_position} query_head={query_head_global} "
            f"elapsed={elapsed:.1f}s eta={eta_seconds:.1f}s",
            flush=True,
        )

    _gc_cleanup()
    guard.check("before_result_collection")
    all_case_payloads = _load_all_case_results_from_checkpoints(
        traces=traces,
        case_order=case_order,
        output_dir=output_dir,
        recent_window=recent_window,
        block_size=block_size,
        tolerance=tolerance,
        precision=precision,
        seed=seed,
    )
    case_results = [BaselineCaseResult(**payload) for payload in all_case_payloads]
    aggregate = aggregate_baseline_results(case_results)
    aggregate_by_layer = _aggregate_by_layer(case_results)
    expected_result_count = total_cases * len(_method_mode_plan())
    if len(case_results) != expected_result_count:
        raise Stage4ExecutionError(
            f"Stage 4 benchmark produced {len(case_results)} results; expected {expected_result_count}."
        )

    return {
        "run_schema": DEFAULT_RUN_SCHEMA,
        "run_label": run_label,
        "result_version": STAGE4_RESULT_VERSION,
        "method_version": STAGE4_METHOD_VERSION,
        "case_checkpoint_schema": DEFAULT_CASE_CHECKPOINT_SCHEMA,
        "budget_matching_policy_version": STAGE4_BUDGET_MATCHING_POLICY_VERSION,
        "capture_dir": str(capture_dir),
        "output_dir": str(output_dir),
        "layers": list(layer_indices),
        "query_positions": list(query_positions),
        "query_local_indices": list(query_local_indices),
        "case_count": total_cases,
        "cases_completed": total_cases,
        "expected_result_count": expected_result_count,
        "actual_result_count": len(case_results),
        "method_mode_plan": [
            {"method_name": method_name, "mode": mode}
            for method_name, mode in _method_mode_plan()
        ],
        "methods": {
            key: {
                **{
                    field: value
                    for field, value in BASELINE_DEFINITIONS[key].__dict__.items()
                },
                "deviations": list(BASELINE_DEFINITIONS[key].deviations),
            }
            for key in BASELINE_DEFINITIONS
        },
        "optional_deferred_methods": {
            "delta_or_ojakv": {
                "implemented": False,
                "reason": "Deferred in Stage 4 because a faithful source-backed offline implementation was not completed from the frozen local record without delaying the representative-layer benchmark.",
                "candidate_citation_keys": ["hao2026deltakv", "zhu2025ojakv"],
            }
        },
        "configuration": {
            "recent_window": recent_window,
            "block_size": block_size,
            "rack_kv_tolerance": tolerance,
            "precision": precision,
            "seed": seed,
            "selected_query_heads": selection_info["selected_query_heads"],
            "selected_kv_heads": selection_info["selected_kv_heads"],
            "query_to_kv_heads": selection_info["query_to_kv_heads"],
        },
        "capture_validation": {
            "layer_summaries": _layer_report_entries(
                capture_report=capture_report,
                layer_indices=layer_indices,
            ),
            "tls_verification": bool(capture_report["tls_verification"]),
            "checkpoint_repo": capture_report["checkpoint"]["repo_id"],
            "checkpoint_revision": capture_report["checkpoint"]["revision"],
        },
        "case_results": [stage4_case_result_to_dict(result) for result in case_results],
        "aggregate_by_method_mode": aggregate,
        "aggregate_by_layer_method_mode": aggregate_by_layer,
        "memory_guard": {
            "start_rss_bytes": guard.start_rss_bytes,
            "peak_rss_bytes": guard.peak_rss_bytes,
            "end_rss_bytes": guard.end_rss_bytes,
            "min_available_bytes": guard.min_available_bytes,
            "max_rss_bytes": guard.max_rss_bytes,
            "min_free_bytes": guard.min_free_bytes,
        },
        "scientific_scope": {
            "offline_runtime_only": "CPU offline runtime is not production inference latency.",
            "certificate_scope": "Only RACK-KV skipping relative to reconstructed compressed KV remains rigorously certified.",
            "compression_scope": "Compression errors remain empirical for all methods.",
            "novelty_scope": "No novelty conclusion follows from this benchmark alone.",
            "baseline_scope": "KIVI-style, SnapKV-style, and Quest-style remain source-inspired approximations rather than official implementations of the original systems.",
        },
    }


def _collect_review_files(
    *,
    output_dir: Path,
    results_filename: str,
    report_filename: str,
) -> list[Path]:
    files = [
        output_dir / results_filename,
        output_dir / report_filename,
        output_dir / DEFAULT_TEST_COMMAND,
        output_dir / DEFAULT_TEST_RETURN_CODE,
        output_dir / DEFAULT_TEST_LOG,
        output_dir / DEFAULT_MANIFEST_FILENAME,
    ]
    files.extend(sorted((output_dir / DEFAULT_CHECKPOINT_DIRNAME).glob("*.json")))
    files.extend(sorted((output_dir / DEFAULT_SERIALIZED_DIRNAME).rglob("*.bin")))
    return [path for path in files if path.exists()]


def _build_manifest(
    *,
    file_map: dict[str, Path],
    source_relative_paths: Sequence[str],
    source_snapshot_base_dir: Path,
) -> dict[str, Any]:
    duplicate_paths = [path for path in file_map if list(file_map.keys()).count(path) > 1]
    if duplicate_paths:
        raise Stage4ExecutionError(f"Duplicate relative paths in manifest: {sorted(set(duplicate_paths))}")
    entries: list[dict[str, Any]] = []
    for relative_path, path in sorted(file_map.items()):
        if not path.exists():
            raise Stage4ExecutionError(f"Cannot add missing file to manifest: {path}")
        entries.append(
            {
                "relative_path": relative_path.replace("/", "\\"),
                "sha256": _sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return {
        "files": entries,
        "file_count": len(entries),
        "source_revision": None,
        "source_files": list(sorted(source_relative_paths)),
        "source_snapshot_sha256": _source_snapshot_sha256(
            base_dir=source_snapshot_base_dir,
            relative_paths=tuple(sorted(source_relative_paths)),
        ),
    }


def _outer_manifest_file_map(
    *,
    output_dir: Path,
    results_filename: str,
    report_filename: str,
) -> dict[str, Path]:
    file_map: dict[str, Path] = {}
    for review_file in _collect_review_files(
        output_dir=output_dir,
        results_filename=results_filename,
        report_filename=report_filename,
    ):
        if review_file.name == DEFAULT_MANIFEST_FILENAME:
            continue
        relative_path = str(review_file.relative_to(output_dir)).replace("\\", "/")
        if relative_path in file_map:
            raise Stage4ExecutionError(f"Duplicate output relative path in manifest: {relative_path}")
        file_map[relative_path] = review_file
    for relative_path, source_path in _source_path_map().items():
        normalized = relative_path.replace("\\", "/")
        if normalized in file_map:
            raise Stage4ExecutionError(f"Duplicate source relative path in manifest: {normalized}")
        file_map[normalized] = source_path
    return file_map


def _create_review_zip(
    *,
    output_dir: Path,
    zip_path: Path,
    results_filename: str,
    report_filename: str,
) -> tuple[Path, str, dict[str, Any]]:
    package_dir = output_dir / "review_package"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    outer_file_map = _outer_manifest_file_map(
        output_dir=output_dir,
        results_filename=results_filename,
        report_filename=report_filename,
    )
    for relative_path, source_path in outer_file_map.items():
        target = package_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)

    package_file_map: dict[str, Path] = {}
    for path in sorted(package_dir.rglob("*")):
        if path.is_file():
            relative_path = str(path.relative_to(package_dir)).replace("\\", "/")
            if relative_path == DEFAULT_MANIFEST_FILENAME:
                continue
            if relative_path in package_file_map:
                raise Stage4ExecutionError(f"Duplicate packaged relative path: {relative_path}")
            package_file_map[relative_path] = path

    inner_manifest = _build_manifest(
        file_map=package_file_map,
        source_relative_paths=tuple(_source_path_map().keys()),
        source_snapshot_base_dir=package_dir,
    )
    _write_json_atomic(package_dir / DEFAULT_MANIFEST_FILENAME, inner_manifest)

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(package_dir)).replace("\\", "/"))

    with zipfile.ZipFile(zip_path, "r") as archive:
        names = set(archive.namelist())
        if DEFAULT_MANIFEST_FILENAME not in names:
            raise Stage4ExecutionError("Review ZIP is missing manifest.json.")
        manifest_payload = json.loads(archive.read(DEFAULT_MANIFEST_FILENAME).decode("utf-8"))
        mismatches = []
        for entry in manifest_payload["files"]:
            relative_path = str(entry["relative_path"]).replace("\\", "/")
            if relative_path not in names:
                mismatches.append({"missing": relative_path})
                continue
            actual_sha = hashlib.sha256(archive.read(relative_path)).hexdigest()
            if actual_sha != entry["sha256"]:
                mismatches.append(
                    {
                        "path": relative_path,
                        "expected": entry["sha256"],
                        "actual": actual_sha,
                    }
                )
        if mismatches:
            raise Stage4ExecutionError(f"Review ZIP packaged hash mismatches: {mismatches[:5]}")
    return zip_path, _sha256_file(zip_path), inner_manifest


def _stats_label(value: float | None, *, scale: float = 1.0, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value * scale}{suffix}"


def _render_aggregate_table(entries: dict[str, Any]) -> list[str]:
    lines = [
        "| method | mode | cases | mean L2 | median L2 | p95 L2 | max L2 | mean rel L2 | mean max abs | mean cosine | mean bytes | mean ratio | mean saving | mean retained token | mean retained block | mean budget diff bytes | mean budget diff % |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, entry in sorted(entries.items()):
        budget_pct = entry["mean_budget_rel_diff_fraction"] * 100.0 if entry["mean_budget_rel_diff_fraction"] is not None else None
        lines.append(
            f"| {entry['method_name']} | {entry['mode']} | {entry['case_count']} | "
            f"{entry['mean_attention_output_l2_error']} | {entry['median_attention_output_l2_error']} | "
            f"{entry['p95_attention_output_l2_error']} | {entry['max_attention_output_l2_error']} | "
            f"{entry['mean_relative_l2_error']} | {entry['mean_max_abs_component_error']} | "
            f"{entry['mean_cosine_similarity']} | {entry['mean_serialized_bytes']} | "
            f"{entry['mean_compression_ratio_vs_full_kv']} | {entry['mean_memory_saving_fraction_vs_full_kv']} | "
            f"{entry['mean_retained_token_fraction']} | "
            f"{entry['mean_retained_block_fraction'] if entry['mean_retained_block_fraction'] is not None else 'n/a'} | "
            f"{entry['mean_budget_abs_diff_bytes'] if entry['mean_budget_abs_diff_bytes'] is not None else 'n/a'} | "
            f"{budget_pct if budget_pct is not None else 'n/a'} |"
        )
    return lines


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        f"# {results['run_label']}",
        "",
        "This is an offline baseline comparison over frozen Stage 3 real traces.",
        "It reuses existing captures and does not recapture or download model weights.",
        "",
        "## Method Definitions",
        "",
        "| method | category | citation key | fidelity | budget tunable | source label | deviations |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for method_name in ("full_kv", "rack_kv", "uniform_int8_kv", "kivi_style", "snapkv_style", "quest_style"):
        definition = results["methods"][method_name]
        deviations = "; ".join(definition["deviations"]) if definition["deviations"] else "None"
        lines.append(
            f"| {definition['display_name']} | {definition['category']} | {definition['citation_key'] or 'n/a'} | "
            f"{definition['fidelity_status']} | {definition['budget_tunable']} | {definition['source_label']} | {deviations} |"
        )
    lines.extend(
        [
            "",
            "## Benchmark Configuration",
            "",
            f"- Capture directory: `{results['capture_dir']}`",
            f"- Layers: `{results['layers']}`",
            f"- Query positions: `{results['query_positions']}`",
            f"- Query local indices: `{results['query_local_indices']}`",
            f"- Selected query heads: `{results['configuration']['selected_query_heads']}`",
            f"- Selected KV heads: `{results['configuration']['selected_kv_heads']}`",
            f"- Query-to-KV mapping: `{results['configuration']['query_to_kv_heads']}`",
            f"- RACK-KV configuration: `W={results['configuration']['recent_window']}`, `B={results['configuration']['block_size']}`, tolerance `{results['configuration']['rack_kv_tolerance']}`, precision `{results['configuration']['precision']}`, seed `{results['configuration']['seed']}`",
            f"- Query cases: `{results['case_count']}`",
            f"- Expected result count: `{results['expected_result_count']}`",
            f"- Actual result count: `{results['actual_result_count']}`",
            "",
            "## Capture Validation",
            "",
            "| layer | trace sha256 | queries shape | final keys shape | final values shape | projected abs diff | decoder abs diff | key-cache abs diff | value-cache abs diff | compact replay abs diff |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for entry in results["capture_validation"]["layer_summaries"]:
        lines.append(
            f"| {entry['layer_index']} | {entry['trace_sha256']} | {entry['queries_shape']} | {entry['final_keys_shape']} | {entry['final_values_shape']} | "
            f"{entry['stock_projected_output_max_abs_diff']} | {entry['stock_decoder_output_max_abs_diff']} | "
            f"{entry['stock_cache_key_max_abs_diff']} | {entry['stock_cache_value_max_abs_diff']} | "
            f"{entry['compact_replay_max_abs_diff']} |"
        )
    lines.extend(
        [
            "",
            "## Overall Aggregate Results",
            "",
            *_render_aggregate_table(results["aggregate_by_method_mode"]),
            "",
            "## Per-Layer Aggregate Results",
            "",
        ]
    )
    for layer_key, layer_summary in sorted(results["aggregate_by_layer_method_mode"].items(), key=lambda item: int(item[0])):
        lines.extend(
            [
                f"### Layer {layer_key}",
                "",
                *_render_aggregate_table(layer_summary),
                "",
            ]
        )
    lines.extend(
        [
            "## RACK-KV Certification Summary",
            "",
            "| scope | mode | cases | total candidate blocks | total certified skipped blocks | weighted certified skip fraction | cases with any skipping | max case skip fraction | max certificate upper bound | max observed skipping error | rigorous interval violations | approximate observed violations | false-safe count | reference decomposition violations | model-relative decomposition violations | numerical fallbacks |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for scope_label, scope_summary in [("aggregate", results["aggregate_by_method_mode"]), *[(f"layer {layer}", summary) for layer, summary in sorted(results["aggregate_by_layer_method_mode"].items(), key=lambda item: int(item[0]))]]:
        for mode in ("native",):
            entry = scope_summary.get(f"rack_kv:{mode}")
            if entry is None:
                continue
            lines.append(
                f"| {scope_label} | {mode} | {entry['case_count']} | "
                f"{entry['total_candidate_blocks']} | {entry['total_certified_skipped_blocks']} | "
                f"{entry['weighted_certified_skip_fraction']} | {entry['cases_with_any_certified_skipping']} | "
                f"{entry['max_case_skip_fraction']} | {entry['max_certificate_upper_bound']} | "
                f"{entry['max_observed_skipping_error']} | {entry['rigorous_interval_violation_count']} | "
                f"{entry['approximate_observed_violation_count']} | {entry['false_safe_count']} | "
                f"{entry['reference_decomposition_violation_count']} | {entry['model_relative_decomposition_violation_count']} | "
                f"{entry['numerical_fallback_count']} |"
            )
    lines.extend(
        [
            "",
            "## Per-Case Results",
            "",
            "| case | method | mode | bytes | ratio vs full KV | memory saving | L2 error | rel L2 | max abs | cosine | retained token fraction | retained block/page fraction | budget tunable | budget attempted | budget possible | budget target | budget diff bytes | budget diff % | within ±2% |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for case in results["case_results"]:
        budget_pct = case["budget_rel_diff_fraction"] * 100.0 if case["budget_rel_diff_fraction"] is not None else None
        lines.append(
            f"| {case['case_key']} | {case['method_name']} | {case['mode']} | {case['total_serialized_bytes']} | "
            f"{case['compression_ratio_vs_full_kv']} | {case['memory_saving_fraction_vs_full_kv']} | {case['attention_output_l2_error']} | "
            f"{case['relative_l2_error']} | {case['max_abs_component_error']} | {case['cosine_similarity']} | "
            f"{case['retained_token_fraction']} | {case['retained_block_fraction'] if case['retained_block_fraction'] is not None else 'n/a'} | "
            f"{case['budget_tunable']} | {case['budget_match_attempted']} | "
            f"{case['budget_match_possible'] if case['budget_match_possible'] is not None else 'n/a'} | "
            f"{case['budget_target_bytes'] if case['budget_target_bytes'] is not None else 'n/a'} | "
            f"{case['budget_abs_diff_bytes'] if case['budget_abs_diff_bytes'] is not None else 'n/a'} | "
            f"{budget_pct if budget_pct is not None else 'n/a'} | "
            f"{case['budget_within_two_percent'] if case['budget_within_two_percent'] is not None else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## Memory Guard",
            "",
            f"- Start RSS bytes: `{results['memory_guard']['start_rss_bytes']}`",
            f"- Peak RSS bytes: `{results['memory_guard']['peak_rss_bytes']}`",
            f"- End RSS bytes: `{results['memory_guard']['end_rss_bytes']}`",
            f"- Minimum available bytes: `{results['memory_guard']['min_available_bytes']}`",
            f"- Hard RSS limit bytes: `{results['memory_guard']['max_rss_bytes']}`",
            f"- Minimum free bytes floor: `{results['memory_guard']['min_free_bytes']}`",
            "",
            "## Limitations",
            "",
            "- CPU offline runtime is not production inference latency.",
            "- KIVI-style is a source-inspired affine 2-bit approximation, not an official KIVI reproduction.",
            "- SnapKV-style uses current-query causal top-k token selection, not an official SnapKV reproduction.",
            "- Quest-style uses exact maximum query-key logit per historical block, not an official Quest reproduction.",
            "- Full KV is the reference, not a memory-matched compressed baseline.",
            "- Only the selected query heads and selected KV heads from the frozen traces are evaluated here.",
            "- Compression error remains empirical; only RACK-KV skipping relative to reconstructed compressed KV remains rigorously certified.",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_test_suite(*, output_dir: Path) -> tuple[int, str]:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
    _write_text_atomic(output_dir / DEFAULT_TEST_COMMAND, " ".join(command))
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    combined = (completed.stdout or "") + (completed.stderr or "")
    _write_text_atomic(output_dir / DEFAULT_TEST_RETURN_CODE, str(int(completed.returncode)))
    _write_text_atomic(output_dir / DEFAULT_TEST_LOG, combined)
    _ = (output_dir / DEFAULT_TEST_LOG).read_text(encoding="utf-8")
    if completed.returncode != 0:
        raise Stage4ExecutionError("Deterministic unit-test suite failed; see stage4 test log.")
    lines = [line for line in combined.splitlines() if line.strip()]
    summary_line = lines[-1] if lines else ""
    return completed.returncode, summary_line


def _full_stage4_manual_command() -> str:
    return (
        f"{sys.executable} scripts/run_stage4_baselines_smoke.py "
        f"--capture-dir .tmp/stage3_multilayer_full/capture "
        f"--output-dir .tmp/stage4_baselines_full "
        f"--results-filename {DEFAULT_FULL_RESULTS_FILENAME} "
        f"--report-filename {DEFAULT_FULL_REPORT_FILENAME} "
        f"--review-zip .tmp/stage4_baselines_full_review.zip "
        f"--run-label \"{DEFAULT_RUN_LABEL_FULL}\" "
        f"--layers 0,8,16,24,31 "
        f"--query-positions 31,47,63,79,95,111,127,143,159,175,191,207,223,239,248,249,250,251,252,253,254,255 "
        f"--query-local-indices 0,1,2 "
        f"--recent-window {STAGE4_RECENT_WINDOW} "
        f"--block-size {STAGE4_BLOCK_SIZE} "
        f"--rack-tolerance {STAGE4_RACK_TOLERANCE} "
        f"--precision {STAGE4_PRECISION} "
        f"--seed {STAGE4_SEED}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 4 offline baseline benchmark over frozen traces.")
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-filename", default=None)
    parser.add_argument("--report-filename", default=None)
    parser.add_argument("--review-zip", type=Path, default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--layers", default="0,31")
    parser.add_argument("--query-positions", default="127,255")
    parser.add_argument("--query-local-indices", default="0,1,2")
    parser.add_argument("--recent-window", type=int, default=STAGE4_RECENT_WINDOW)
    parser.add_argument("--block-size", type=int, default=STAGE4_BLOCK_SIZE)
    parser.add_argument("--rack-tolerance", type=float, default=STAGE4_RACK_TOLERANCE)
    parser.add_argument("--precision", type=int, default=STAGE4_PRECISION)
    parser.add_argument("--seed", type=int, default=STAGE4_SEED)
    parser.add_argument("--max-rss-gb", type=float, default=DEFAULT_MAX_RSS_GB)
    parser.add_argument("--min-free-memory-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_filename = args.results_filename or _default_results_filename(output_dir)
    report_filename = args.report_filename or _default_report_filename(output_dir)
    review_zip = args.review_zip or _default_review_zip(output_dir)
    run_label = args.run_label or _default_run_label(output_dir)

    layer_indices = _parse_csv_ints(args.layers, "layers")
    query_positions = _parse_csv_ints(args.query_positions, "query-positions")
    query_local_indices = _parse_csv_ints(args.query_local_indices, "query-local-indices")

    if args.run_tests:
        _run_test_suite(output_dir=output_dir)

    benchmark_results = _run_benchmark(
        capture_dir=args.capture_dir,
        output_dir=output_dir,
        run_label=run_label,
        layer_indices=layer_indices,
        query_positions=query_positions,
        query_local_indices=query_local_indices,
        recent_window=args.recent_window,
        block_size=args.block_size,
        tolerance=args.rack_tolerance,
        precision=args.precision,
        seed=args.seed,
        max_rss_bytes=int(args.max_rss_gb * (1024 ** 3)),
        min_free_bytes=int(args.min_free_memory_gb * (1024 ** 3)),
    )
    benchmark_results["dependency_versions"] = _dependency_versions()
    benchmark_results["full_stage4_manual_command"] = _full_stage4_manual_command()

    results_path = output_dir / results_filename
    report_path = output_dir / report_filename
    _write_json_atomic(results_path, benchmark_results)
    _write_text_atomic(report_path, _render_report(benchmark_results))

    outer_manifest = _build_manifest(
        file_map=_outer_manifest_file_map(
            output_dir=output_dir,
            results_filename=results_filename,
            report_filename=report_filename,
        ),
        source_relative_paths=tuple(_source_path_map().keys()),
        source_snapshot_base_dir=Path("."),
    )
    _write_json_atomic(output_dir / DEFAULT_MANIFEST_FILENAME, outer_manifest)

    review_zip_path, review_zip_sha256, inner_manifest = _create_review_zip(
        output_dir=output_dir,
        zip_path=review_zip,
        results_filename=results_filename,
        report_filename=report_filename,
    )
    print(
        json.dumps(
            {
                "results_path": str(results_path),
                "report_path": str(report_path),
                "manifest_path": str(output_dir / DEFAULT_MANIFEST_FILENAME),
                "review_zip_path": str(review_zip_path),
                "review_zip_sha256": review_zip_sha256,
                "source_snapshot_sha256": inner_manifest["source_snapshot_sha256"],
                "manifest_file_count": inner_manifest["file_count"],
                "case_count": benchmark_results["case_count"],
                "result_count": benchmark_results["actual_result_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
