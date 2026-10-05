from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import socket
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any
import zipfile

import gmpy2
import numpy as np
import psutil
import requests
import safetensors
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rack_kv.llama_trace import (  # noqa: E402
    DEFAULT_LLAMA31_BASE_REPO,
    ensure_minimal_llama31_assets,
    run_minimal_llama31_capture,
)
from rack_kv.stage2 import result_to_dict as stage2a_result_to_dict  # noqa: E402
from rack_kv.stage2 import run_stage2_smoke_experiment  # noqa: E402
from rack_kv.stage2 import (  # noqa: E402
    _build_prefix_result,
    _build_query_case_base,
    _canonical_attention_scale,
    _run_query_case,
    _run_query_case_tolerance_bundle,
    validate_compact_trace,
)
from rack_kv.stage2b import result_to_dict as stage2b_result_to_dict  # noqa: E402
from rack_kv.stage2b import run_stage2b_pilot, select_stage2b_query_positions  # noqa: E402
from rack_kv.stage2b_prompt import (  # noqa: E402
    PINNED_LLAMA31_REVISION,
    Stage2BPromptArtifacts,
    STAGE2B_PROMPT_SEED,
    STAGE2B_TARGET_TOKEN_COUNT,
    generate_stage2b_prompt_artifacts,
    write_stage2b_prompt_artifacts,
)


DEFAULT_STAGE2A_TRACE_PATH = Path(".tmp/llama31_capture/llama31_layer0_trace.safetensors")
DEFAULT_STAGE2A_RESULTS_PATH = Path(".tmp/stage2_layer0_smoke/stage2_smoke_results.json")
DEFAULT_OUTPUT_DIR = Path(".tmp/stage2b_layer0_pilot")
DEFAULT_CAPTURE_DIRNAME = "capture"
DEFAULT_PROMPT_DIRNAME = "prompt"
DEFAULT_RESULTS_FILENAME = "stage2b_pilot_results.json"
DEFAULT_REPORT_FILENAME = "stage2b_pilot_report.md"
DEFAULT_DEPENDENCY_FILENAME = "stage2b_dependency_versions.json"
DEFAULT_CAPTURE_COMMAND_FILENAME = "capture_command.txt"
DEFAULT_EXPERIMENT_COMMAND_FILENAME = "experiment_command.txt"
DEFAULT_TEST_COMMAND_FILENAME = "test_command.txt"
DEFAULT_TEST_RETURN_CODE_FILENAME = "test_return_code.txt"
DEFAULT_TEST_LOG_FILENAME = "stage2b_layer0_pilot_test_log.txt"
DEFAULT_MANIFEST_FILENAME = "review_package_manifest.json"
DEFAULT_REVIEW_ZIP = "rack_kv_stage2b_layer0_pilot_review.zip"
DEFAULT_CAPTURE_TRACE_FILENAME = "llama31_layer0_trace.safetensors"
DEFAULT_CAPTURE_REPORT_FILENAME = "capture_report.json"
DEFAULT_CAPTURE_DEPENDENCY_FILENAME = "dependency_versions.json"
DEFAULT_STAGE2A_RESULTS_COPY = "accepted_stage2a_results.json"
DEFAULT_PAIR_CHECKPOINT_DIRNAME = "pair_checkpoints"
DEFAULT_PAIR_CHECKPOINT_SCHEMA = "stage2b_pair_checkpoint_v2"
DEFAULT_PAIR_CODE_VERSION = "streaming_case_checkpoint_v1"
DEFAULT_RESUME_LOCK_FILENAME = "stage2b_resumable.lock.json"
SHARED_STAGE1_ASSET_ROOT = Path(".tmp/llama31_capture")
DEFAULT_PRE_CASE_RSS_MARGIN_BYTES = 256 * 1024 * 1024
DEFAULT_PRE_CASE_AVAILABLE_MARGIN_BYTES = 256 * 1024 * 1024
REVIEW_SOURCE_FILES = (
    "rack_kv/__init__.py",
    "rack_kv/accounting.py",
    "rack_kv/codec.py",
    "rack_kv/certificate.py",
    "rack_kv/ieee.py",
    "rack_kv/llama_trace.py",
    "rack_kv/rigorous.py",
    "rack_kv/stage2.py",
    "rack_kv/stage2b.py",
    "rack_kv/stage2b_prompt.py",
    "rack_kv/types.py",
    "scripts/run_minimal_llama31_capture.py",
    "scripts/run_stage2b_layer0_pilot.py",
    "tests/test_codec_and_accounting.py",
    "tests/test_llama_trace.py",
    "tests/test_rigorous_certificate.py",
    "tests/test_stage2_integration.py",
    "tests/test_stage2b.py",
    "tests/test_stage2b_prompt.py",
)


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(piece.strip()) for piece in raw.split(",") if piece.strip())


def _dependency_versions() -> dict[str, str]:
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _write_json(path: Path, payload: dict) -> Path:
    return _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True))


def _rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def _available_memory_bytes() -> int:
    return int(psutil.virtual_memory().available)


def _gb_to_bytes(value_gb: float) -> int:
    return int(value_gb * (1024 ** 3))


def _best_effort_trim_memory() -> None:
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass


class Stage2BExecutionError(RuntimeError):
    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class MemoryGuardExceeded(Stage2BExecutionError):
    pass


class StreamingMemoryGuard:
    def __init__(
        self,
        *,
        max_rss_bytes: int,
        min_free_memory_bytes: int,
        pre_case_rss_margin_bytes: int = DEFAULT_PRE_CASE_RSS_MARGIN_BYTES,
        pre_case_available_margin_bytes: int = DEFAULT_PRE_CASE_AVAILABLE_MARGIN_BYTES,
    ) -> None:
        self.max_rss_bytes = int(max_rss_bytes)
        self.min_free_memory_bytes = int(min_free_memory_bytes)
        self.pre_case_rss_margin_bytes = max(0, int(pre_case_rss_margin_bytes))
        self.pre_case_available_margin_bytes = max(0, int(pre_case_available_margin_bytes))
        self.starting_rss_bytes = _rss_bytes()
        self.peak_rss_bytes = self.starting_rss_bytes
        self.ending_rss_bytes = self.starting_rss_bytes
        self.minimum_available_memory_bytes = _available_memory_bytes()

    def record(self) -> tuple[int, int]:
        rss = _rss_bytes()
        available = _available_memory_bytes()
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self.minimum_available_memory_bytes = min(self.minimum_available_memory_bytes, available)
        self.ending_rss_bytes = rss
        return rss, available

    def limit_message(self, stage: str, *, rss: int, available: int, before_case: bool = False) -> str | None:
        if before_case:
            start_case_limit = max(0, self.max_rss_bytes - self.pre_case_rss_margin_bytes)
            if rss >= start_case_limit:
                return (
                    f"Refusing to start a new Stage 2B case at '{stage}': RSS {rss} bytes is too close to the "
                    f"hard limit {self.max_rss_bytes} bytes."
                )
            start_case_floor = self.min_free_memory_bytes + self.pre_case_available_margin_bytes
            if available <= start_case_floor:
                return (
                    f"Refusing to start a new Stage 2B case at '{stage}': available physical memory {available} bytes "
                    f"is too close to the floor {self.min_free_memory_bytes} bytes."
                )
        if rss > self.max_rss_bytes:
            return (
                f"Stage 2B memory guard triggered at '{stage}': RSS {rss} bytes exceeded the hard limit "
                f"{self.max_rss_bytes} bytes."
            )
        if available < self.min_free_memory_bytes:
            return (
                f"Stage 2B memory guard triggered at '{stage}': available physical memory {available} bytes fell "
                f"below the floor {self.min_free_memory_bytes} bytes."
            )
        return None

    def check(self, stage: str, *, before_case: bool = False) -> tuple[int, int]:
        rss, available = self.record()
        message = self.limit_message(stage, rss=rss, available=available, before_case=before_case)
        if message is not None:
            raise MemoryGuardExceeded(message, exit_code=17)
        return rss, available


def _try_git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except Exception:
        return None
    revision = result.stdout.strip()
    return revision or None


def _source_snapshot_sha256(source_files: tuple[str, ...]) -> str:
    entries = []
    for relative_path in sorted(source_files):
        path = Path(relative_path)
        entries.append(f"{relative_path}\t{_sha256_file(path)}")
    payload = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _measure_call(fn, /, *args, **kwargs):
    process = psutil.Process()
    peak_rss = {"value": process.memory_info().rss}
    stop = threading.Event()

    def _poll() -> None:
        while not stop.is_set():
            try:
                peak_rss["value"] = max(peak_rss["value"], process.memory_info().rss)
            except Exception:
                pass
            stop.wait(0.05)

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        thread.join(timeout=1.0)
        try:
            peak_rss["value"] = max(peak_rss["value"], process.memory_info().rss)
        except Exception:
            pass
    return result, elapsed, int(peak_rss["value"])


def _measure_call_with_memory(fn, /, *args, **kwargs):
    process = psutil.Process()
    start_rss = int(process.memory_info().rss)
    peak_rss = {"value": start_rss}
    min_available = {"value": int(psutil.virtual_memory().available)}
    stop = threading.Event()

    def _poll() -> None:
        while not stop.is_set():
            try:
                peak_rss["value"] = max(peak_rss["value"], int(process.memory_info().rss))
                min_available["value"] = min(min_available["value"], int(psutil.virtual_memory().available))
            except Exception:
                pass
            stop.wait(0.05)

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        thread.join(timeout=1.0)
        try:
            peak_rss["value"] = max(peak_rss["value"], int(process.memory_info().rss))
            min_available["value"] = min(min_available["value"], int(psutil.virtual_memory().available))
        except Exception:
            pass
    return result, elapsed, start_rss, int(peak_rss["value"]), int(min_available["value"])


def _run_tests_capture(
    *,
    test_command_path: Path,
    test_return_code_path: Path,
    test_log_path: Path,
) -> dict[str, str | int]:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
    command_text = subprocess.list2cmdline(command)
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    test_command_path.write_text(command_text, encoding="utf-8")
    test_return_code_path.write_text(str(result.returncode), encoding="utf-8")
    test_log_path.write_text(result.stdout, encoding="utf-8")
    reopened = test_log_path.read_text(encoding="utf-8")
    if reopened != result.stdout:
        raise RuntimeError("UTF-8 test log verification failed after writing the captured unittest output.")
    if result.returncode != 0:
        raise RuntimeError(f"Full test suite failed with return code {result.returncode}.")
    return {
        "test_command": command_text,
        "test_return_code": result.returncode,
        "test_log_path": str(test_log_path),
    }


def _copy_stage1_assets_if_present(*, output_dir: Path, repo_revision: str) -> Path | None:
    source_dir = SHARED_STAGE1_ASSET_ROOT / "llama31_base_assets" / repo_revision
    destination_dir = output_dir / "llama31_base_assets" / repo_revision
    if not source_dir.exists():
        return None
    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file():
            target = destination_dir / path.name
            if not target.exists():
                shutil.copy2(path, target)
    return destination_dir


def _ensure_assets_with_tls_fallback(
    *,
    output_dir: Path,
    repo_id: str,
    repo_revision: str,
) -> tuple[Path, dict]:
    try:
        asset_result = ensure_minimal_llama31_assets(
            output_dir=output_dir,
            repo_id=repo_id,
            repo_revision=repo_revision,
            allow_insecure_tls=False,
        )
        return asset_result.asset_dir, {
            "tls_verification": True,
            "repo_revision_source": asset_result.repo_revision_source,
            "asset_sources": asset_result.asset_sources,
            "asset_hashes": asset_result.asset_hashes,
            "fallback_reason": None,
        }
    except requests.exceptions.RequestException as error:
        asset_result = ensure_minimal_llama31_assets(
            output_dir=output_dir,
            repo_id=repo_id,
            repo_revision=repo_revision,
            allow_insecure_tls=True,
        )
        return asset_result.asset_dir, {
            "tls_verification": False,
            "repo_revision_source": asset_result.repo_revision_source,
            "asset_sources": asset_result.asset_sources,
            "asset_hashes": asset_result.asset_hashes,
            "fallback_reason": str(error),
        }


def _capture_with_tls_fallback(
    *,
    capture_dir: Path,
    repo_id: str,
    repo_revision: str,
    prompt_text: str,
    prompt_token_ids: tuple[int, ...],
    selected_query_heads: tuple[int, ...],
) -> tuple[object, float, int, dict]:
    kwargs = dict(
        output_dir=capture_dir,
        repo_id=repo_id,
        repo_revision=repo_revision,
        prompt=prompt_text,
        prompt_token_ids=prompt_token_ids,
        layer_index=0,
        selected_query_heads=selected_query_heads,
    )
    try:
        result, elapsed_s, peak_rss = _measure_call(
            run_minimal_llama31_capture,
            allow_insecure_tls=False,
            **kwargs,
        )
        return result, elapsed_s, peak_rss, {"tls_verification": True, "fallback_reason": None}
    except requests.exceptions.RequestException as error:
        result, elapsed_s, peak_rss = _measure_call(
            run_minimal_llama31_capture,
            allow_insecure_tls=True,
            **kwargs,
        )
        return result, elapsed_s, peak_rss, {"tls_verification": False, "fallback_reason": str(error)}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_existing_prompt_artifacts(metadata_path: Path) -> Stage2BPromptArtifacts:
    payload = _load_json(metadata_path)
    return Stage2BPromptArtifacts(
        seed=int(payload["seed"]),
        target_token_count=int(payload["target_token_count"]),
        source_text=str(payload["source_text"]),
        source_sha256=str(payload["source_sha256"]),
        token_ids=tuple(int(value) for value in payload["token_ids"]),
        decoded_text=str(payload["decoded_text"]),
    )


def _reuse_existing_capture(
    *,
    trace_path: Path,
    capture_report_path: Path,
    dependency_report_path: Path,
) -> object:
    capture_report = _load_json(capture_report_path)
    trace_sha256 = _sha256_file(trace_path)
    if capture_report.get("trace_sha256") != trace_sha256:
        raise RuntimeError("Existing capture trace SHA-256 does not match the saved capture report.")
    checkpoint = SimpleNamespace(
        repo_id=str(capture_report["checkpoint"]["repo_id"]),
        repo_revision=str(capture_report["checkpoint"]["revision"]),
    )
    return SimpleNamespace(
        checkpoint=checkpoint,
        trace_path=trace_path,
        capture_report_path=capture_report_path,
        dependency_report_path=dependency_report_path,
        max_roundtrip_abs_diff=float(capture_report["roundtrip"]["max_abs_diff"]),
        max_compact_replay_abs_diff=float(capture_report["compact_replay"]["max_abs_diff"]),
        stock_projected_output_max_abs_diff=float(capture_report["stock_forward_comparison"]["projected_output_max_abs_diff"]),
        stock_projected_output_max_rel_diff=float(capture_report["stock_forward_comparison"]["projected_output_max_rel_diff"]),
        stock_decoder_output_max_abs_diff=float(capture_report["stock_forward_comparison"]["decoder_output_max_abs_diff"]),
        stock_decoder_output_max_rel_diff=float(capture_report["stock_forward_comparison"]["decoder_output_max_rel_diff"]),
        stock_cache_key_max_abs_diff=float(capture_report["cache_comparison"]["key_max_abs_diff"]),
        stock_cache_key_max_rel_diff=float(capture_report["cache_comparison"]["key_max_rel_diff"]),
        stock_cache_value_max_abs_diff=float(capture_report["cache_comparison"]["value_max_abs_diff"]),
        stock_cache_value_max_rel_diff=float(capture_report["cache_comparison"]["value_max_rel_diff"]),
        peak_rss_bytes=int(capture_report["peak_rss_bytes"]),
    )


def _pair_checkpoint_path(*, output_dir: Path, recent_window: int, block_size: int) -> Path:
    return output_dir / DEFAULT_PAIR_CHECKPOINT_DIRNAME / f"pair_w{recent_window}_b{block_size}.json"


def _projection_case_fields(case: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "record_index",
        "query_local_index",
        "query_head_global",
        "kv_head_global",
        "visible_length",
        "query_position",
        "historical_tokens",
        "recent_exact_tokens",
        "candidate_blocks",
        "certified_skipped_blocks",
        "decoded_blocks",
        "decoded_block_starts",
        "skipped_block_starts",
        "certificate_name",
        "z_k_lower_text",
        "z_k_lower_upper_float",
        "u_s_upper_text",
        "u_s_upper_float",
        "nu_s_upper_text",
        "nu_s_upper_float",
        "kept_output_norm_upper_text",
        "kept_output_norm_upper_float",
        "certificate_bound_text",
        "certificate_bound_upper_float",
        "observed_skip_error",
        "rigorous_skip_error_upper_text",
        "rigorous_skip_error_upper_float",
        "bound_to_observed_ratio",
        "approximate_observed_violation",
        "rigorous_interval_violation",
        "numerical_fallback_used",
        "model_reference_gap",
        "compression_error",
        "reference_total_error",
        "captured_model_total_gap",
        "reference_decomposition_lhs",
        "reference_decomposition_rhs",
        "model_relative_decomposition_lhs",
        "model_relative_decomposition_rhs",
    )
    return {field: case[field] for field in fields}


def _projection_config_fields(config: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "recent_window",
        "block_size",
        "tolerance",
        "evaluated_query_cases",
        "eligible_query_cases",
        "ineligible_query_cases",
        "total_candidate_blocks",
        "certified_skipped_blocks",
        "decoded_blocks",
        "weighted_skipped_block_fraction",
        "weighted_decoded_block_fraction",
        "mean_case_skipped_fraction",
        "median_case_skipped_fraction",
        "max_case_skipped_fraction",
        "fraction_cases_with_any_skipped_block",
        "max_certified_bound",
        "max_observed_reconstructed_skipping_error",
        "max_rigorous_skip_error_upper",
        "min_bound_to_observed_ratio_nonzero",
        "max_bound_to_observed_ratio_nonzero",
        "rigorous_interval_violation_count",
        "approximate_observed_violation_count",
        "false_safe_violation_count",
        "fallback_count",
        "max_model_reference_gap",
        "max_true_compression_error",
        "max_reference_total_error",
        "max_captured_model_total_gap",
        "reference_decomposition_violation_count",
        "model_relative_decomposition_violation_count",
    )
    projection = {field: config[field] for field in fields}
    projection["query_case_results"] = [_projection_case_fields(case) for case in config["query_case_results"]]
    return projection


def _projection_from_stage2b_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    return [_projection_config_fields(config) for config in results["configurations"]]


def _tolerance_key(value: float) -> str:
    return format(float(value), ".17g")


def _normalize_query_positions(trace: Any, query_positions: tuple[int, ...] | None) -> tuple[int, ...]:
    if query_positions is None:
        positions = select_stage2b_query_positions(trace.sequence_length)
    else:
        positions = tuple(sorted({int(position) for position in query_positions}))
    if not positions:
        raise ValueError("No Stage 2B query positions were selected.")
    if any(position < 0 or position >= trace.sequence_length for position in positions):
        raise ValueError("Stage 2B query positions must stay within the trace sequence length.")
    return positions


def _pair_case_key(
    *,
    recent_window: int,
    block_size: int,
    record_index: int,
    query_position: int,
    query_local_index: int,
    query_head_global: int,
) -> str:
    return (
        f"w{recent_window}_b{block_size}_r{record_index}_pos{query_position}"
        f"_ql{query_local_index}_qh{query_head_global}"
    )


def _pair_prefix_key(*, record_index: int, kv_head_global: int) -> str:
    return f"r{record_index}_kv{kv_head_global}"


def _pair_case_order(
    *,
    trace: Any,
    recent_window: int,
    block_size: int,
    query_positions: tuple[int, ...],
) -> tuple[dict[str, Any], ...]:
    ordered: list[dict[str, Any]] = []
    for record_index in query_positions:
        query_position = int(trace.query_positions[record_index])
        for query_local_index in range(trace.query_head_count):
            query_head_global = int(trace.selected_query_heads[query_local_index])
            kv_head_global = int(trace.query_to_kv_heads[query_local_index])
            ordered.append(
                {
                    "case_key": _pair_case_key(
                        recent_window=recent_window,
                        block_size=block_size,
                        record_index=record_index,
                        query_position=query_position,
                        query_local_index=query_local_index,
                        query_head_global=query_head_global,
                    ),
                    "record_index": int(record_index),
                    "query_position": query_position,
                    "query_local_index": int(query_local_index),
                    "query_head_global": query_head_global,
                    "kv_head_global": kv_head_global,
                    "prefix_key": _pair_prefix_key(record_index=record_index, kv_head_global=kv_head_global),
                }
            )
    return tuple(ordered)


def _serialize_prefix_record(prefix_result: Any) -> dict[str, Any]:
    key_diff = np.asarray(prefix_result.reconstructed_keys - prefix_result.original_keys, dtype=np.float64)
    value_diff = np.asarray(prefix_result.reconstructed_values - prefix_result.original_values, dtype=np.float64)
    return {
        "record_index": int(prefix_result.record_index),
        "kv_head_global": int(prefix_result.kv_head_global),
        "visible_length": int(prefix_result.visible_length),
        "recent_window": int(prefix_result.recent_window),
        "block_size": int(prefix_result.block_size),
        "historical_tokens": int(prefix_result.historical_tokens),
        "recent_exact_tokens": int(prefix_result.recent_exact_tokens),
        "original_kv_bytes": int(prefix_result.original_kv_bytes),
        "recent_exact_bytes": int(prefix_result.recent_exact_bytes),
        "compressed_historical_bytes": int(prefix_result.compressed_historical_bytes),
        "total_compressed_bytes": int(prefix_result.total_compressed_bytes),
        "bytes_per_historical_token": float(prefix_result.bytes_per_historical_token),
        "compression_ratio": float(prefix_result.compression_ratio),
        "key_anchor_bytes": int(prefix_result.key_anchor_bytes),
        "value_anchor_bytes": int(prefix_result.value_anchor_bytes),
        "quantized_key_residual_bytes": int(prefix_result.quantized_key_residual_bytes),
        "quantized_value_residual_bytes": int(prefix_result.quantized_value_residual_bytes),
        "scale_bytes": int(prefix_result.scale_bytes),
        "certificate_metadata_bytes": int(prefix_result.certificate_metadata_bytes),
        "block_header_bytes": int(prefix_result.block_header_bytes),
        "container_header_bytes": int(prefix_result.container_header_bytes),
        "container_index_bytes": int(prefix_result.container_index_bytes),
        "padding_alignment_bytes": int(prefix_result.padding_alignment_bytes),
        "independent_blocks_tested": int(prefix_result.independent_blocks_tested),
        "independent_decode_max_diff": float(prefix_result.independent_decode_max_diff),
        "unaffected_block_decode_tested": bool(prefix_result.unaffected_block_decode_tested),
        "unaffected_block_decode_passed": bool(prefix_result.unaffected_block_decode_passed),
        "max_key_abs_error": float(prefix_result.max_key_abs_error),
        "key_rmse": float(prefix_result.key_rmse),
        "key_squared_error_sum": float(np.square(key_diff, dtype=np.float64).sum(dtype=np.float64)),
        "key_element_count": int(key_diff.size),
        "max_value_abs_error": float(prefix_result.max_value_abs_error),
        "value_rmse": float(prefix_result.value_rmse),
        "value_squared_error_sum": float(np.square(value_diff, dtype=np.float64).sum(dtype=np.float64)),
        "value_element_count": int(value_diff.size),
        "number_of_blocks": int(len(prefix_result.blocks)),
        "authoritative_serialized_roundtrip_used": bool(prefix_result.authoritative_serialized_roundtrip_used),
    }


def _pair_checkpoint_settings(
    *,
    trace: Any,
    prompt_source_sha256: str,
    recent_window: int,
    block_size: int,
    tolerances: tuple[float, ...],
    precision: int,
    random_seed: int,
    query_positions: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "schema": DEFAULT_PAIR_CHECKPOINT_SCHEMA,
        "code_version": DEFAULT_PAIR_CODE_VERSION,
        "trace_path": str(trace.trace_path),
        "trace_sha256": trace.trace_sha256,
        "trace_schema": trace.trace_schema,
        "trace_scaling": float(trace.scaling),
        "prompt_source_sha256": str(prompt_source_sha256),
        "recent_window": int(recent_window),
        "block_size": int(block_size),
        "tolerances": [float(value) for value in tolerances],
        "precision": int(precision),
        "random_seed": int(random_seed),
        "query_positions": [int(position) for position in query_positions],
        "selected_query_heads": [int(value) for value in trace.selected_query_heads],
        "selected_kv_heads": [int(value) for value in trace.selected_kv_heads],
        "query_to_kv_heads": [int(value) for value in trace.query_to_kv_heads],
        "trace_sequence_length": int(trace.sequence_length),
        "trace_query_shape": [int(dim) for dim in trace.queries.shape],
        "trace_final_key_shape": [int(dim) for dim in trace.final_keys.shape],
        "trace_final_value_shape": [int(dim) for dim in trace.final_values.shape],
        "trace_model_head_output_shape": [int(dim) for dim in trace.model_head_outputs.shape],
    }


def _pair_checkpoint_matches_settings(payload: dict[str, Any], settings: dict[str, Any]) -> tuple[bool, str | None]:
    for key in (
        "schema",
        "code_version",
        "trace_sha256",
        "trace_schema",
        "prompt_source_sha256",
        "recent_window",
        "block_size",
        "precision",
        "random_seed",
        "trace_sequence_length",
    ):
        if payload.get(key) != settings.get(key):
            return False, key
    for key in (
        "tolerances",
        "query_positions",
        "selected_query_heads",
        "selected_kv_heads",
        "query_to_kv_heads",
        "trace_query_shape",
        "trace_final_key_shape",
        "trace_final_value_shape",
        "trace_model_head_output_shape",
    ):
        if list(payload.get(key, [])) != list(settings.get(key, [])):
            return False, key
    return True, None


def _new_pair_checkpoint_payload(*, settings: dict[str, Any], case_order: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    payload = dict(settings)
    payload.update(
        {
            "status": "in_progress",
            "case_order": list(case_order),
            "prefix_records": {},
            "case_records": {},
            "equivalence": {
                "exact_match": True,
                "first_difference": None,
            },
            "metrics": {
                "cases_total": len(case_order),
                "cases_completed": 0,
                "cases_reused_from_checkpoint": 0,
                "cases_newly_computed": 0,
            },
        }
    )
    return payload


def _completed_case_keys(payload: dict[str, Any], tolerances: tuple[float, ...]) -> set[str]:
    tolerance_keys = {_tolerance_key(value) for value in tolerances}
    completed: set[str] = set()
    for case_key, case_record in payload.get("case_records", {}).items():
        if case_record.get("status") != "completed":
            continue
        if set(case_record.get("results_by_tolerance", {}).keys()) != tolerance_keys:
            raise Stage2BExecutionError(
                f"Checkpoint case '{case_key}' does not contain the expected tolerance set.",
                exit_code=2,
            )
        if not bool(case_record.get("equivalence", {}).get("exact_match")):
            raise Stage2BExecutionError(
                f"Checkpoint case '{case_key}' recorded a bundled/reference mismatch and will not be reused.",
                exit_code=2,
            )
        completed.add(case_key)
    return completed


def _load_pair_checkpoint(
    *,
    checkpoint_path: Path,
    settings: dict[str, Any],
    tolerances: tuple[float, ...],
) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None
    payload = _load_json(checkpoint_path)
    matches, mismatch_key = _pair_checkpoint_matches_settings(payload, settings)
    if not matches:
        raise Stage2BExecutionError(
            f"Existing pair checkpoint '{checkpoint_path}' does not match the current settings at '{mismatch_key}'.",
            exit_code=2,
        )
    if payload.get("status") not in {"in_progress", "completed", "stopped"}:
        raise Stage2BExecutionError(
            f"Checkpoint '{checkpoint_path}' has unsupported status {payload.get('status')!r}.",
            exit_code=2,
        )
    _completed_case_keys(payload, tolerances)
    return payload


def _independent_pair_projection(
    *,
    trace_path: Path,
    recent_window: int,
    block_size: int,
    tolerances: tuple[float, ...],
    precision: int,
) -> list[dict[str, Any]]:
    trace = validate_compact_trace(trace_path)
    selected_positions = select_stage2b_query_positions(trace.sequence_length)
    prefix_cache: dict[tuple[int, int], Any] = {}
    for record_index in selected_positions:
        for kv_head_global in trace.selected_kv_heads:
            prefix_keys, prefix_values = trace.kv_prefix(record_index=record_index, kv_head_global=kv_head_global)
            prefix_cache[(record_index, kv_head_global)] = _build_prefix_result(
                record_index=record_index,
                kv_head_global=kv_head_global,
                prefix_keys=prefix_keys,
                prefix_values=prefix_values,
                recent_window=recent_window,
                block_size=block_size,
                precision=precision,
            )
    case_bases = {
        (record_index, query_local_index): _build_query_case_base(
            trace=trace,
            prefix_result=prefix_cache[(record_index, trace.query_to_kv_heads[query_local_index])],
            query_local_index=query_local_index,
            precision=precision,
        )
        for record_index in selected_positions
        for query_local_index in range(trace.query_head_count)
    }

    projections: list[dict[str, Any]] = []
    for tolerance in tolerances:
        case_results = []
        for record_index in selected_positions:
            for query_local_index in range(trace.query_head_count):
                kv_head_global = trace.query_to_kv_heads[query_local_index]
                prefix_result = prefix_cache[(record_index, kv_head_global)]
                case_results.append(
                    _run_query_case(
                        trace=trace,
                        prefix_result=prefix_result,
                        query_local_index=query_local_index,
                        tolerance=tolerance,
                        precision=precision,
                        base_case=case_bases[(record_index, query_local_index)],
                    )
                )

        candidate_blocks = sum(case.candidate_blocks for case in case_results)
        certified_skipped_blocks = sum(case.certified_skipped_blocks for case in case_results)
        decoded_blocks = sum(case.decoded_blocks for case in case_results)
        eligible_cases = [case for case in case_results if case.candidate_blocks > 0]
        skipped_case_fractions = [
            float(case.certified_skipped_blocks / case.candidate_blocks)
            for case in eligible_cases
            if case.candidate_blocks > 0
        ]
        nonzero_ratios = [case.bound_to_observed_ratio for case in case_results if case.bound_to_observed_ratio is not None]
        projections.append(
            {
                "recent_window": recent_window,
                "block_size": block_size,
                "tolerance": float(tolerance),
                "evaluated_query_cases": len(case_results),
                "eligible_query_cases": len(eligible_cases),
                "ineligible_query_cases": len(case_results) - len(eligible_cases),
                "total_candidate_blocks": candidate_blocks,
                "certified_skipped_blocks": certified_skipped_blocks,
                "decoded_blocks": decoded_blocks,
                "weighted_skipped_block_fraction": float(certified_skipped_blocks / candidate_blocks) if candidate_blocks > 0 else 0.0,
                "weighted_decoded_block_fraction": float(decoded_blocks / candidate_blocks) if candidate_blocks > 0 else 0.0,
                "mean_case_skipped_fraction": float(np.mean(skipped_case_fractions)) if skipped_case_fractions else 0.0,
                "median_case_skipped_fraction": float(np.median(skipped_case_fractions)) if skipped_case_fractions else 0.0,
                "max_case_skipped_fraction": max(skipped_case_fractions, default=0.0),
                "fraction_cases_with_any_skipped_block": (
                    float(sum(1 for case in eligible_cases if case.certified_skipped_blocks > 0) / len(eligible_cases))
                    if eligible_cases
                    else 0.0
                ),
                "max_certified_bound": max((case.certificate_bound_upper_float for case in case_results), default=0.0),
                "max_observed_reconstructed_skipping_error": max((case.observed_skip_error for case in case_results), default=0.0),
                "max_rigorous_skip_error_upper": max((case.rigorous_skip_error_upper_float for case in case_results), default=0.0),
                "min_bound_to_observed_ratio_nonzero": min(nonzero_ratios) if nonzero_ratios else None,
                "max_bound_to_observed_ratio_nonzero": max(nonzero_ratios) if nonzero_ratios else None,
                "rigorous_interval_violation_count": sum(1 for case in case_results if case.rigorous_interval_violation),
                "approximate_observed_violation_count": sum(1 for case in case_results if case.approximate_observed_violation),
                "false_safe_violation_count": sum(1 for case in case_results if case.rigorous_interval_violation),
                "fallback_count": sum(1 for case in case_results if case.numerical_fallback_used),
                "max_model_reference_gap": max((case.model_reference_gap for case in case_results), default=0.0),
                "max_true_compression_error": max((case.compression_error for case in case_results), default=0.0),
                "max_reference_total_error": max((case.reference_total_error for case in case_results), default=0.0),
                "max_captured_model_total_gap": max((case.captured_model_total_gap for case in case_results), default=0.0),
                "reference_decomposition_violation_count": sum(
                    1
                    for case in case_results
                    if case.reference_decomposition_lhs > case.reference_decomposition_rhs + 1e-9
                ),
                "model_relative_decomposition_violation_count": sum(
                    1
                    for case in case_results
                    if case.model_relative_decomposition_lhs > case.model_relative_decomposition_rhs + 1e-9
                ),
                "query_case_results": [_projection_case_fields(case.__dict__) for case in case_results],
            }
        )
    return projections


def _process_create_time(pid: int) -> float | None:
    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.Error, OSError, ValueError):
        return None


def _current_lock_payload() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "process_create_time": _process_create_time(os.getpid()),
        "command": subprocess.list2cmdline(sys.argv),
        "hostname": socket.gethostname(),
        "start_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _lock_payload_is_live(payload: dict[str, Any]) -> bool:
    pid = int(payload["pid"])
    recorded_create_time = float(payload["process_create_time"])
    actual_create_time = _process_create_time(pid)
    if actual_create_time is None:
        return False
    return abs(actual_create_time - recorded_create_time) <= 1e-6


def _acquire_resume_lock(lock_path: Path) -> dict[str, Any]:
    current_payload = _current_lock_payload()
    if lock_path.exists():
        try:
            existing_payload = _load_json(lock_path)
            existing_pid = int(existing_payload["pid"])
            existing_create_time = float(existing_payload["process_create_time"])
        except Exception as error:
            raise Stage2BExecutionError(
                f"Existing Stage 2B lock '{lock_path}' is malformed and will not be removed automatically: {error}",
                exit_code=2,
            ) from error
        if existing_pid == os.getpid() and abs(existing_create_time - float(current_payload["process_create_time"])) <= 1e-6:
            return current_payload
        if _lock_payload_is_live(existing_payload):
            raise Stage2BExecutionError(
                f"Another live Stage 2B worker holds the lock (pid {existing_pid}).",
                exit_code=2,
            )
        lock_path.unlink()
    _write_json(lock_path, current_payload)
    return current_payload


def _release_resume_lock(lock_path: Path, lock_payload: dict[str, Any] | None) -> None:
    if lock_payload is None or not lock_path.exists():
        return
    try:
        existing_payload = _load_json(lock_path)
    except Exception:
        return
    if (
        int(existing_payload.get("pid", -1)) == int(lock_payload["pid"])
        and float(existing_payload.get("process_create_time", -1.0)) == float(lock_payload["process_create_time"])
    ):
        lock_path.unlink()


def _run_pair_checkpoint_internal(
    *,
    trace_path: Path,
    checkpoint_path: Path,
    prompt_source_sha256: str,
    recent_window: int,
    block_size: int,
    tolerances: tuple[float, ...],
    precision: int,
    random_seed: int,
    max_rss_bytes: int,
    min_free_memory_bytes: int,
    query_positions: tuple[int, ...] | None = None,
    stop_after_completed_cases: int | None = None,
) -> dict[str, Any]:
    trace = validate_compact_trace(trace_path)
    normalized_positions = _normalize_query_positions(trace, query_positions)
    case_order = _pair_case_order(
        trace=trace,
        recent_window=recent_window,
        block_size=block_size,
        query_positions=normalized_positions,
    )
    settings = _pair_checkpoint_settings(
        trace=trace,
        prompt_source_sha256=prompt_source_sha256,
        recent_window=recent_window,
        block_size=block_size,
        tolerances=tolerances,
        precision=precision,
        random_seed=random_seed,
        query_positions=normalized_positions,
    )
    payload = _load_pair_checkpoint(
        checkpoint_path=checkpoint_path,
        settings=settings,
        tolerances=tolerances,
    )
    if payload is None:
        payload = _new_pair_checkpoint_payload(settings=settings, case_order=case_order)
    elif tuple(payload.get("case_order", ())) != tuple(case_order):
        raise Stage2BExecutionError(
            f"Checkpoint '{checkpoint_path}' has a different case order than the current Stage 2B run.",
            exit_code=2,
        )

    memory_guard = StreamingMemoryGuard(
        max_rss_bytes=max_rss_bytes,
        min_free_memory_bytes=min_free_memory_bytes,
    )
    pair_start = time.perf_counter()
    memory_guard.record()
    completed_case_keys = _completed_case_keys(payload, tolerances)
    cases_reused_from_checkpoint = len(completed_case_keys)
    cases_newly_computed = 0

    try:
        for case_entry in case_order:
            case_key = case_entry["case_key"]
            if case_key in completed_case_keys:
                continue
            memory_guard.check("before_case", before_case=True)
            case_record, prefix_record, case_metrics = _run_case_with_equivalence(
                trace=trace,
                record_index=int(case_entry["record_index"]),
                query_local_index=int(case_entry["query_local_index"]),
                recent_window=recent_window,
                block_size=block_size,
                tolerances=tolerances,
                precision=precision,
                memory_guard=memory_guard,
            )
            payload["prefix_records"].setdefault(case_entry["prefix_key"], prefix_record)
            payload["case_records"][case_key] = case_record
            payload["equivalence"]["exact_match"] = bool(payload["equivalence"]["exact_match"]) and bool(case_record["equivalence"]["exact_match"])
            if not case_record["equivalence"]["exact_match"] and payload["equivalence"]["first_difference"] is None:
                payload["equivalence"]["first_difference"] = {
                    "case_key": case_key,
                    **case_record["equivalence"]["first_difference"],
                }
            completed_case_keys.add(case_key)
            cases_newly_computed += 1
            payload["status"] = "in_progress"
            payload["metrics"].update(
                {
                    "cases_total": len(case_order),
                    "cases_completed": len(completed_case_keys),
                    "cases_reused_from_checkpoint": cases_reused_from_checkpoint,
                    "cases_newly_computed": cases_newly_computed,
                    "last_completed_case_key": case_key,
                    "starting_rss_bytes": memory_guard.starting_rss_bytes,
                    "peak_rss_bytes": memory_guard.peak_rss_bytes,
                    "minimum_available_memory_bytes": memory_guard.minimum_available_memory_bytes,
                    "rss_limit_bytes": max_rss_bytes,
                    "minimum_free_memory_limit_bytes": min_free_memory_bytes,
                }
            )
            _write_json(checkpoint_path, payload)

            if not case_record["equivalence"]["exact_match"]:
                raise Stage2BExecutionError(
                    f"Bundled and independent Stage 2B case results diverged for '{case_key}'.",
                    exit_code=2,
                )
            limit_message = (
                case_metrics.get("limit_message_after_independent_equivalence")
                or case_metrics.get("limit_message_after_cleanup")
            )
            if limit_message is not None:
                payload["status"] = "stopped"
                payload["metrics"]["stop_reason"] = limit_message
                payload["metrics"]["pair_runtime_seconds"] = time.perf_counter() - pair_start
                _write_json(checkpoint_path, payload)
                raise MemoryGuardExceeded(limit_message, exit_code=17)
            if stop_after_completed_cases is not None and cases_newly_computed >= stop_after_completed_cases:
                payload["status"] = "stopped"
                payload["metrics"]["stop_reason"] = "stop_after_completed_cases"
                payload["metrics"]["pair_runtime_seconds"] = time.perf_counter() - pair_start
                _write_json(checkpoint_path, payload)
                return {
                    "checkpoint_path": str(checkpoint_path),
                    "status": payload["status"],
                    "payload": payload,
                    "newly_computed_cases": cases_newly_computed,
                    "reused_completed_cases": cases_reused_from_checkpoint,
                    "starting_rss_bytes": memory_guard.starting_rss_bytes,
                    "peak_rss_bytes": memory_guard.peak_rss_bytes,
                    "ending_rss_bytes_after_cleanup": memory_guard.ending_rss_bytes,
                    "minimum_available_memory_bytes": memory_guard.minimum_available_memory_bytes,
                }
        payload["aggregated_results"] = _aggregate_pair_checkpoint_results(payload=payload, trace=trace)
        payload["status"] = "completed"
        payload["metrics"].update(
            {
                "cases_total": len(case_order),
                "cases_completed": len(completed_case_keys),
                "cases_reused_from_checkpoint": cases_reused_from_checkpoint,
                "cases_newly_computed": cases_newly_computed,
                "starting_rss_bytes": memory_guard.starting_rss_bytes,
                "peak_rss_bytes": memory_guard.peak_rss_bytes,
                "minimum_available_memory_bytes": memory_guard.minimum_available_memory_bytes,
                "pair_runtime_seconds": time.perf_counter() - pair_start,
            }
        )
    except MemoryGuardExceeded as error:
        gc.collect()
        _best_effort_trim_memory()
        memory_guard.record()
        payload["status"] = "stopped"
        payload["metrics"].update(
            {
                "cases_total": len(case_order),
                "cases_completed": len(completed_case_keys),
                "cases_reused_from_checkpoint": cases_reused_from_checkpoint,
                "cases_newly_computed": cases_newly_computed,
                "starting_rss_bytes": memory_guard.starting_rss_bytes,
                "peak_rss_bytes": memory_guard.peak_rss_bytes,
                "minimum_available_memory_bytes": memory_guard.minimum_available_memory_bytes,
                "pair_runtime_seconds": time.perf_counter() - pair_start,
                "stop_reason": str(error),
            }
        )
        _write_json(checkpoint_path, payload)
        raise

    gc.collect()
    _best_effort_trim_memory()
    memory_guard.record()
    payload["metrics"]["ending_rss_bytes_after_cleanup"] = memory_guard.ending_rss_bytes
    payload["metrics"]["memory_budget_ok"] = (
        memory_guard.peak_rss_bytes <= max_rss_bytes
        and memory_guard.minimum_available_memory_bytes >= min_free_memory_bytes
        and memory_guard.ending_rss_bytes <= max_rss_bytes
    )
    _write_json(checkpoint_path, payload)
    return {
        "checkpoint_path": str(checkpoint_path),
        "status": payload["status"],
        "payload": payload,
        "newly_computed_cases": cases_newly_computed,
        "reused_completed_cases": cases_reused_from_checkpoint,
        "starting_rss_bytes": memory_guard.starting_rss_bytes,
        "peak_rss_bytes": memory_guard.peak_rss_bytes,
        "ending_rss_bytes_after_cleanup": memory_guard.ending_rss_bytes,
        "minimum_available_memory_bytes": memory_guard.minimum_available_memory_bytes,
    }


def _resume_or_run_pair_checkpoint(
    *,
    output_dir: Path,
    trace_path: Path,
    prompt_source_sha256: str,
    recent_window: int,
    block_size: int,
    tolerances: tuple[float, ...],
    precision: int,
    random_seed: int,
    max_rss_bytes: int,
    min_free_memory_bytes: int,
    query_positions: tuple[int, ...] | None = None,
    stop_after_completed_cases: int | None = None,
) -> dict[str, Any]:
    checkpoint_path = _pair_checkpoint_path(
        output_dir=output_dir,
        recent_window=recent_window,
        block_size=block_size,
    )
    trace = validate_compact_trace(trace_path)
    normalized_positions = _normalize_query_positions(trace, query_positions)
    settings = _pair_checkpoint_settings(
        trace=trace,
        prompt_source_sha256=prompt_source_sha256,
        recent_window=recent_window,
        block_size=block_size,
        tolerances=tolerances,
        precision=precision,
        random_seed=random_seed,
        query_positions=normalized_positions,
    )
    existing = _load_pair_checkpoint(
        checkpoint_path=checkpoint_path,
        settings=settings,
        tolerances=tolerances,
    )
    if existing is not None:
        completed_case_keys = _completed_case_keys(existing, tolerances)
        if existing.get("status") == "completed" and len(completed_case_keys) == len(existing.get("case_order", ())):
            if "aggregated_results" not in existing:
                existing["aggregated_results"] = _aggregate_pair_checkpoint_results(payload=existing, trace=trace)
                _write_json(checkpoint_path, existing)
            return {"used_checkpoint": True, "checkpoint_path": str(checkpoint_path), "payload": existing}

    result = _run_pair_checkpoint_internal(
        trace_path=trace_path,
        checkpoint_path=checkpoint_path,
        prompt_source_sha256=prompt_source_sha256,
        recent_window=recent_window,
        block_size=block_size,
        tolerances=tolerances,
        precision=precision,
        random_seed=random_seed,
        max_rss_bytes=max_rss_bytes,
        min_free_memory_bytes=min_free_memory_bytes,
        query_positions=normalized_positions,
        stop_after_completed_cases=stop_after_completed_cases,
    )
    payload = _load_json(checkpoint_path)
    return {"used_checkpoint": False, "checkpoint_path": str(checkpoint_path), "payload": payload, "result": result}


def _run_one_pair_probe(
    *,
    output_dir: Path,
    trace_path: Path,
    prompt_source_sha256: str,
    recent_window: int,
    block_size: int,
    tolerances: tuple[float, ...],
    precision: int,
    random_seed: int,
    max_rss_bytes: int,
    min_free_memory_bytes: int,
    query_positions: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    first = _resume_or_run_pair_checkpoint(
        output_dir=output_dir,
        trace_path=trace_path,
        prompt_source_sha256=prompt_source_sha256,
        recent_window=recent_window,
        block_size=block_size,
        tolerances=tolerances,
        precision=precision,
        random_seed=random_seed,
        max_rss_bytes=max_rss_bytes,
        min_free_memory_bytes=min_free_memory_bytes,
        query_positions=query_positions,
    )
    checkpoint_path = Path(first["checkpoint_path"])
    first_hash = _sha256_file(checkpoint_path)

    resume_start = time.perf_counter()
    second = _resume_or_run_pair_checkpoint(
        output_dir=output_dir,
        trace_path=trace_path,
        prompt_source_sha256=prompt_source_sha256,
        recent_window=recent_window,
        block_size=block_size,
        tolerances=tolerances,
        precision=precision,
        random_seed=random_seed,
        max_rss_bytes=max_rss_bytes,
        min_free_memory_bytes=min_free_memory_bytes,
        query_positions=query_positions,
    )
    resume_elapsed_seconds = time.perf_counter() - resume_start
    second_hash = _sha256_file(checkpoint_path)
    checkpoint_payload = second["payload"]
    return {
        "first_run_used_checkpoint": bool(first["used_checkpoint"]),
        "second_run_used_checkpoint": bool(second["used_checkpoint"]),
        "resume_elapsed_seconds": resume_elapsed_seconds,
        "checkpoint_hash_unchanged_on_resume": first_hash == second_hash,
        "checkpoint_path": str(checkpoint_path),
        "payload": checkpoint_payload,
    }

def _summarize_stage2a_results(payload: dict) -> dict:
    configs = payload.get("configurations", [])
    query_case_results = [case for config in configs for case in config.get("query_case_results", [])]
    eligible_cases = [case for case in query_case_results if int(case.get("candidate_blocks", 0)) > 0]
    skipped_case_fractions = [
        float(case["certified_skipped_blocks"]) / float(case["candidate_blocks"])
        for case in eligible_cases
        if float(case["candidate_blocks"]) > 0.0
    ]
    return {
        "trace_tokens": int(payload["trace"]["sequence_length"]),
        "global_prefix_ratio_summary": dict(payload.get("global_prefix_ratio_summary", {})),
        "final_prefix_compression_ratio_min": min((cfg["final_prefix_compression_ratio"] for cfg in configs), default=1.0),
        "final_prefix_compression_ratio_max": max((cfg["final_prefix_compression_ratio"] for cfg in configs), default=1.0),
        "mean_skipped_block_fraction": float(np.mean([cfg.get("skipped_block_fraction", 0.0) for cfg in configs])) if configs else 0.0,
        "max_skipped_block_fraction": max((cfg.get("skipped_block_fraction", 0.0) for cfg in configs), default=0.0),
        "fraction_cases_with_any_skipped_block": (
            float(sum(1 for case in eligible_cases if int(case.get("certified_skipped_blocks", 0)) > 0) / len(eligible_cases))
            if eligible_cases
            else 0.0
        ),
        "mean_case_skipped_fraction": float(np.mean(skipped_case_fractions)) if skipped_case_fractions else 0.0,
        "max_case_skipped_fraction": max(skipped_case_fractions, default=0.0),
        "max_true_compression_error": max((cfg.get("max_true_compression_error", 0.0) for cfg in configs), default=0.0),
        "max_observed_skipping_error": max((cfg.get("max_observed_reconstructed_skipping_error", 0.0) for cfg in configs), default=0.0),
        "max_rigorous_skip_error_upper": max((cfg.get("max_rigorous_skip_error_upper", 0.0) for cfg in configs), default=0.0),
        "max_reference_total_error": max((cfg.get("max_reference_total_error", 0.0) for cfg in configs), default=0.0),
        "max_captured_model_total_gap": max((cfg.get("max_captured_model_total_gap", 0.0) for cfg in configs), default=0.0),
        "rigorous_interval_violation_count": sum(int(cfg.get("rigorous_interval_violation_count", 0)) for cfg in configs),
        "approximate_observed_violation_count": sum(int(cfg.get("approximate_observed_violation_count", 0)) for cfg in configs),
        "reference_decomposition_violation_count": sum(int(cfg.get("reference_decomposition_violation_count", 0)) for cfg in configs),
        "model_relative_decomposition_violation_count": sum(int(cfg.get("model_relative_decomposition_violation_count", 0)) for cfg in configs),
    }


def _compare_stage2a_stage2b(
    *,
    stage2a_summary: dict,
    stage2a_runtime_s: float,
    stage2a_peak_rss_bytes: int,
    stage2b_results: dict,
    stage2b_runtime_s: float,
    stage2b_peak_rss_bytes: int,
) -> dict:
    stage2b_configs = stage2b_results["configurations"]
    stage2b_memory = stage2b_results["memory_by_wb"]
    eligible_cases = [
        case
        for config in stage2b_configs
        for case in config["query_case_results"]
        if int(case["candidate_blocks"]) > 0
    ]
    skipped_case_fractions = [
        float(case["certified_skipped_blocks"]) / float(case["candidate_blocks"])
        for case in eligible_cases
        if float(case["candidate_blocks"]) > 0.0
    ]
    return {
        "scope_note": (
            "This is a descriptive comparison between the accepted 12-token Stage 2A smoke result and the new 256-token Stage 2B pilot. "
            "It is not a statistical benchmark: one deterministic synthetic prompt, layer 0 only, selected heads only, and no external baselines."
        ),
        "stage2a_accepted": {
            **stage2a_summary,
            "runtime_seconds_rerun_current_code": stage2a_runtime_s,
            "peak_rss_bytes_rerun_current_code": stage2a_peak_rss_bytes,
        },
        "stage2b_pilot": {
            "trace_tokens": int(stage2b_results["trace"]["sequence_length"]),
            "global_prefix_ratio_summary": dict(stage2b_results["global_prefix_ratio_summary"]),
            "final_prefix_compression_ratio_min": min((row["final_prefix_compression_ratio"] for row in stage2b_memory), default=1.0),
            "final_prefix_compression_ratio_max": max((row["final_prefix_compression_ratio"] for row in stage2b_memory), default=1.0),
            "mean_skipped_block_fraction": float(np.mean([cfg["weighted_skipped_block_fraction"] for cfg in stage2b_configs])) if stage2b_configs else 0.0,
            "max_skipped_block_fraction": max((cfg["weighted_skipped_block_fraction"] for cfg in stage2b_configs), default=0.0),
            "fraction_cases_with_any_skipped_block": (
                float(sum(1 for case in eligible_cases if int(case["certified_skipped_blocks"]) > 0) / len(eligible_cases))
                if eligible_cases
                else 0.0
            ),
            "mean_case_skipped_fraction": float(np.mean(skipped_case_fractions)) if skipped_case_fractions else 0.0,
            "max_case_skipped_fraction": max(skipped_case_fractions, default=0.0),
            "max_true_compression_error": max((cfg["max_true_compression_error"] for cfg in stage2b_configs), default=0.0),
            "max_observed_skipping_error": max((cfg["max_observed_reconstructed_skipping_error"] for cfg in stage2b_configs), default=0.0),
            "max_rigorous_skip_error_upper": max((cfg["max_rigorous_skip_error_upper"] for cfg in stage2b_configs), default=0.0),
            "max_reference_total_error": max((cfg["max_reference_total_error"] for cfg in stage2b_configs), default=0.0),
            "max_captured_model_total_gap": max((cfg["max_captured_model_total_gap"] for cfg in stage2b_configs), default=0.0),
            "rigorous_interval_violation_count": sum(int(cfg["rigorous_interval_violation_count"]) for cfg in stage2b_configs),
            "approximate_observed_violation_count": sum(int(cfg["approximate_observed_violation_count"]) for cfg in stage2b_configs),
            "reference_decomposition_violation_count": sum(int(cfg["reference_decomposition_violation_count"]) for cfg in stage2b_configs),
            "model_relative_decomposition_violation_count": sum(int(cfg["model_relative_decomposition_violation_count"]) for cfg in stage2b_configs),
            "runtime_seconds": stage2b_runtime_s,
            "peak_rss_bytes": stage2b_peak_rss_bytes,
        },
    }


def _first_projection_difference(
    optimized: list[dict[str, Any]],
    reference: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(optimized) != len(reference):
        return {"kind": "config_length", "optimized": len(optimized), "reference": len(reference)}
    for config_index, (optimized_config, reference_config) in enumerate(zip(optimized, reference)):
        if optimized_config.keys() != reference_config.keys():
            return {
                "kind": "config_keys",
                "config_index": config_index,
                "optimized_keys": sorted(optimized_config.keys()),
                "reference_keys": sorted(reference_config.keys()),
            }
        for key in optimized_config:
            if key == "query_case_results":
                optimized_cases = optimized_config[key]
                reference_cases = reference_config[key]
                if len(optimized_cases) != len(reference_cases):
                    return {
                        "kind": "case_length",
                        "config_index": config_index,
                        "optimized": len(optimized_cases),
                        "reference": len(reference_cases),
                    }
                for case_index, (optimized_case, reference_case) in enumerate(zip(optimized_cases, reference_cases)):
                    if optimized_case != reference_case:
                        for case_key in optimized_case:
                            if optimized_case[case_key] != reference_case[case_key]:
                                return {
                                    "kind": "case_field",
                                    "config_index": config_index,
                                    "case_index": case_index,
                                    "field": case_key,
                                    "optimized": optimized_case[case_key],
                                    "reference": reference_case[case_key],
                                }
                        return {
                            "kind": "case_dict",
                            "config_index": config_index,
                            "case_index": case_index,
                        }
            elif optimized_config[key] != reference_config[key]:
                return {
                    "kind": "config_field",
                    "config_index": config_index,
                    "field": key,
                    "optimized": optimized_config[key],
                    "reference": reference_config[key],
                }
    return None


def _case_results_difference(
    optimized_case_results: list[dict[str, Any]],
    reference_case_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(optimized_case_results) != len(reference_case_results):
        return {
            "kind": "tolerance_count",
            "optimized": len(optimized_case_results),
            "reference": len(reference_case_results),
        }
    for tolerance_index, (optimized_case, reference_case) in enumerate(zip(optimized_case_results, reference_case_results)):
        if optimized_case != reference_case:
            for field in optimized_case:
                if optimized_case[field] != reference_case.get(field):
                    return {
                        "kind": "case_field",
                        "tolerance_index": tolerance_index,
                        "field": field,
                        "optimized": optimized_case[field],
                        "reference": reference_case.get(field),
                    }
            return {"kind": "case_dict", "tolerance_index": tolerance_index}
    return None


def _aggregate_pair_memory_summary(
    *,
    trace: Any,
    recent_window: int,
    block_size: int,
    prefix_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not prefix_records:
        raise Stage2BExecutionError("Cannot aggregate a pair memory summary without prefix records.", exit_code=2)
    ratios = [float(record["compression_ratio"]) for record in prefix_records]
    aggregate_prefix_original_bytes = sum(int(record["original_kv_bytes"]) for record in prefix_records)
    aggregate_prefix_serialized_bytes = sum(int(record["total_compressed_bytes"]) for record in prefix_records)
    aggregate_prefix_historical_serialized_bytes = sum(int(record["compressed_historical_bytes"]) for record in prefix_records)
    total_historical_tokens = sum(int(record["historical_tokens"]) for record in prefix_records)
    final_prefix_records = [
        record for record in prefix_records if int(record["record_index"]) == trace.query_records - 1
    ]
    final_prefix_original_bytes = sum(int(record["original_kv_bytes"]) for record in final_prefix_records)
    final_prefix_serialized_bytes = sum(int(record["total_compressed_bytes"]) for record in final_prefix_records)
    final_prefix_historical_serialized_bytes = sum(int(record["compressed_historical_bytes"]) for record in final_prefix_records)
    key_squared_error_sum = sum(float(record["key_squared_error_sum"]) for record in prefix_records)
    key_element_count = sum(int(record["key_element_count"]) for record in prefix_records)
    value_squared_error_sum = sum(float(record["value_squared_error_sum"]) for record in prefix_records)
    value_element_count = sum(int(record["value_element_count"]) for record in prefix_records)
    return {
        "recent_window": recent_window,
        "block_size": block_size,
        "unique_prefix_slices": len(prefix_records),
        "unique_prefix_slices_with_history": sum(1 for record in prefix_records if int(record["historical_tokens"]) > 0),
        "aggregate_prefix_original_bytes": aggregate_prefix_original_bytes,
        "aggregate_prefix_serialized_bytes": aggregate_prefix_serialized_bytes,
        "aggregate_prefix_historical_serialized_bytes": aggregate_prefix_historical_serialized_bytes,
        "aggregate_prefix_compression_ratio": (
            float(aggregate_prefix_original_bytes / aggregate_prefix_serialized_bytes)
            if aggregate_prefix_serialized_bytes > 0
            else 1.0
        ),
        "min_prefix_compression_ratio": float(min(ratios)),
        "mean_prefix_compression_ratio": float(np.mean(np.asarray(ratios, dtype=np.float64))),
        "median_prefix_compression_ratio": float(np.median(np.asarray(ratios, dtype=np.float64))),
        "max_prefix_compression_ratio": float(max(ratios)),
        "final_prefix_original_bytes": final_prefix_original_bytes,
        "final_prefix_serialized_bytes": final_prefix_serialized_bytes,
        "final_prefix_historical_serialized_bytes": final_prefix_historical_serialized_bytes,
        "final_prefix_compression_ratio": (
            float(final_prefix_original_bytes / final_prefix_serialized_bytes)
            if final_prefix_serialized_bytes > 0
            else 1.0
        ),
        "final_prefix_values_per_selected_kv_head": int(trace.visible_lengths[-1]),
        "final_prefix_total_values_selected_kv_heads": int(trace.visible_lengths[-1]) * len(trace.selected_kv_heads),
        "bytes_per_historical_token": (
            float(aggregate_prefix_historical_serialized_bytes / total_historical_tokens)
            if total_historical_tokens > 0
            else 0.0
        ),
        "number_of_blocks": sum(int(record["number_of_blocks"]) for record in prefix_records),
        "independently_decoded_blocks_tested": sum(int(record["independent_blocks_tested"]) for record in prefix_records),
        "independent_versus_full_decode_max_difference": max(float(record["independent_decode_max_diff"]) for record in prefix_records),
        "unaffected_block_decoding_tests": sum(1 for record in prefix_records if bool(record["unaffected_block_decode_tested"])),
        "unaffected_block_decoding_passed": all(bool(record["unaffected_block_decode_passed"]) for record in prefix_records),
        "serializer_roundtrip_checks_passed": all(bool(record["authoritative_serialized_roundtrip_used"]) for record in prefix_records),
        "max_key_abs_error": max(float(record["max_key_abs_error"]) for record in prefix_records),
        "key_rmse": float(np.sqrt(key_squared_error_sum / key_element_count)) if key_element_count > 0 else 0.0,
        "max_value_abs_error": max(float(record["max_value_abs_error"]) for record in prefix_records),
        "value_rmse": float(np.sqrt(value_squared_error_sum / value_element_count)) if value_element_count > 0 else 0.0,
    }


def _aggregate_pair_configurations(
    *,
    recent_window: int,
    block_size: int,
    tolerances: tuple[float, ...],
    case_order: tuple[dict[str, Any], ...],
    case_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    tolerance_to_results: dict[str, list[dict[str, Any]]] = {_tolerance_key(value): [] for value in tolerances}
    for case_entry in case_order:
        case_record = case_records[case_entry["case_key"]]
        for tolerance_key, case_result in case_record["results_by_tolerance"].items():
            tolerance_to_results[tolerance_key].append(case_result)

    config_summaries: list[dict[str, Any]] = []
    for tolerance in tolerances:
        tolerance_key = _tolerance_key(tolerance)
        case_results = tolerance_to_results[tolerance_key]
        candidate_blocks = sum(int(case["candidate_blocks"]) for case in case_results)
        certified_skipped_blocks = sum(int(case["certified_skipped_blocks"]) for case in case_results)
        decoded_blocks = sum(int(case["decoded_blocks"]) for case in case_results)
        eligible_cases = [case for case in case_results if int(case["candidate_blocks"]) > 0]
        skipped_case_fractions = [
            float(case["certified_skipped_blocks"]) / float(case["candidate_blocks"])
            for case in eligible_cases
            if int(case["candidate_blocks"]) > 0
        ]
        nonzero_ratios = [case["bound_to_observed_ratio"] for case in case_results if case["bound_to_observed_ratio"] is not None]
        config_summaries.append(
            {
                "recent_window": recent_window,
                "block_size": block_size,
                "tolerance": float(tolerance),
                "evaluated_query_cases": len(case_results),
                "eligible_query_cases": len(eligible_cases),
                "ineligible_query_cases": len(case_results) - len(eligible_cases),
                "total_candidate_blocks": candidate_blocks,
                "certified_skipped_blocks": certified_skipped_blocks,
                "decoded_blocks": decoded_blocks,
                "weighted_skipped_block_fraction": float(certified_skipped_blocks / candidate_blocks) if candidate_blocks > 0 else 0.0,
                "weighted_decoded_block_fraction": float(decoded_blocks / candidate_blocks) if candidate_blocks > 0 else 0.0,
                "mean_case_skipped_fraction": float(np.mean(np.asarray(skipped_case_fractions, dtype=np.float64))) if skipped_case_fractions else 0.0,
                "median_case_skipped_fraction": float(np.median(np.asarray(skipped_case_fractions, dtype=np.float64))) if skipped_case_fractions else 0.0,
                "max_case_skipped_fraction": max(skipped_case_fractions, default=0.0),
                "fraction_cases_with_any_skipped_block": (
                    float(sum(1 for case in eligible_cases if int(case["certified_skipped_blocks"]) > 0) / len(eligible_cases))
                    if eligible_cases
                    else 0.0
                ),
                "max_certified_bound": max(float(case["certificate_bound_upper_float"]) for case in case_results),
                "max_observed_reconstructed_skipping_error": max(float(case["observed_skip_error"]) for case in case_results),
                "max_rigorous_skip_error_upper": max(float(case["rigorous_skip_error_upper_float"]) for case in case_results),
                "min_bound_to_observed_ratio_nonzero": min(nonzero_ratios) if nonzero_ratios else None,
                "max_bound_to_observed_ratio_nonzero": max(nonzero_ratios) if nonzero_ratios else None,
                "rigorous_interval_violation_count": sum(1 for case in case_results if bool(case["rigorous_interval_violation"])),
                "approximate_observed_violation_count": sum(1 for case in case_results if bool(case["approximate_observed_violation"])),
                "false_safe_violation_count": sum(1 for case in case_results if bool(case["rigorous_interval_violation"])),
                "fallback_count": sum(1 for case in case_results if bool(case["numerical_fallback_used"])),
                "max_model_reference_gap": max(float(case["model_reference_gap"]) for case in case_results),
                "max_true_compression_error": max(float(case["compression_error"]) for case in case_results),
                "max_reference_total_error": max(float(case["reference_total_error"]) for case in case_results),
                "max_captured_model_total_gap": max(float(case["captured_model_total_gap"]) for case in case_results),
                "reference_decomposition_violation_count": sum(
                    1
                    for case in case_results
                    if float(case["reference_decomposition_lhs"]) > float(case["reference_decomposition_rhs"]) + 1e-9
                ),
                "model_relative_decomposition_violation_count": sum(
                    1
                    for case in case_results
                    if float(case["model_relative_decomposition_lhs"]) > float(case["model_relative_decomposition_rhs"]) + 1e-9
                ),
                "query_case_results": case_results,
            }
        )
    return config_summaries


def _aggregate_pair_checkpoint_results(*, payload: dict[str, Any], trace: Any) -> dict[str, Any]:
    case_order = tuple(payload["case_order"])
    case_records = payload["case_records"]
    tolerances = tuple(float(value) for value in payload["tolerances"])
    prefix_records = list(payload["prefix_records"].values())
    completed_cases = _completed_case_keys(payload, tolerances)
    if len(completed_cases) != len(case_order):
        raise Stage2BExecutionError("Cannot aggregate a Stage 2B pair checkpoint before all cases are complete.", exit_code=2)
    memory_summary = _aggregate_pair_memory_summary(
        trace=trace,
        recent_window=int(payload["recent_window"]),
        block_size=int(payload["block_size"]),
        prefix_records=prefix_records,
    )
    config_summaries = _aggregate_pair_configurations(
        recent_window=int(payload["recent_window"]),
        block_size=int(payload["block_size"]),
        tolerances=tolerances,
        case_order=case_order,
        case_records=case_records,
    )
    ratios = [float(record["compression_ratio"]) for record in prefix_records]
    return {
        "trace": {
            "trace_path": str(trace.trace_path),
            "trace_sha256": trace.trace_sha256,
            "sequence_length": int(trace.sequence_length),
            "queries_shape": [int(dim) for dim in trace.queries.shape],
            "final_keys_shape": [int(dim) for dim in trace.final_keys.shape],
            "final_values_shape": [int(dim) for dim in trace.final_values.shape],
            "model_head_outputs_shape": [int(dim) for dim in trace.model_head_outputs.shape],
            "selected_query_heads": list(trace.selected_query_heads),
            "selected_kv_heads": list(trace.selected_kv_heads),
            "query_to_kv_heads": list(trace.query_to_kv_heads),
            "visible_lengths": list(trace.visible_lengths),
            "query_positions": list(trace.query_positions),
            "evaluated_query_positions": list(payload["query_positions"]),
            "scaling": float(trace.scaling),
            "canonical_one_over_sqrt_head_dim": _canonical_attention_scale(trace.head_dim),
            "scaling_minus_canonical": float(trace.scaling - _canonical_attention_scale(trace.head_dim)),
        },
        "precision": int(payload["precision"]),
        "random_seed": int(payload["random_seed"]),
        "global_prefix_ratio_summary": {
            "minimum": float(min(ratios)),
            "mean": float(np.mean(np.asarray(ratios, dtype=np.float64))),
            "median": float(np.median(np.asarray(ratios, dtype=np.float64))),
            "maximum": float(max(ratios)),
        },
        "memory_by_wb": [memory_summary],
        "configurations": config_summaries,
    }


def _run_case_with_equivalence(
    *,
    trace: Any,
    record_index: int,
    query_local_index: int,
    recent_window: int,
    block_size: int,
    tolerances: tuple[float, ...],
    precision: int,
    memory_guard: StreamingMemoryGuard | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    kv_head_global = int(trace.query_to_kv_heads[query_local_index])
    case_start = time.perf_counter()
    rss_before_case = _rss_bytes()
    prefix_keys, prefix_values = trace.kv_prefix(record_index=record_index, kv_head_global=kv_head_global)
    prefix_result = _build_prefix_result(
        record_index=record_index,
        kv_head_global=kv_head_global,
        prefix_keys=prefix_keys,
        prefix_values=prefix_values,
        recent_window=recent_window,
        block_size=block_size,
        precision=precision,
    )
    rss_after_prefix = None
    rss_after_bundled = None
    rss_after_reference = None
    available_after_reference = None
    if memory_guard is not None:
        rss_after_prefix, _ = memory_guard.check("after_prefix_preparation")
    try:
        base_case = _build_query_case_base(
            trace=trace,
            prefix_result=prefix_result,
            query_local_index=query_local_index,
            precision=precision,
        )
        bundled_results = _run_query_case_tolerance_bundle(
            trace=trace,
            prefix_result=prefix_result,
            query_local_index=query_local_index,
            tolerances=tolerances,
            precision=precision,
            base_case=base_case,
        )
        optimized_case_results = [_projection_case_fields(case.__dict__) for case in bundled_results]
        del bundled_results
        rss_after_bundled = None
        if memory_guard is not None:
            rss_after_bundled, _ = memory_guard.check("after_progressive_certification")

        reference_case_results: list[dict[str, Any]] = []
        for tolerance in tolerances:
            reference_result = _run_query_case(
                trace=trace,
                prefix_result=prefix_result,
                query_local_index=query_local_index,
                tolerance=float(tolerance),
                precision=precision,
                base_case=base_case,
            )
            reference_case_results.append(_projection_case_fields(reference_result.__dict__))
            del reference_result
        rss_after_reference = None
        available_after_reference = None
        if memory_guard is not None:
            rss_after_reference, available_after_reference = memory_guard.record()
    finally:
        del prefix_keys
        del prefix_values

    first_difference = _case_results_difference(optimized_case_results, reference_case_results)
    exact_match = first_difference is None
    tolerance_map = {
        _tolerance_key(tolerance): optimized_case_results[index]
        for index, tolerance in enumerate(tolerances)
    }
    case_record = {
        "status": "completed",
        "record_index": int(record_index),
        "query_local_index": int(query_local_index),
        "query_head_global": int(trace.selected_query_heads[query_local_index]),
        "kv_head_global": int(kv_head_global),
        "query_position": int(trace.query_positions[record_index]),
        "results_by_tolerance": tolerance_map,
        "equivalence": {
            "exact_match": exact_match,
            "first_difference": first_difference,
        },
    }
    prefix_record = _serialize_prefix_record(prefix_result)
    del base_case
    del prefix_result
    gc.collect()
    _best_effort_trim_memory()
    rss_after_cleanup = _rss_bytes()
    available_after_cleanup = _available_memory_bytes()
    limit_message_after_independent_equivalence = None
    limit_message_after_cleanup = None
    if memory_guard is not None:
        limit_message_after_independent_equivalence = memory_guard.limit_message(
            "after_independent_equivalence",
            rss=rss_after_reference if rss_after_reference is not None else rss_after_cleanup,
            available=available_after_reference if available_after_reference is not None else available_after_cleanup,
        )
        rss_after_cleanup, available_after_cleanup = memory_guard.record()
        limit_message_after_cleanup = memory_guard.limit_message(
            "after_case_cleanup",
            rss=rss_after_cleanup,
            available=available_after_cleanup,
        )
    case_record["metrics"] = {
        "runtime_seconds": time.perf_counter() - case_start,
        "rss_before_case": rss_before_case,
        "rss_after_prefix_preparation": rss_after_prefix,
        "rss_after_progressive_certification": rss_after_bundled,
        "rss_after_independent_equivalence": rss_after_reference,
        "available_memory_after_independent_equivalence": available_after_reference,
        "rss_after_cleanup": rss_after_cleanup,
        "available_memory_after_cleanup": available_after_cleanup,
        "limit_message_after_independent_equivalence": limit_message_after_independent_equivalence,
        "limit_message_after_cleanup": limit_message_after_cleanup,
    }
    return case_record, prefix_record, case_record["metrics"]


def _render_report(results: dict) -> str:
    prompt = results["prompt"]
    capture = results["capture"]
    trace = results["trace"]
    comparison = results["stage2a_comparison"]
    memory_rows = results["memory_by_wb"]
    configs = results["configurations"]
    lines = [
        "# RACK-KV Stage 2B Layer-0 256-Token Pilot",
        "",
        "This is a controlled 256-token layer-0 pilot.",
        "It is longer than the 12-token smoke test but not a true long-context benchmark.",
        "Only selected heads are evaluated.",
        "Only one deterministic synthetic prompt is used.",
        "No external baseline comparison exists yet.",
        "No multi-layer or generation integration exists yet.",
        "Compression error is empirical.",
        "Only skipping relative to reconstructed compressed KV is certified.",
        "No end-to-end quality or speedup claim follows.",
        "No novelty conclusion follows from this experiment alone.",
        "",
        "## Prompt",
        "",
        f"- Prompt source SHA-256: `{prompt['prompt_source_sha256']}`",
        f"- Prompt token count: `{prompt['prompt_token_count']}`",
        f"- Prompt seed: `{prompt['seed']}`",
        f"- Prompt source file: `{prompt['prompt_source_path']}`",
        f"- Prompt decoded file: `{prompt['prompt_decoded_path']}`",
        f"- Prompt metadata file: `{prompt['prompt_metadata_path']}`",
        "",
        "## Capture Validation",
        "",
        f"- Checkpoint repo: `{capture['checkpoint_repo']}`",
        f"- Checkpoint revision: `{capture['checkpoint_revision']}`",
        f"- TLS verification: `{capture['tls_verification']}`",
        f"- TLS fallback reason: `{capture['tls_fallback_reason']}`",
        f"- Trace SHA-256: `{trace['trace_sha256']}`",
        f"- Trace path: `{trace['trace_path']}`",
        f"- Trace size bytes: `{trace['trace_size_bytes']}`",
        f"- Queries shape: `{trace['queries_shape']}`",
        f"- Final keys shape: `{trace['final_keys_shape']}`",
        f"- Final values shape: `{trace['final_values_shape']}`",
        f"- Model head outputs shape: `{trace['model_head_outputs_shape']}`",
        f"- Selected query heads: `{trace['selected_query_heads']}`",
        f"- Selected KV heads: `{trace['selected_kv_heads']}`",
        f"- Query-to-KV mapping: `{trace['query_to_kv_heads']}`",
        f"- Evaluated query positions: `{trace['evaluated_query_positions']}`",
        f"- Saved attention scaling: `{trace['scaling']}`",
        f"- Canonical 1/sqrt(head_dim): `{trace['canonical_one_over_sqrt_head_dim']}`",
        f"- Scaling difference: `{trace['scaling_minus_canonical']}`",
        f"- Stock projected-output max abs diff: `{capture['stock_projected_output_max_abs_diff']}`",
        f"- Stock projected-output max rel diff: `{capture['stock_projected_output_max_rel_diff']}`",
        f"- Stock decoder-output max abs diff: `{capture['stock_decoder_output_max_abs_diff']}`",
        f"- Stock decoder-output max rel diff: `{capture['stock_decoder_output_max_rel_diff']}`",
        f"- Stock cache-key max abs diff: `{capture['stock_cache_key_max_abs_diff']}`",
        f"- Stock cache-key max rel diff: `{capture['stock_cache_key_max_rel_diff']}`",
        f"- Stock cache-value max abs diff: `{capture['stock_cache_value_max_abs_diff']}`",
        f"- Stock cache-value max rel diff: `{capture['stock_cache_value_max_rel_diff']}`",
        f"- Compact replay max abs diff: `{capture['max_compact_replay_abs_diff']}`",
        "",
        "## Memory By W/B",
        "",
        "| W | B | agg ratio | min prefix ratio | mean prefix ratio | median prefix ratio | max prefix ratio | final prefix ratio | bytes/historical token |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in memory_rows:
        lines.append(
            "| {recent_window} | {block_size} | {aggregate_prefix_compression_ratio:.6f} | "
            "{min_prefix_compression_ratio:.16f} | {mean_prefix_compression_ratio:.16f} | "
            "{median_prefix_compression_ratio:.16f} | {max_prefix_compression_ratio:.16f} | "
            "{final_prefix_compression_ratio:.16f} | {bytes_per_historical_token:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Some very short or metadata-heavy prefixes may expand because block headers, certificate metadata, and container indices dominate payload bytes.",
            "",
            "## Configuration Sweep",
            "",
            "| W | B | tol | eligible cases | weighted skip frac | mean case skip frac | median case skip frac | max case skip frac | cases with any skip | max compression err | max observed skip err | max rigorous skip upper | rigorous viols | fallbacks |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for config in configs:
        lines.append(
            "| {recent_window} | {block_size} | {tolerance:.6f} | {eligible_query_cases} | "
            "{weighted_skipped_block_fraction:.6f} | {mean_case_skipped_fraction:.6f} | "
            "{median_case_skipped_fraction:.6f} | {max_case_skipped_fraction:.6f} | "
            "{fraction_cases_with_any_skipped_block:.6f} | {max_true_compression_error:.6f} | "
            "{max_observed_reconstructed_skipping_error:.6f} | {max_rigorous_skip_error_upper:.6f} | "
            "{rigorous_interval_violation_count} | {fallback_count} |".format(
                **config
            )
        )
    lines.extend(
        [
            "",
            "## Stage 2A Versus Stage 2B",
            "",
            comparison["scope_note"],
            "",
            f"- Stage 2A accepted trace tokens: `{comparison['stage2a_accepted']['trace_tokens']}`",
            f"- Stage 2A final-prefix compression-ratio range: `{comparison['stage2a_accepted']['final_prefix_compression_ratio_min']}` to `{comparison['stage2a_accepted']['final_prefix_compression_ratio_max']}`",
            f"- Stage 2A global prefix ratios: `{comparison['stage2a_accepted']['global_prefix_ratio_summary']}`",
            f"- Stage 2A mean/max skipped block fraction: `{comparison['stage2a_accepted']['mean_skipped_block_fraction']}` / `{comparison['stage2a_accepted']['max_skipped_block_fraction']}`",
            f"- Stage 2A fraction of eligible cases with nonzero skipping: `{comparison['stage2a_accepted']['fraction_cases_with_any_skipped_block']}`",
            f"- Stage 2A max true compression error: `{comparison['stage2a_accepted']['max_true_compression_error']}`",
            f"- Stage 2A max observed skipping error: `{comparison['stage2a_accepted']['max_observed_skipping_error']}`",
            f"- Stage 2A rigorous interval violations: `{comparison['stage2a_accepted']['rigorous_interval_violation_count']}`",
            f"- Stage 2A runtime/peak RSS (current-code rerun for measurement only): `{comparison['stage2a_accepted']['runtime_seconds_rerun_current_code']}` s / `{comparison['stage2a_accepted']['peak_rss_bytes_rerun_current_code']}` bytes",
            "",
            f"- Stage 2B trace tokens: `{comparison['stage2b_pilot']['trace_tokens']}`",
            f"- Stage 2B final-prefix compression-ratio range: `{comparison['stage2b_pilot']['final_prefix_compression_ratio_min']}` to `{comparison['stage2b_pilot']['final_prefix_compression_ratio_max']}`",
            f"- Stage 2B global prefix ratios: `{comparison['stage2b_pilot']['global_prefix_ratio_summary']}`",
            f"- Stage 2B mean/max skipped block fraction: `{comparison['stage2b_pilot']['mean_skipped_block_fraction']}` / `{comparison['stage2b_pilot']['max_skipped_block_fraction']}`",
            f"- Stage 2B fraction of eligible cases with nonzero skipping: `{comparison['stage2b_pilot']['fraction_cases_with_any_skipped_block']}`",
            f"- Stage 2B max true compression error: `{comparison['stage2b_pilot']['max_true_compression_error']}`",
            f"- Stage 2B max observed skipping error: `{comparison['stage2b_pilot']['max_observed_skipping_error']}`",
            f"- Stage 2B rigorous interval violations: `{comparison['stage2b_pilot']['rigorous_interval_violation_count']}`",
            f"- Stage 2B runtime/peak RSS: `{comparison['stage2b_pilot']['runtime_seconds']}` s / `{comparison['stage2b_pilot']['peak_rss_bytes']}` bytes",
            "",
            "## Runtime And Provenance",
            "",
            f"- Capture runtime seconds: `{results['capture_runtime_seconds']}`",
            f"- Capture peak RSS bytes: `{results['capture_peak_rss_bytes']}`",
            f"- Experiment runtime seconds: `{results['experiment_runtime_seconds']}`",
            f"- Experiment peak RSS bytes: `{results['experiment_peak_rss_bytes']}`",
            f"- Review package size bytes: `{results['review_package_size_bytes']}`",
            f"- Source snapshot SHA-256: `{results['source_snapshot_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _package_review(
    *,
    output_dir: Path,
    review_zip_path: Path,
    files_to_copy: list[Path],
    manifest_output_path: Path,
) -> dict:
    package_root = output_dir / "review_package"
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)

    for relative_path in REVIEW_SOURCE_FILES:
        source = Path(relative_path)
        destination = package_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for source_path in files_to_copy:
        destination = package_root / source_path.name
        shutil.copy2(source_path, destination)

    source_snapshot_sha256 = _source_snapshot_sha256(REVIEW_SOURCE_FILES)
    manifest = {
        "source_revision": _try_git_revision(),
        "source_snapshot_sha256": source_snapshot_sha256,
        "files": [],
    }
    for path in sorted(package_root.rglob("*")):
        if path.is_file():
            manifest["files"].append(
                {
                    "relative_path": path.relative_to(package_root).as_posix(),
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    manifest_file = package_root / DEFAULT_MANIFEST_FILENAME
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest_output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with zipfile.ZipFile(review_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in package_root.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(package_root))

    return {
        "source_revision": manifest["source_revision"],
        "source_snapshot_sha256": source_snapshot_sha256,
        "review_zip_path": str(review_zip_path),
        "review_package_size_bytes": review_zip_path.stat().st_size,
    }


def _verify_packaged_outputs(*, review_zip_path: Path) -> None:
    with zipfile.ZipFile(review_zip_path, "r") as archive:
        manifest = json.loads(archive.read(DEFAULT_MANIFEST_FILENAME).decode("utf-8"))
        for entry in manifest["files"]:
            payload = archive.read(entry["relative_path"])
            digest = hashlib.sha256(payload).hexdigest()
            if digest != entry["sha256"]:
                raise RuntimeError(f"Manifest hash mismatch for {entry['relative_path']}.")
        json.loads(archive.read(DEFAULT_RESULTS_FILENAME).decode("utf-8"))
        archive.read(DEFAULT_CAPTURE_TRACE_FILENAME)
        archive.read(DEFAULT_TEST_LOG_FILENAME).decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the controlled Stage 2B 256-token layer-0 RACK-KV pilot.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_LLAMA31_BASE_REPO)
    parser.add_argument("--repo-revision", type=str, default=PINNED_LLAMA31_REVISION)
    parser.add_argument("--selected-query-heads", type=str, default="0,1,4")
    parser.add_argument("--recent-windows", type=str, default="16,32")
    parser.add_argument("--block-sizes", type=str, default="8,16,32")
    parser.add_argument("--tolerances", type=str, default="0.0,0.01,0.05,0.1")
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-log-path", type=Path, default=None)
    parser.add_argument("--review-zip-path", type=Path, default=None)
    parser.add_argument("--pair-recent-window", type=int, default=None)
    parser.add_argument("--pair-block-size", type=int, default=None)
    parser.add_argument("--pair-output-json", type=Path, default=None)
    parser.add_argument("--query-positions", type=str, default=None)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--max-rss-gb", type=float, default=8.0)
    parser.add_argument("--min-free-memory-gb", type=float, default=2.0)
    args = parser.parse_args()

    output_dir = args.output_dir
    capture_dir = output_dir / DEFAULT_CAPTURE_DIRNAME
    prompt_dir = output_dir / DEFAULT_PROMPT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    existing_trace_path = capture_dir / DEFAULT_CAPTURE_TRACE_FILENAME
    existing_capture_report_path = capture_dir / DEFAULT_CAPTURE_REPORT_FILENAME
    existing_capture_dependency_path = capture_dir / DEFAULT_CAPTURE_DEPENDENCY_FILENAME
    prompt_paths = {
        "source_path": prompt_dir / "stage2b_prompt_source.txt",
        "decoded_path": prompt_dir / "stage2b_prompt_decoded_256.txt",
        "metadata_path": prompt_dir / "stage2b_prompt_metadata.json",
    }
    reuse_existing_capture = (
        existing_trace_path.exists()
        and existing_capture_report_path.exists()
        and existing_capture_dependency_path.exists()
        and all(path.exists() for path in prompt_paths.values())
    )

    if reuse_existing_capture:
        prompt_artifacts = _load_existing_prompt_artifacts(prompt_paths["metadata_path"])
        capture_result = _reuse_existing_capture(
            trace_path=existing_trace_path,
            capture_report_path=existing_capture_report_path,
            dependency_report_path=existing_capture_dependency_path,
        )
        capture_runtime_s = 0.0
        capture_peak_rss_bytes = int(capture_result.peak_rss_bytes)
        capture_report = _load_json(capture_result.capture_report_path)
        capture_tls = {
            "tls_verification": bool(capture_report["tls_verification"]),
            "fallback_reason": "reused existing validated capture",
        }
        asset_provenance = {
            "tls_verification": bool(capture_report["tls_verification"]),
            "repo_revision_source": str(capture_report.get("repo_revision_source", "provided")),
            "asset_sources": dict(capture_report.get("asset_sources", {})),
            "asset_hashes": dict(capture_report.get("asset_hashes", {})),
            "fallback_reason": "reused existing validated capture",
        }
    else:
        _copy_stage1_assets_if_present(output_dir=capture_dir, repo_revision=args.repo_revision)
        asset_dir, asset_provenance = _ensure_assets_with_tls_fallback(
            output_dir=capture_dir,
            repo_id=args.repo_id,
            repo_revision=args.repo_revision,
        )
        prompt_artifacts = generate_stage2b_prompt_artifacts(
            asset_dir=asset_dir,
            seed=STAGE2B_PROMPT_SEED,
            target_token_count=STAGE2B_TARGET_TOKEN_COUNT,
        )
        prompt_paths = write_stage2b_prompt_artifacts(prompt_dir, prompt_artifacts)

        selected_query_heads = _parse_csv_ints(args.selected_query_heads)
        capture_result, capture_runtime_s, capture_peak_rss_bytes, capture_tls = _capture_with_tls_fallback(
            capture_dir=capture_dir,
            repo_id=args.repo_id,
            repo_revision=args.repo_revision,
            prompt_text=prompt_artifacts.source_text,
            prompt_token_ids=prompt_artifacts.token_ids,
            selected_query_heads=selected_query_heads,
        )
        capture_report = _load_json(capture_result.capture_report_path)

    selected_query_heads = _parse_csv_ints(args.selected_query_heads)
    recent_windows = _parse_csv_ints(args.recent_windows)
    block_sizes = _parse_csv_ints(args.block_sizes)
    tolerances = _parse_csv_floats(args.tolerances)
    explicit_query_positions = _parse_csv_ints(args.query_positions) if args.query_positions else None
    max_rss_bytes = _gb_to_bytes(args.max_rss_gb)
    min_free_memory_bytes = _gb_to_bytes(args.min_free_memory_gb)
    lock_path = output_dir / DEFAULT_RESUME_LOCK_FILENAME

    results_path = output_dir / DEFAULT_RESULTS_FILENAME
    report_path = output_dir / DEFAULT_REPORT_FILENAME
    dependency_path = output_dir / DEFAULT_DEPENDENCY_FILENAME
    capture_command_path = output_dir / DEFAULT_CAPTURE_COMMAND_FILENAME
    experiment_command_path = output_dir / DEFAULT_EXPERIMENT_COMMAND_FILENAME
    test_command_path = output_dir / DEFAULT_TEST_COMMAND_FILENAME
    test_return_code_path = output_dir / DEFAULT_TEST_RETURN_CODE_FILENAME
    manifest_path = output_dir / DEFAULT_MANIFEST_FILENAME
    review_zip_path = args.review_zip_path or (output_dir / DEFAULT_REVIEW_ZIP)
    test_log_path = args.test_log_path or (output_dir / DEFAULT_TEST_LOG_FILENAME)

    if args.pair_output_json is not None:
        if args.pair_recent_window is None or args.pair_block_size is None:
            raise RuntimeError("Internal pair mode requires --pair-recent-window and --pair-block-size.")
        lock_payload = _acquire_resume_lock(lock_path)
        try:
            summary = _run_pair_checkpoint_internal(
                trace_path=capture_result.trace_path,
                checkpoint_path=args.pair_output_json,
                prompt_source_sha256=prompt_artifacts.source_sha256,
                recent_window=args.pair_recent_window,
                block_size=args.pair_block_size,
                tolerances=tolerances,
                precision=args.precision,
                random_seed=args.seed,
                max_rss_bytes=max_rss_bytes,
                min_free_memory_bytes=min_free_memory_bytes,
                query_positions=explicit_query_positions,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        finally:
            _release_resume_lock(lock_path, lock_payload)

    if args.probe_only:
        if len(recent_windows) != 1 or len(block_sizes) != 1:
            raise RuntimeError("Probe mode requires exactly one recent window and one block size.")
        lock_payload = _acquire_resume_lock(lock_path)
        try:
            probe = _run_one_pair_probe(
                output_dir=output_dir,
                trace_path=capture_result.trace_path,
                prompt_source_sha256=prompt_artifacts.source_sha256,
                recent_window=recent_windows[0],
                block_size=block_sizes[0],
                tolerances=tolerances,
                precision=args.precision,
                random_seed=args.seed,
                max_rss_bytes=max_rss_bytes,
                min_free_memory_bytes=min_free_memory_bytes,
                query_positions=explicit_query_positions,
            )
        finally:
            _release_resume_lock(lock_path, lock_payload)

        payload = probe["payload"]
        first_config = payload["aggregated_results"]["configurations"][0]
        print(
            json.dumps(
                {
                    "checkpoint_path": probe["checkpoint_path"],
                    "first_run_used_checkpoint": probe["first_run_used_checkpoint"],
                    "second_run_used_checkpoint": probe["second_run_used_checkpoint"],
                    "checkpoint_hash_unchanged_on_resume": probe["checkpoint_hash_unchanged_on_resume"],
                    "resume_elapsed_seconds": probe["resume_elapsed_seconds"],
                    "equivalence_exact_match": payload["equivalence"]["exact_match"],
                    "equivalence_first_difference": payload["equivalence"]["first_difference"],
                    "runtime_seconds": payload["metrics"]["pair_runtime_seconds"],
                    "starting_rss_bytes": payload["metrics"]["starting_rss_bytes"],
                    "peak_rss_bytes": payload["metrics"]["peak_rss_bytes"],
                    "ending_rss_bytes_after_cleanup": payload["metrics"]["ending_rss_bytes_after_cleanup"],
                    "minimum_available_memory_bytes": payload["metrics"]["minimum_available_memory_bytes"],
                    "memory_budget_ok": payload["metrics"]["memory_budget_ok"],
                    "rigorous_interval_violation_count": first_config["rigorous_interval_violation_count"],
                    "approximate_observed_violation_count": first_config["approximate_observed_violation_count"],
                    "reference_decomposition_violation_count": first_config["reference_decomposition_violation_count"],
                    "model_relative_decomposition_violation_count": first_config["model_relative_decomposition_violation_count"],
                    "fallback_count": first_config["fallback_count"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    stage2b_result, experiment_runtime_s, experiment_peak_rss_bytes = _measure_call(
        run_stage2b_pilot,
        trace_path=capture_result.trace_path,
        recent_windows=recent_windows,
        block_sizes=block_sizes,
        tolerances=tolerances,
        precision=args.precision,
        random_seed=args.seed,
    )
    stage2b_results = stage2b_result_to_dict(stage2b_result)
    test_capture = _run_tests_capture(
        test_command_path=test_command_path,
        test_return_code_path=test_return_code_path,
        test_log_path=test_log_path,
    )

    stage2a_accepted_payload = _load_json(DEFAULT_STAGE2A_RESULTS_PATH)
    stage2a_summary = _summarize_stage2a_results(stage2a_accepted_payload)
    _, stage2a_runtime_s, stage2a_peak_rss_bytes = _measure_call(
        run_stage2_smoke_experiment,
        trace_path=DEFAULT_STAGE2A_TRACE_PATH,
        recent_windows=(1, 2, 4),
        block_sizes=(2, 4),
        tolerances=(0.0, 0.01, 0.05, 0.1),
        precision=args.precision,
        random_seed=args.seed,
    )

    trace_path = capture_result.trace_path
    stage2b_results.update(
        {
            "prompt": {
                "seed": prompt_artifacts.seed,
                "prompt_source_sha256": prompt_artifacts.source_sha256,
                "prompt_token_count": len(prompt_artifacts.token_ids),
                "prompt_source_path": str(prompt_paths["source_path"]),
                "prompt_decoded_path": str(prompt_paths["decoded_path"]),
                "prompt_metadata_path": str(prompt_paths["metadata_path"]),
            },
            "capture": {
                "checkpoint_repo": capture_result.checkpoint.repo_id,
                "checkpoint_revision": capture_result.checkpoint.repo_revision,
                "tls_verification": capture_tls["tls_verification"],
                "tls_fallback_reason": capture_tls["fallback_reason"],
                "asset_provenance": asset_provenance,
                "max_roundtrip_abs_diff": capture_result.max_roundtrip_abs_diff,
                "max_compact_replay_abs_diff": capture_result.max_compact_replay_abs_diff,
                "stock_projected_output_max_abs_diff": capture_result.stock_projected_output_max_abs_diff,
                "stock_projected_output_max_rel_diff": capture_result.stock_projected_output_max_rel_diff,
                "stock_decoder_output_max_abs_diff": capture_result.stock_decoder_output_max_abs_diff,
                "stock_decoder_output_max_rel_diff": capture_result.stock_decoder_output_max_rel_diff,
                "stock_cache_key_max_abs_diff": capture_result.stock_cache_key_max_abs_diff,
                "stock_cache_key_max_rel_diff": capture_result.stock_cache_key_max_rel_diff,
                "stock_cache_value_max_abs_diff": capture_result.stock_cache_value_max_abs_diff,
                "stock_cache_value_max_rel_diff": capture_result.stock_cache_value_max_rel_diff,
            },
            "trace": {
                **stage2b_results["trace"],
                "trace_size_bytes": trace_path.stat().st_size,
                "checkpoint_repo": capture_result.checkpoint.repo_id,
                "checkpoint_revision": capture_result.checkpoint.repo_revision,
            },
            "capture_runtime_seconds": capture_runtime_s,
            "capture_peak_rss_bytes": capture_peak_rss_bytes,
            "experiment_runtime_seconds": experiment_runtime_s,
            "experiment_peak_rss_bytes": experiment_peak_rss_bytes,
            "test_capture": test_capture,
        }
    )
    comparison = _compare_stage2a_stage2b(
        stage2a_summary=stage2a_summary,
        stage2a_runtime_s=stage2a_runtime_s,
        stage2a_peak_rss_bytes=stage2a_peak_rss_bytes,
        stage2b_results=stage2b_results,
        stage2b_runtime_s=experiment_runtime_s,
        stage2b_peak_rss_bytes=experiment_peak_rss_bytes,
    )
    stage2b_results["stage2a_comparison"] = comparison
    stage2b_results["source_snapshot_sha256"] = _source_snapshot_sha256(REVIEW_SOURCE_FILES)

    capture_command_path.write_text(
        subprocess.list2cmdline(
            [
                sys.executable,
                "scripts/run_minimal_llama31_capture.py",
                "--output-dir",
                str(capture_dir),
                "--repo-id",
                args.repo_id,
                "--repo-revision",
                args.repo_revision,
                "--prompt-file",
                str(prompt_paths["source_path"]),
                "--prompt-token-ids-json",
                str(prompt_paths["metadata_path"]),
                "--layer-index",
                "0",
                "--selected-query-heads",
                args.selected_query_heads,
            ]
            + ([] if capture_tls["tls_verification"] else ["--allow-insecure-tls"])
        ),
        encoding="utf-8",
    )
    experiment_command_path.write_text(" ".join(sys.argv), encoding="utf-8")
    _write_json(results_path, stage2b_results)
    dependency_payload = {
        **_dependency_versions(),
        "source_snapshot_sha256": stage2b_results["source_snapshot_sha256"],
        "repo_id": args.repo_id,
        "repo_revision": args.repo_revision,
        "trace_sha256": _sha256_file(trace_path),
    }
    _write_json(dependency_path, dependency_payload)

    review_package_stub = {"review_package_size_bytes": 0}
    stage2b_results["review_package_size_bytes"] = 0
    report_path.write_text(_render_report(stage2b_results), encoding="utf-8")
    _write_json(results_path, stage2b_results)

    files_to_copy = [
        results_path,
        report_path,
        dependency_path,
        capture_command_path,
        experiment_command_path,
        test_command_path,
        test_return_code_path,
        test_log_path,
        trace_path,
        capture_result.capture_report_path,
        capture_result.dependency_report_path,
        prompt_paths["source_path"],
        prompt_paths["decoded_path"],
        prompt_paths["metadata_path"],
        DEFAULT_STAGE2A_RESULTS_PATH,
    ]
    manifest = _package_review(
        output_dir=output_dir,
        review_zip_path=review_zip_path,
        files_to_copy=files_to_copy,
        manifest_output_path=manifest_path,
    )
    stage2b_results["review_package_size_bytes"] = manifest["review_package_size_bytes"]
    stage2b_results["source_snapshot_sha256"] = manifest["source_snapshot_sha256"]
    report_path.write_text(_render_report(stage2b_results), encoding="utf-8")
    _write_json(results_path, stage2b_results)
    _package_review(
        output_dir=output_dir,
        review_zip_path=review_zip_path,
        files_to_copy=files_to_copy,
        manifest_output_path=manifest_path,
    )
    _verify_packaged_outputs(review_zip_path=review_zip_path)

    print(
        json.dumps(
            {
                "results_path": str(results_path),
                "report_path": str(report_path),
                "review_zip_path": str(review_zip_path),
                "trace_path": str(trace_path),
                "trace_sha256": _sha256_file(trace_path),
                "source_snapshot_sha256": stage2b_results["source_snapshot_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Stage2BExecutionError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(error.exit_code)
