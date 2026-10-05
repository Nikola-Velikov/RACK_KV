from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import gmpy2
import numpy as np
import torch

from .certificate import exact_reference_output_mpfr
from .codec import CompressedBlock, SerializedBlockContainer, deserialize_block_container
from .stage2 import (
    PrefixCompressionResult,
    QueryCaseResult,
    ValidatedCompactTrace,
    _build_prefix_result,
    _build_query_case_base,
    _run_query_case_tolerance_bundle,
    validate_compact_trace,
)


STAGE4_METHOD_VERSION = "stage4_baselines_v2"
STAGE4_RESULT_VERSION = "stage4_results_v2"
STAGE4_CAPTURE_REPORT_FILENAME = "capture_report.json"
STAGE4_RECENT_WINDOW = 16
STAGE4_BLOCK_SIZE = 8
STAGE4_RACK_TOLERANCE = 0.05
STAGE4_PRECISION = 256
STAGE4_SEED = 0
STAGE4_SNAP_KEEP_FRACTION = 0.25
STAGE4_QUEST_KEEP_FRACTION = 0.25
STAGE4_BUDGET_MATCHING_POLICY_VERSION = "nearest_serialized_bytes_v2"


@dataclass(frozen=True)
class BaselineDefinition:
    name: str
    display_name: str
    category: str
    citation_key: str | None
    fidelity_status: str
    source_label: str
    deviations: tuple[str, ...]
    budget_tunable: bool


@dataclass(frozen=True)
class ByteBreakdown:
    encoded_key_bytes: int
    encoded_value_bytes: int
    scales_bytes: int
    metadata_bytes: int
    indices_bytes: int
    block_page_metadata_bytes: int
    recent_window_bytes: int
    total_serialized_bytes: int


@dataclass(frozen=True)
class SerializedBaselineArtifact:
    definition: BaselineDefinition
    mode: str
    payload: bytes
    payload_sha256: str
    total_serialized_bytes: int
    byte_breakdown: ByteBreakdown
    parameters: dict[str, Any]
    budget_tunable: bool
    budget_match_attempted: bool
    budget_target_bytes: int | None
    budget_abs_diff_bytes: int | None
    budget_rel_diff_fraction: float | None
    budget_match_possible: bool | None


@dataclass(frozen=True)
class Stage4CaseContext:
    trace_path: Path
    trace_sha256: str
    layer_index: int
    record_index: int
    query_local_index: int
    query_head_global: int
    kv_head_global: int
    query_position: int
    visible_length: int
    historical_tokens: int
    recent_exact_tokens: int
    head_dim: int
    attention_scale: float
    selected_query_heads: tuple[int, ...]
    selected_kv_heads: tuple[int, ...]
    query_to_kv_heads: tuple[int, ...]
    query_tensor: torch.Tensor
    prefix_keys_tensor: torch.Tensor
    prefix_values_tensor: torch.Tensor
    query: np.ndarray
    prefix_keys: np.ndarray
    prefix_values: np.ndarray
    captured_model_output: np.ndarray
    original_reference_output_mpfr: tuple[gmpy2.mpfr, ...]
    original_reference_output: np.ndarray
    full_kv_reference_bytes: int


@dataclass(frozen=True)
class DecodedBaselineState:
    attention_keys: np.ndarray
    attention_values: np.ndarray
    reconstructed_full_keys: np.ndarray | None
    reconstructed_full_values: np.ndarray | None
    retained_token_fraction: float
    retained_block_fraction: float | None
    key_max_abs_error: float | None
    key_rmse: float | None
    value_max_abs_error: float | None
    value_rmse: float | None
    retained_old_token_count: int
    retained_old_block_count: int | None
    total_old_block_count: int | None
    extra: dict[str, Any]


@dataclass(frozen=True)
class BaselineCaseResult:
    case_key: str
    method_name: str
    mode: str
    method_version: str
    category: str
    citation_key: str | None
    fidelity_status: str
    source_label: str
    deviations: tuple[str, ...]
    layer_index: int
    record_index: int
    query_position: int
    query_local_index: int
    query_head_global: int
    kv_head_global: int
    visible_length: int
    historical_tokens: int
    recent_exact_tokens: int
    attention_scale: float
    full_kv_reference_bytes: int
    payload_sha256: str
    total_serialized_bytes: int
    encoded_key_bytes: int
    encoded_value_bytes: int
    scales_bytes: int
    metadata_bytes: int
    indices_bytes: int
    block_page_metadata_bytes: int
    recent_window_bytes: int
    compression_ratio_vs_full_kv: float
    memory_saving_fraction_vs_full_kv: float
    budget_tunable: bool
    budget_match_attempted: bool
    budget_target_bytes: int | None
    budget_abs_diff_bytes: int | None
    budget_rel_diff_fraction: float | None
    budget_match_possible: bool | None
    budget_within_two_percent: bool | None
    retained_token_fraction: float
    retained_block_fraction: float | None
    attention_output_l2_error: float
    relative_l2_error: float
    max_abs_component_error: float
    cosine_similarity: float
    model_reference_gap: float
    captured_model_total_gap: float
    key_max_abs_error: float | None
    key_rmse: float | None
    value_max_abs_error: float | None
    value_rmse: float | None
    candidate_blocks: int | None
    certified_skipped_blocks: int | None
    certificate_upper_bound: float | None
    observed_skipping_error: float | None
    decoded_blocks: int | None
    false_safe_count: int | None
    rigorous_interval_violations: int | None
    approximate_observed_violations: int | None
    decomposition_violations: int | None
    reference_decomposition_violations: int | None
    model_relative_decomposition_violations: int | None
    numerical_fallbacks: int | None
    payload_relative_path: str | None
    parameters: dict[str, Any]
    extra: dict[str, Any]


BASELINE_DEFINITIONS: dict[str, BaselineDefinition] = {
    "full_kv": BaselineDefinition(
        name="full_kv",
        display_name="Full KV",
        category="reference",
        citation_key=None,
        fidelity_status="reference",
        source_label="Internal exact reconstructed/full-cache reference",
        deviations=tuple(),
        budget_tunable=False,
    ),
    "rack_kv": BaselineDefinition(
        name="rack_kv",
        display_name="RACK-KV",
        category="compression+random_access+certified_skipping",
        citation_key=None,
        fidelity_status="internal",
        source_label="Internal RACK-KV implementation",
        deviations=tuple(),
        budget_tunable=False,
    ),
    "uniform_int8_kv": BaselineDefinition(
        name="uniform_int8_kv",
        display_name="Uniform INT8 KV",
        category="quantization_control",
        citation_key=None,
        fidelity_status="control",
        source_label="Internal transparent control baseline",
        deviations=(
            "Per-token symmetric INT8 quantization for old keys and values with recent window kept exact.",
        ),
        budget_tunable=False,
    ),
    "kivi_style": BaselineDefinition(
        name="kivi_style",
        display_name="KIVI-style",
        category="quantization",
        citation_key="liu2024kivi",
        fidelity_status="style",
        source_label="Frozen corpus citation key liu2024kivi",
        deviations=(
            "Implemented as a source-inspired approximation because the exact frozen local record did not expose a complete executable reproduction path.",
            "Uses asymmetric granularity: old keys quantized per channel across tokens and old values quantized per token across dimensions.",
            "Uses affine 2-bit packing with stored scale/zero parameters and recent window kept exact.",
        ),
        budget_tunable=False,
    ),
    "snapkv_style": BaselineDefinition(
        name="snapkv_style",
        display_name="SnapKV-style",
        category="eviction",
        citation_key="li2024snapkv",
        fidelity_status="style",
        source_label="Frozen corpus citation key li2024snapkv",
        deviations=(
            "Implemented as a causal top-k old-token retention approximation because the exact frozen local record did not expose a complete executable reproduction path.",
            "Uses current-query old-token attention scores only and preserves the recent exact window.",
            "Native configuration keeps a fixed 25% of historical tokens, rounded up to at least one token when history exists.",
        ),
        budget_tunable=True,
    ),
    "quest_style": BaselineDefinition(
        name="quest_style",
        display_name="Quest-style",
        category="query_aware_page_retrieval",
        citation_key="tang2024quest",
        fidelity_status="style",
        source_label="Frozen corpus citation key tang2024quest",
        deviations=(
            "Implemented as a causal top-k historical block retention approximation because the exact frozen local record did not expose a complete executable reproduction path.",
            "Uses block size 8 and scores each historical block by the maximum exact query-key score inside the visible block.",
            "Native configuration keeps a fixed 25% of historical blocks, rounded up to at least one block when history exists.",
        ),
        budget_tunable=True,
    ),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _float64_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float64).numpy()


def _mpfr_vector_to_numpy(values: Sequence[gmpy2.mpfr]) -> np.ndarray:
    return np.asarray([float(component) for component in values], dtype=np.float64)


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64), ord=2))


def _rmse(diff: np.ndarray) -> float:
    array = np.asarray(diff, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64), dtype=np.float64)))


def _relative_l2_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    numerator = _norm(reference - candidate)
    denominator = _norm(reference)
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return float(numerator / denominator)


def _cosine_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    ref_norm = _norm(ref)
    cand_norm = _norm(cand)
    if ref_norm == 0.0 or cand_norm == 0.0:
        return 1.0 if ref_norm == cand_norm else 0.0
    return float(np.dot(ref, cand) / (ref_norm * cand_norm))


def _max_abs_component_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(reference, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)), initial=0.0))


def _section_header_payload(
    *,
    method_name: str,
    parameters: Mapping[str, Any],
    sections: Sequence[tuple[str, bytes, dict[str, Any]]],
) -> tuple[bytes, int]:
    offset = 0
    section_descriptors: list[dict[str, Any]] = []
    payload_parts: list[bytes] = []
    for name, payload, descriptor in sections:
        section_descriptors.append(
            {
                "name": name,
                "offset": offset,
                "length": len(payload),
                **descriptor,
            }
        )
        payload_parts.append(payload)
        offset += len(payload)
    header = {
        "format": "rack_kv_stage4_payload_v1",
        "method_name": method_name,
        "method_version": STAGE4_METHOD_VERSION,
        "parameters": dict(parameters),
        "sections": section_descriptors,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    wrapper = struct.pack("<I", len(header_bytes))
    return wrapper + header_bytes + b"".join(payload_parts), len(wrapper) + len(header_bytes)


def _parse_payload(payload: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    if len(payload) < 4:
        raise ValueError("Serialized payload is truncated.")
    header_len = struct.unpack("<I", payload[:4])[0]
    if header_len < 0 or 4 + header_len > len(payload):
        raise ValueError("Serialized payload header length is invalid.")
    header = json.loads(payload[4 : 4 + header_len].decode("utf-8"))
    body = payload[4 + header_len :]
    sections: dict[str, bytes] = {}
    for descriptor in header["sections"]:
        offset = int(descriptor["offset"])
        length = int(descriptor["length"])
        if offset < 0 or length < 0 or offset + length > len(body):
            raise ValueError("Serialized payload section range is invalid.")
        sections[str(descriptor["name"])] = body[offset : offset + length]
    return header, sections


def _bf16_tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    bf16 = tensor.detach().cpu().contiguous().to(torch.bfloat16)
    return bf16.view(torch.uint16).numpy().tobytes(order="C")


def _bf16_bytes_to_numpy(payload: bytes, *, shape: Sequence[int]) -> np.ndarray:
    raw = np.frombuffer(payload, dtype=np.uint16).copy().reshape(tuple(int(dim) for dim in shape))
    tensor = torch.from_numpy(raw).view(torch.bfloat16)
    return tensor.to(torch.float64).numpy()


def _np_section_bytes(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array).tobytes(order="C")


def _np_from_section(payload: bytes, *, dtype: np.dtype[Any], shape: Sequence[int]) -> np.ndarray:
    return np.frombuffer(payload, dtype=dtype).copy().reshape(tuple(int(dim) for dim in shape))


def _pack_uint2(values: np.ndarray) -> bytes:
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    if flat.size == 0:
        return b""
    if np.any(flat > 3):
        raise ValueError("uint2 packing values must stay within [0, 3].")
    packed = bytearray((flat.size + 3) // 4)
    for index, value in enumerate(flat.tolist()):
        packed[index // 4] |= int(value) << ((index % 4) * 2)
    return bytes(packed)


def _unpack_uint2(payload: bytes, *, shape: Sequence[int]) -> np.ndarray:
    total = math.prod(int(dim) for dim in shape)
    if total == 0:
        return np.zeros(tuple(int(dim) for dim in shape), dtype=np.uint8)
    unpacked = np.zeros((total,), dtype=np.uint8)
    for index in range(total):
        unpacked[index] = (payload[index // 4] >> ((index % 4) * 2)) & 0x03
    return unpacked.reshape(tuple(int(dim) for dim in shape))


def _block_slices(token_count: int, block_size: int) -> tuple[tuple[int, int], ...]:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if token_count < 0:
        raise ValueError("token_count must be nonnegative.")
    return tuple((start, min(start + block_size, token_count)) for start in range(0, token_count, block_size))


def validate_stage4_capture_inputs(
    *,
    capture_dir: str | Path,
    layer_indices: Sequence[int],
) -> tuple[dict[int, ValidatedCompactTrace], dict[str, Any]]:
    capture_root = Path(capture_dir)
    report_path = capture_root / STAGE4_CAPTURE_REPORT_FILENAME
    if not report_path.exists():
        raise FileNotFoundError(f"Stage 3 capture report does not exist: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    layer_entries = {
        int(entry["layer_index"]): entry
        for entry in report.get("layers", [])
    }
    traces: dict[int, ValidatedCompactTrace] = {}
    for layer_index in layer_indices:
        if layer_index not in layer_entries:
            raise ValueError(f"Capture report does not contain layer {layer_index}.")
        entry = layer_entries[layer_index]
        trace_path = Path(entry["trace_path"])
        if not trace_path.exists():
            raise FileNotFoundError(f"Expected frozen Stage 3 trace does not exist: {trace_path}")
        actual_sha = _sha256_file(trace_path)
        expected_sha = str(entry["trace_sha256"])
        if actual_sha != expected_sha:
            raise ValueError(
                f"Stage 3 trace SHA-256 mismatch for layer {layer_index}: expected {expected_sha}, got {actual_sha}."
            )
        trace = validate_compact_trace(trace_path, allow_nonzero_layer=True)
        if trace.layer_index != layer_index:
            raise ValueError(f"Trace {trace_path} saved layer {trace.layer_index}, expected {layer_index}.")
        traces[layer_index] = trace
    return traces, report


def build_stage4_case_context(
    *,
    trace: ValidatedCompactTrace,
    record_index: int,
    query_local_index: int,
    recent_window: int = STAGE4_RECENT_WINDOW,
    precision: int = STAGE4_PRECISION,
) -> Stage4CaseContext:
    if record_index < 0 or record_index >= trace.query_records:
        raise ValueError("record_index is out of range for the trace.")
    if query_local_index < 0 or query_local_index >= trace.query_head_count:
        raise ValueError("query_local_index is out of range for the trace.")
    prefix_keys_tensor, prefix_values_tensor = trace.query_case_prefix(
        record_index=record_index,
        query_local_index=query_local_index,
    )
    visible_length = int(prefix_keys_tensor.shape[0])
    recent_exact_tokens = min(recent_window, visible_length)
    historical_tokens = max(visible_length - recent_exact_tokens, 0)
    query_tensor = trace.queries[record_index, query_local_index, :].detach().cpu().clone()
    query = _float64_numpy(query_tensor)
    prefix_keys = _float64_numpy(prefix_keys_tensor)
    prefix_values = _float64_numpy(prefix_values_tensor)
    original_reference_output_mpfr = tuple(
        exact_reference_output_mpfr(
            query,
            prefix_keys,
            prefix_values,
            precision=precision,
            attention_scale=trace.scaling,
        )
    )
    original_reference_output = _mpfr_vector_to_numpy(original_reference_output_mpfr)
    captured_model_output = _float64_numpy(trace.model_head_outputs[record_index, query_local_index, :])
    return Stage4CaseContext(
        trace_path=trace.trace_path,
        trace_sha256=trace.trace_sha256,
        layer_index=trace.layer_index,
        record_index=record_index,
        query_local_index=query_local_index,
        query_head_global=trace.selected_query_heads[query_local_index],
        kv_head_global=trace.query_to_kv_heads[query_local_index],
        query_position=trace.query_positions[record_index],
        visible_length=visible_length,
        historical_tokens=historical_tokens,
        recent_exact_tokens=recent_exact_tokens,
        head_dim=trace.head_dim,
        attention_scale=trace.scaling,
        selected_query_heads=trace.selected_query_heads,
        selected_kv_heads=trace.selected_kv_heads,
        query_to_kv_heads=trace.query_to_kv_heads,
        query_tensor=query_tensor,
        prefix_keys_tensor=prefix_keys_tensor,
        prefix_values_tensor=prefix_values_tensor,
        query=query,
        prefix_keys=prefix_keys,
        prefix_values=prefix_values,
        captured_model_output=captured_model_output,
        original_reference_output_mpfr=original_reference_output_mpfr,
        original_reference_output=original_reference_output,
        full_kv_reference_bytes=_tensor_bytes(prefix_keys_tensor) + _tensor_bytes(prefix_values_tensor),
    )


def _recent_exact_tensors(context: Stage4CaseContext) -> tuple[torch.Tensor, torch.Tensor]:
    if context.historical_tokens <= 0:
        return context.prefix_keys_tensor, context.prefix_values_tensor
    return (
        context.prefix_keys_tensor[context.historical_tokens :, :],
        context.prefix_values_tensor[context.historical_tokens :, :],
    )


def _history_float64(context: Stage4CaseContext) -> tuple[np.ndarray, np.ndarray]:
    if context.historical_tokens <= 0:
        return (
            np.zeros((0, context.head_dim), dtype=np.float64),
            np.zeros((0, context.prefix_values.shape[1]), dtype=np.float64),
        )
    return (
        context.prefix_keys[: context.historical_tokens, :],
        context.prefix_values[: context.historical_tokens, :],
    )


def _old_exact_tensors(context: Stage4CaseContext) -> tuple[torch.Tensor, torch.Tensor]:
    if context.historical_tokens <= 0:
        return (
            context.prefix_keys_tensor[:0, :],
            context.prefix_values_tensor[:0, :],
        )
    return (
        context.prefix_keys_tensor[: context.historical_tokens, :],
        context.prefix_values_tensor[: context.historical_tokens, :],
    )


def _build_case_key(context: Stage4CaseContext) -> str:
    return (
        f"layer{context.layer_index}_pos{context.query_position}_"
        f"record{context.record_index}_ql{context.query_local_index}_"
        f"qh{context.query_head_global}_kv{context.kv_head_global}"
    )


def _serialize_full_kv(context: Stage4CaseContext, *, mode: str) -> SerializedBaselineArtifact:
    key_payload = _bf16_tensor_to_bytes(context.prefix_keys_tensor)
    value_payload = _bf16_tensor_to_bytes(context.prefix_values_tensor)
    parameters = {
        "mode": mode,
        "representation": "exact_full_kv_bfloat16",
        "visible_length": context.visible_length,
        "head_dim": context.head_dim,
        "value_dim": int(context.prefix_values_tensor.shape[1]),
    }
    payload, header_bytes = _section_header_payload(
        method_name="full_kv",
        parameters=parameters,
        sections=(
            (
                "keys_bf16",
                key_payload,
                {"encoding": "bf16_tensor", "shape": list(context.prefix_keys_tensor.shape)},
            ),
            (
                "values_bf16",
                value_payload,
                {"encoding": "bf16_tensor", "shape": list(context.prefix_values_tensor.shape)},
            ),
        ),
    )
    breakdown = ByteBreakdown(
        encoded_key_bytes=len(key_payload),
        encoded_value_bytes=len(value_payload),
        scales_bytes=0,
        metadata_bytes=header_bytes,
        indices_bytes=0,
        block_page_metadata_bytes=0,
        recent_window_bytes=0,
        total_serialized_bytes=len(payload),
    )
    return SerializedBaselineArtifact(
        definition=BASELINE_DEFINITIONS["full_kv"],
        mode=mode,
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        total_serialized_bytes=len(payload),
        byte_breakdown=breakdown,
        parameters=parameters,
        budget_tunable=BASELINE_DEFINITIONS["full_kv"].budget_tunable,
        budget_match_attempted=False,
        budget_target_bytes=None,
        budget_abs_diff_bytes=None,
        budget_rel_diff_fraction=None,
        budget_match_possible=None,
    )


def _decode_full_kv(artifact: SerializedBaselineArtifact, context: Stage4CaseContext) -> DecodedBaselineState:
    header, sections = _parse_payload(artifact.payload)
    key_shape = next(section["shape"] for section in header["sections"] if section["name"] == "keys_bf16")
    value_shape = next(section["shape"] for section in header["sections"] if section["name"] == "values_bf16")
    keys = _bf16_bytes_to_numpy(sections["keys_bf16"], shape=key_shape)
    values = _bf16_bytes_to_numpy(sections["values_bf16"], shape=value_shape)
    key_diff = keys - context.prefix_keys
    value_diff = values - context.prefix_values
    return DecodedBaselineState(
        attention_keys=keys,
        attention_values=values,
        reconstructed_full_keys=keys,
        reconstructed_full_values=values,
        retained_token_fraction=1.0,
        retained_block_fraction=1.0 if context.historical_tokens > 0 else None,
        key_max_abs_error=float(np.max(np.abs(key_diff), initial=0.0)),
        key_rmse=_rmse(key_diff),
        value_max_abs_error=float(np.max(np.abs(value_diff), initial=0.0)),
        value_rmse=_rmse(value_diff),
        retained_old_token_count=context.historical_tokens,
        retained_old_block_count=len(_block_slices(context.historical_tokens, STAGE4_BLOCK_SIZE)) if context.historical_tokens > 0 else 0,
        total_old_block_count=len(_block_slices(context.historical_tokens, STAGE4_BLOCK_SIZE)) if context.historical_tokens > 0 else 0,
        extra={},
    )


def _symmetric_int8_quantize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float16), np.zeros_like(matrix, dtype=np.int8)
    max_abs = np.max(np.abs(matrix), axis=1)
    max_abs = np.where(max_abs == 0.0, 1.0, max_abs)
    scales = np.asarray(max_abs / 127.0, dtype=np.float16)
    q = np.clip(np.rint(matrix / scales.astype(np.float64)[:, None]), -127, 127).astype(np.int8)
    return scales, q


def _serialize_uniform_int8_kv(context: Stage4CaseContext, *, mode: str) -> SerializedBaselineArtifact:
    old_keys, old_values = _history_float64(context)
    recent_keys_t, recent_values_t = _recent_exact_tensors(context)
    old_key_scales, old_key_q = _symmetric_int8_quantize_rows(old_keys)
    old_value_scales, old_value_q = _symmetric_int8_quantize_rows(old_values)
    recent_key_payload = _bf16_tensor_to_bytes(recent_keys_t)
    recent_value_payload = _bf16_tensor_to_bytes(recent_values_t)
    sections = [
        ("recent_keys_bf16", recent_key_payload, {"encoding": "bf16_tensor", "shape": list(recent_keys_t.shape)}),
        ("recent_values_bf16", recent_value_payload, {"encoding": "bf16_tensor", "shape": list(recent_values_t.shape)}),
        ("old_key_scales_f16", _np_section_bytes(old_key_scales), {"encoding": "ndarray", "dtype": "float16", "shape": list(old_key_scales.shape)}),
        ("old_value_scales_f16", _np_section_bytes(old_value_scales), {"encoding": "ndarray", "dtype": "float16", "shape": list(old_value_scales.shape)}),
        ("old_keys_q_i8", _np_section_bytes(old_key_q), {"encoding": "ndarray", "dtype": "int8", "shape": list(old_key_q.shape)}),
        ("old_values_q_i8", _np_section_bytes(old_value_q), {"encoding": "ndarray", "dtype": "int8", "shape": list(old_value_q.shape)}),
    ]
    parameters = {
        "mode": mode,
        "recent_window": context.recent_exact_tokens,
        "historical_tokens": context.historical_tokens,
        "quantization": "per_token_symmetric_int8",
    }
    payload, header_bytes = _section_header_payload(
        method_name="uniform_int8_kv",
        parameters=parameters,
        sections=tuple(sections),
    )
    breakdown = ByteBreakdown(
        encoded_key_bytes=len(_np_section_bytes(old_key_q)),
        encoded_value_bytes=len(_np_section_bytes(old_value_q)),
        scales_bytes=len(_np_section_bytes(old_key_scales)) + len(_np_section_bytes(old_value_scales)),
        metadata_bytes=header_bytes,
        indices_bytes=0,
        block_page_metadata_bytes=0,
        recent_window_bytes=len(recent_key_payload) + len(recent_value_payload),
        total_serialized_bytes=len(payload),
    )
    return SerializedBaselineArtifact(
        definition=BASELINE_DEFINITIONS["uniform_int8_kv"],
        mode=mode,
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        total_serialized_bytes=len(payload),
        byte_breakdown=breakdown,
        parameters=parameters,
        budget_tunable=BASELINE_DEFINITIONS["uniform_int8_kv"].budget_tunable,
        budget_match_attempted=False,
        budget_target_bytes=None,
        budget_abs_diff_bytes=None,
        budget_rel_diff_fraction=None,
        budget_match_possible=None,
    )


def _decode_uniform_int8_kv(artifact: SerializedBaselineArtifact, context: Stage4CaseContext) -> DecodedBaselineState:
    header, sections = _parse_payload(artifact.payload)
    shapes = {str(section["name"]): section["shape"] for section in header["sections"]}
    recent_keys = _bf16_bytes_to_numpy(sections["recent_keys_bf16"], shape=shapes["recent_keys_bf16"])
    recent_values = _bf16_bytes_to_numpy(sections["recent_values_bf16"], shape=shapes["recent_values_bf16"])
    old_key_scales = _np_from_section(sections["old_key_scales_f16"], dtype=np.float16, shape=shapes["old_key_scales_f16"]).astype(np.float64)
    old_value_scales = _np_from_section(sections["old_value_scales_f16"], dtype=np.float16, shape=shapes["old_value_scales_f16"]).astype(np.float64)
    old_key_q = _np_from_section(sections["old_keys_q_i8"], dtype=np.int8, shape=shapes["old_keys_q_i8"]).astype(np.float64)
    old_value_q = _np_from_section(sections["old_values_q_i8"], dtype=np.int8, shape=shapes["old_values_q_i8"]).astype(np.float64)
    old_keys = old_key_q * old_key_scales[:, None] if old_key_q.size else np.zeros((0, context.head_dim), dtype=np.float64)
    old_values = old_value_q * old_value_scales[:, None] if old_value_q.size else np.zeros((0, recent_values.shape[1]), dtype=np.float64)
    reconstructed_keys = np.vstack([old_keys, recent_keys])
    reconstructed_values = np.vstack([old_values, recent_values])
    key_diff = reconstructed_keys - context.prefix_keys
    value_diff = reconstructed_values - context.prefix_values
    total_old_blocks = len(_block_slices(context.historical_tokens, STAGE4_BLOCK_SIZE))
    return DecodedBaselineState(
        attention_keys=reconstructed_keys,
        attention_values=reconstructed_values,
        reconstructed_full_keys=reconstructed_keys,
        reconstructed_full_values=reconstructed_values,
        retained_token_fraction=1.0,
        retained_block_fraction=1.0 if total_old_blocks > 0 else None,
        key_max_abs_error=float(np.max(np.abs(key_diff), initial=0.0)),
        key_rmse=_rmse(key_diff),
        value_max_abs_error=float(np.max(np.abs(value_diff), initial=0.0)),
        value_rmse=_rmse(value_diff),
        retained_old_token_count=context.historical_tokens,
        retained_old_block_count=total_old_blocks,
        total_old_block_count=total_old_blocks,
        extra={},
    )


def _affine_uint2_quantize_per_channel(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.size == 0:
        dim = matrix.shape[1] if matrix.ndim == 2 else 0
        return (
            np.zeros((dim,), dtype=np.float16),
            np.zeros((dim,), dtype=np.float16),
            np.zeros_like(matrix, dtype=np.uint8),
        )
    mins = np.min(matrix, axis=0)
    maxs = np.max(matrix, axis=0)
    scales = np.where(maxs > mins, (maxs - mins) / 3.0, 1.0)
    zeros = mins
    q = np.clip(np.rint((matrix - zeros[None, :]) / scales[None, :]), 0, 3).astype(np.uint8)
    return np.asarray(scales, dtype=np.float16), np.asarray(zeros, dtype=np.float16), q


def _affine_uint2_quantize_per_row(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.size == 0:
        rows = matrix.shape[0] if matrix.ndim == 2 else 0
        return (
            np.zeros((rows,), dtype=np.float16),
            np.zeros((rows,), dtype=np.float16),
            np.zeros_like(matrix, dtype=np.uint8),
        )
    mins = np.min(matrix, axis=1)
    maxs = np.max(matrix, axis=1)
    scales = np.where(maxs > mins, (maxs - mins) / 3.0, 1.0)
    zeros = mins
    q = np.clip(np.rint((matrix - zeros[:, None]) / scales[:, None]), 0, 3).astype(np.uint8)
    return np.asarray(scales, dtype=np.float16), np.asarray(zeros, dtype=np.float16), q


def _serialize_kivi_style(context: Stage4CaseContext, *, mode: str) -> SerializedBaselineArtifact:
    old_keys, old_values = _history_float64(context)
    recent_keys_t, recent_values_t = _recent_exact_tensors(context)
    key_scales, key_zeros, key_q = _affine_uint2_quantize_per_channel(old_keys)
    value_scales, value_zeros, value_q = _affine_uint2_quantize_per_row(old_values)
    recent_key_payload = _bf16_tensor_to_bytes(recent_keys_t)
    recent_value_payload = _bf16_tensor_to_bytes(recent_values_t)
    key_q_payload = _pack_uint2(key_q)
    value_q_payload = _pack_uint2(value_q)
    sections = [
        ("recent_keys_bf16", recent_key_payload, {"encoding": "bf16_tensor", "shape": list(recent_keys_t.shape)}),
        ("recent_values_bf16", recent_value_payload, {"encoding": "bf16_tensor", "shape": list(recent_values_t.shape)}),
        ("old_key_scales_f16", _np_section_bytes(key_scales), {"encoding": "ndarray", "dtype": "float16", "shape": list(key_scales.shape)}),
        ("old_key_zeros_f16", _np_section_bytes(key_zeros), {"encoding": "ndarray", "dtype": "float16", "shape": list(key_zeros.shape)}),
        ("old_value_scales_f16", _np_section_bytes(value_scales), {"encoding": "ndarray", "dtype": "float16", "shape": list(value_scales.shape)}),
        ("old_value_zeros_f16", _np_section_bytes(value_zeros), {"encoding": "ndarray", "dtype": "float16", "shape": list(value_zeros.shape)}),
        ("old_keys_q2", key_q_payload, {"encoding": "packed_uint2", "shape": list(key_q.shape)}),
        ("old_values_q2", value_q_payload, {"encoding": "packed_uint2", "shape": list(value_q.shape)}),
    ]
    parameters = {
        "mode": mode,
        "recent_window": context.recent_exact_tokens,
        "historical_tokens": context.historical_tokens,
        "quantization": "kivi_style_affine_uint2",
    }
    payload, header_bytes = _section_header_payload(
        method_name="kivi_style",
        parameters=parameters,
        sections=tuple(sections),
    )
    quant_param_bytes = (
        len(_np_section_bytes(key_scales))
        + len(_np_section_bytes(key_zeros))
        + len(_np_section_bytes(value_scales))
        + len(_np_section_bytes(value_zeros))
    )
    breakdown = ByteBreakdown(
        encoded_key_bytes=len(key_q_payload),
        encoded_value_bytes=len(value_q_payload),
        scales_bytes=quant_param_bytes,
        metadata_bytes=header_bytes,
        indices_bytes=0,
        block_page_metadata_bytes=0,
        recent_window_bytes=len(recent_key_payload) + len(recent_value_payload),
        total_serialized_bytes=len(payload),
    )
    return SerializedBaselineArtifact(
        definition=BASELINE_DEFINITIONS["kivi_style"],
        mode=mode,
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        total_serialized_bytes=len(payload),
        byte_breakdown=breakdown,
        parameters=parameters,
        budget_tunable=BASELINE_DEFINITIONS["kivi_style"].budget_tunable,
        budget_match_attempted=False,
        budget_target_bytes=None,
        budget_abs_diff_bytes=None,
        budget_rel_diff_fraction=None,
        budget_match_possible=None,
    )


def _decode_kivi_style(artifact: SerializedBaselineArtifact, context: Stage4CaseContext) -> DecodedBaselineState:
    header, sections = _parse_payload(artifact.payload)
    shapes = {str(section["name"]): section["shape"] for section in header["sections"]}
    recent_keys = _bf16_bytes_to_numpy(sections["recent_keys_bf16"], shape=shapes["recent_keys_bf16"])
    recent_values = _bf16_bytes_to_numpy(sections["recent_values_bf16"], shape=shapes["recent_values_bf16"])
    key_scales = _np_from_section(sections["old_key_scales_f16"], dtype=np.float16, shape=shapes["old_key_scales_f16"]).astype(np.float64)
    key_zeros = _np_from_section(sections["old_key_zeros_f16"], dtype=np.float16, shape=shapes["old_key_zeros_f16"]).astype(np.float64)
    value_scales = _np_from_section(sections["old_value_scales_f16"], dtype=np.float16, shape=shapes["old_value_scales_f16"]).astype(np.float64)
    value_zeros = _np_from_section(sections["old_value_zeros_f16"], dtype=np.float16, shape=shapes["old_value_zeros_f16"]).astype(np.float64)
    key_q = _unpack_uint2(sections["old_keys_q2"], shape=shapes["old_keys_q2"]).astype(np.float64)
    value_q = _unpack_uint2(sections["old_values_q2"], shape=shapes["old_values_q2"]).astype(np.float64)
    old_keys = key_zeros[None, :] + key_scales[None, :] * key_q if key_q.size else np.zeros((0, context.head_dim), dtype=np.float64)
    old_values = value_zeros[:, None] + value_scales[:, None] * value_q if value_q.size else np.zeros((0, recent_values.shape[1]), dtype=np.float64)
    reconstructed_keys = np.vstack([old_keys, recent_keys])
    reconstructed_values = np.vstack([old_values, recent_values])
    key_diff = reconstructed_keys - context.prefix_keys
    value_diff = reconstructed_values - context.prefix_values
    total_old_blocks = len(_block_slices(context.historical_tokens, STAGE4_BLOCK_SIZE))
    return DecodedBaselineState(
        attention_keys=reconstructed_keys,
        attention_values=reconstructed_values,
        reconstructed_full_keys=reconstructed_keys,
        reconstructed_full_values=reconstructed_values,
        retained_token_fraction=1.0,
        retained_block_fraction=1.0 if total_old_blocks > 0 else None,
        key_max_abs_error=float(np.max(np.abs(key_diff), initial=0.0)),
        key_rmse=_rmse(key_diff),
        value_max_abs_error=float(np.max(np.abs(value_diff), initial=0.0)),
        value_rmse=_rmse(value_diff),
        retained_old_token_count=context.historical_tokens,
        retained_old_block_count=total_old_blocks,
        total_old_block_count=total_old_blocks,
        extra={},
    )


def _old_token_scores(context: Stage4CaseContext) -> np.ndarray:
    old_keys, _ = _history_float64(context)
    if old_keys.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return old_keys @ context.query * context.attention_scale


def _serialize_snapkv_style(
    context: Stage4CaseContext,
    *,
    mode: str,
    keep_count: int,
    budget_target_bytes: int | None,
) -> SerializedBaselineArtifact:
    old_keys_t, old_values_t = _old_exact_tensors(context)
    recent_keys_t, recent_values_t = _recent_exact_tensors(context)
    scores = _old_token_scores(context)
    keep_count = max(0, min(int(keep_count), context.historical_tokens))
    if keep_count > 0 and scores.size > 0:
        order = np.argsort(scores)[::-1][:keep_count]
        kept_positions = np.sort(order.astype(np.uint16))
    else:
        kept_positions = np.zeros((0,), dtype=np.uint16)
    kept_keys_t = old_keys_t[kept_positions.tolist(), :] if kept_positions.size else old_keys_t[:0, :]
    kept_values_t = old_values_t[kept_positions.tolist(), :] if kept_positions.size else old_values_t[:0, :]
    kept_key_payload = _bf16_tensor_to_bytes(kept_keys_t)
    kept_value_payload = _bf16_tensor_to_bytes(kept_values_t)
    recent_key_payload = _bf16_tensor_to_bytes(recent_keys_t)
    recent_value_payload = _bf16_tensor_to_bytes(recent_values_t)
    kept_positions_payload = _np_section_bytes(kept_positions)
    parameters = {
        "mode": mode,
        "recent_window": context.recent_exact_tokens,
        "historical_tokens": context.historical_tokens,
        "keep_count": int(keep_count),
        "scoring": "current_query_exact_old_token_scores",
        "budget_matching_policy_version": STAGE4_BUDGET_MATCHING_POLICY_VERSION if budget_target_bytes is not None else None,
    }
    payload, header_bytes = _section_header_payload(
        method_name="snapkv_style",
        parameters=parameters,
        sections=(
            ("kept_positions_u16", kept_positions_payload, {"encoding": "ndarray", "dtype": "uint16", "shape": list(kept_positions.shape)}),
            ("kept_old_keys_bf16", kept_key_payload, {"encoding": "bf16_tensor", "shape": list(kept_keys_t.shape)}),
            ("kept_old_values_bf16", kept_value_payload, {"encoding": "bf16_tensor", "shape": list(kept_values_t.shape)}),
            ("recent_keys_bf16", recent_key_payload, {"encoding": "bf16_tensor", "shape": list(recent_keys_t.shape)}),
            ("recent_values_bf16", recent_value_payload, {"encoding": "bf16_tensor", "shape": list(recent_values_t.shape)}),
        ),
    )
    target = int(budget_target_bytes) if budget_target_bytes is not None else None
    abs_diff = abs(len(payload) - target) if target is not None else None
    rel_diff = (float(abs_diff / target) if target not in (None, 0) else None)
    breakdown = ByteBreakdown(
        encoded_key_bytes=len(kept_key_payload),
        encoded_value_bytes=len(kept_value_payload),
        scales_bytes=0,
        metadata_bytes=header_bytes,
        indices_bytes=len(kept_positions_payload),
        block_page_metadata_bytes=0,
        recent_window_bytes=len(recent_key_payload) + len(recent_value_payload),
        total_serialized_bytes=len(payload),
    )
    return SerializedBaselineArtifact(
        definition=BASELINE_DEFINITIONS["snapkv_style"],
        mode=mode,
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        total_serialized_bytes=len(payload),
        byte_breakdown=breakdown,
        parameters=parameters,
        budget_tunable=BASELINE_DEFINITIONS["snapkv_style"].budget_tunable,
        budget_match_attempted=target is not None,
        budget_target_bytes=target,
        budget_abs_diff_bytes=abs_diff,
        budget_rel_diff_fraction=rel_diff,
        budget_match_possible=(True if target is not None else None),
    )


def _decode_snapkv_style(artifact: SerializedBaselineArtifact, context: Stage4CaseContext) -> DecodedBaselineState:
    header, sections = _parse_payload(artifact.payload)
    shapes = {str(section["name"]): section["shape"] for section in header["sections"]}
    kept_positions = _np_from_section(sections["kept_positions_u16"], dtype=np.uint16, shape=shapes["kept_positions_u16"]).astype(np.int64)
    kept_keys = _bf16_bytes_to_numpy(sections["kept_old_keys_bf16"], shape=shapes["kept_old_keys_bf16"])
    kept_values = _bf16_bytes_to_numpy(sections["kept_old_values_bf16"], shape=shapes["kept_old_values_bf16"])
    recent_keys = _bf16_bytes_to_numpy(sections["recent_keys_bf16"], shape=shapes["recent_keys_bf16"])
    recent_values = _bf16_bytes_to_numpy(sections["recent_values_bf16"], shape=shapes["recent_values_bf16"])
    attention_keys = np.vstack([kept_keys, recent_keys])
    attention_values = np.vstack([kept_values, recent_values])
    retained_old = int(kept_positions.shape[0])
    total_blocks = len(_block_slices(context.historical_tokens, STAGE4_BLOCK_SIZE))
    return DecodedBaselineState(
        attention_keys=attention_keys,
        attention_values=attention_values,
        reconstructed_full_keys=None,
        reconstructed_full_values=None,
        retained_token_fraction=float((retained_old + context.recent_exact_tokens) / context.visible_length) if context.visible_length > 0 else 0.0,
        retained_block_fraction=None,
        key_max_abs_error=None,
        key_rmse=None,
        value_max_abs_error=None,
        value_rmse=None,
        retained_old_token_count=retained_old,
        retained_old_block_count=None,
        total_old_block_count=total_blocks,
        extra={"kept_positions": kept_positions.tolist()},
    )


def _historical_blocks(context: Stage4CaseContext, block_size: int) -> tuple[tuple[int, int], ...]:
    return _block_slices(context.historical_tokens, block_size)


def _historical_block_scores(context: Stage4CaseContext, block_size: int) -> np.ndarray:
    old_keys, _ = _history_float64(context)
    blocks = _historical_blocks(context, block_size)
    scores: list[float] = []
    for start, end in blocks:
        block = old_keys[start:end, :]
        if block.size == 0:
            scores.append(float("-inf"))
            continue
        logits = block @ context.query * context.attention_scale
        scores.append(float(np.max(logits)))
    return np.asarray(scores, dtype=np.float64)


def _serialize_quest_style(
    context: Stage4CaseContext,
    *,
    mode: str,
    block_size: int,
    keep_block_count: int,
    budget_target_bytes: int | None,
) -> SerializedBaselineArtifact:
    old_keys_t, old_values_t = _old_exact_tensors(context)
    recent_keys_t, recent_values_t = _recent_exact_tensors(context)
    blocks = _historical_blocks(context, block_size)
    scores = _historical_block_scores(context, block_size)
    keep_block_count = max(0, min(int(keep_block_count), len(blocks)))
    if keep_block_count > 0 and scores.size > 0:
        selected = np.argsort(scores)[::-1][:keep_block_count]
        selected = np.sort(selected.astype(np.int64))
    else:
        selected = np.zeros((0,), dtype=np.int64)
    kept_starts: list[int] = []
    kept_lengths: list[int] = []
    kept_key_tensors: list[torch.Tensor] = []
    kept_value_tensors: list[torch.Tensor] = []
    for block_index in selected.tolist():
        start, end = blocks[block_index]
        kept_starts.append(int(start))
        kept_lengths.append(int(end - start))
        kept_key_tensors.append(old_keys_t[start:end, :])
        kept_value_tensors.append(old_values_t[start:end, :])
    kept_keys_t = torch.cat(kept_key_tensors, dim=0) if kept_key_tensors else old_keys_t[:0, :]
    kept_values_t = torch.cat(kept_value_tensors, dim=0) if kept_value_tensors else old_values_t[:0, :]
    kept_starts_arr = np.asarray(kept_starts, dtype=np.uint16)
    kept_lengths_arr = np.asarray(kept_lengths, dtype=np.uint16)
    kept_key_payload = _bf16_tensor_to_bytes(kept_keys_t)
    kept_value_payload = _bf16_tensor_to_bytes(kept_values_t)
    recent_key_payload = _bf16_tensor_to_bytes(recent_keys_t)
    recent_value_payload = _bf16_tensor_to_bytes(recent_values_t)
    kept_starts_payload = _np_section_bytes(kept_starts_arr)
    kept_lengths_payload = _np_section_bytes(kept_lengths_arr)
    parameters = {
        "mode": mode,
        "recent_window": context.recent_exact_tokens,
        "historical_tokens": context.historical_tokens,
        "block_size": int(block_size),
        "keep_block_count": int(keep_block_count),
        "scoring": "max_exact_query_key_logit_per_block",
        "budget_matching_policy_version": STAGE4_BUDGET_MATCHING_POLICY_VERSION if budget_target_bytes is not None else None,
    }
    payload, header_bytes = _section_header_payload(
        method_name="quest_style",
        parameters=parameters,
        sections=(
            ("kept_block_starts_u16", kept_starts_payload, {"encoding": "ndarray", "dtype": "uint16", "shape": list(kept_starts_arr.shape)}),
            ("kept_block_lengths_u16", kept_lengths_payload, {"encoding": "ndarray", "dtype": "uint16", "shape": list(kept_lengths_arr.shape)}),
            ("kept_block_keys_bf16", kept_key_payload, {"encoding": "bf16_tensor", "shape": list(kept_keys_t.shape)}),
            ("kept_block_values_bf16", kept_value_payload, {"encoding": "bf16_tensor", "shape": list(kept_values_t.shape)}),
            ("recent_keys_bf16", recent_key_payload, {"encoding": "bf16_tensor", "shape": list(recent_keys_t.shape)}),
            ("recent_values_bf16", recent_value_payload, {"encoding": "bf16_tensor", "shape": list(recent_values_t.shape)}),
        ),
    )
    target = int(budget_target_bytes) if budget_target_bytes is not None else None
    abs_diff = abs(len(payload) - target) if target is not None else None
    rel_diff = (float(abs_diff / target) if target not in (None, 0) else None)
    breakdown = ByteBreakdown(
        encoded_key_bytes=len(kept_key_payload),
        encoded_value_bytes=len(kept_value_payload),
        scales_bytes=0,
        metadata_bytes=header_bytes,
        indices_bytes=0,
        block_page_metadata_bytes=len(kept_starts_payload) + len(kept_lengths_payload),
        recent_window_bytes=len(recent_key_payload) + len(recent_value_payload),
        total_serialized_bytes=len(payload),
    )
    return SerializedBaselineArtifact(
        definition=BASELINE_DEFINITIONS["quest_style"],
        mode=mode,
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        total_serialized_bytes=len(payload),
        byte_breakdown=breakdown,
        parameters=parameters,
        budget_tunable=BASELINE_DEFINITIONS["quest_style"].budget_tunable,
        budget_match_attempted=target is not None,
        budget_target_bytes=target,
        budget_abs_diff_bytes=abs_diff,
        budget_rel_diff_fraction=rel_diff,
        budget_match_possible=(True if target is not None else None),
    )


def _decode_quest_style(artifact: SerializedBaselineArtifact, context: Stage4CaseContext) -> DecodedBaselineState:
    header, sections = _parse_payload(artifact.payload)
    shapes = {str(section["name"]): section["shape"] for section in header["sections"]}
    kept_starts = _np_from_section(sections["kept_block_starts_u16"], dtype=np.uint16, shape=shapes["kept_block_starts_u16"]).astype(np.int64)
    kept_lengths = _np_from_section(sections["kept_block_lengths_u16"], dtype=np.uint16, shape=shapes["kept_block_lengths_u16"]).astype(np.int64)
    kept_keys = _bf16_bytes_to_numpy(sections["kept_block_keys_bf16"], shape=shapes["kept_block_keys_bf16"])
    kept_values = _bf16_bytes_to_numpy(sections["kept_block_values_bf16"], shape=shapes["kept_block_values_bf16"])
    recent_keys = _bf16_bytes_to_numpy(sections["recent_keys_bf16"], shape=shapes["recent_keys_bf16"])
    recent_values = _bf16_bytes_to_numpy(sections["recent_values_bf16"], shape=shapes["recent_values_bf16"])
    attention_keys = np.vstack([kept_keys, recent_keys])
    attention_values = np.vstack([kept_values, recent_values])
    retained_old_tokens = int(np.sum(kept_lengths, dtype=np.int64)) if kept_lengths.size else 0
    total_blocks = len(_historical_blocks(context, STAGE4_BLOCK_SIZE))
    retained_blocks = int(kept_starts.shape[0])
    return DecodedBaselineState(
        attention_keys=attention_keys,
        attention_values=attention_values,
        reconstructed_full_keys=None,
        reconstructed_full_values=None,
        retained_token_fraction=float((retained_old_tokens + context.recent_exact_tokens) / context.visible_length) if context.visible_length > 0 else 0.0,
        retained_block_fraction=float(retained_blocks / total_blocks) if total_blocks > 0 else None,
        key_max_abs_error=None,
        key_rmse=None,
        value_max_abs_error=None,
        value_rmse=None,
        retained_old_token_count=retained_old_tokens,
        retained_old_block_count=retained_blocks,
        total_old_block_count=total_blocks,
        extra={
            "kept_block_starts": kept_starts.tolist(),
            "kept_block_lengths": kept_lengths.tolist(),
        },
    )


def _serialize_rack_kv(
    context: Stage4CaseContext,
    *,
    mode: str,
    recent_window: int,
    block_size: int,
    precision: int,
) -> tuple[SerializedBaselineArtifact, PrefixCompressionResult]:
    prefix_result = _build_prefix_result(
        record_index=context.record_index,
        kv_head_global=context.kv_head_global,
        prefix_keys=context.prefix_keys_tensor,
        prefix_values=context.prefix_values_tensor,
        recent_window=recent_window,
        block_size=block_size,
        precision=precision,
    )
    recent_keys_t, recent_values_t = _recent_exact_tensors(context)
    recent_key_payload = _bf16_tensor_to_bytes(recent_keys_t)
    recent_value_payload = _bf16_tensor_to_bytes(recent_values_t)
    container_bytes = prefix_result.serialized_container.buffer if prefix_result.serialized_container is not None else b""
    parameters = {
        "mode": mode,
        "recent_window": recent_window,
        "block_size": block_size,
        "precision": precision,
        "historical_tokens": prefix_result.historical_tokens,
        "authoritative_serialized_roundtrip_used": bool(prefix_result.authoritative_serialized_roundtrip_used),
        "certificate_mode": "rigorous_reference",
    }
    payload, wrapper_header_bytes = _section_header_payload(
        method_name="rack_kv",
        parameters=parameters,
        sections=(
            ("recent_keys_bf16", recent_key_payload, {"encoding": "bf16_tensor", "shape": list(recent_keys_t.shape)}),
            ("recent_values_bf16", recent_value_payload, {"encoding": "bf16_tensor", "shape": list(recent_values_t.shape)}),
            ("historical_block_container", container_bytes, {"encoding": "serialized_block_container", "length": len(container_bytes)}),
        ),
    )
    breakdown = ByteBreakdown(
        encoded_key_bytes=prefix_result.key_anchor_bytes + prefix_result.quantized_key_residual_bytes,
        encoded_value_bytes=prefix_result.value_anchor_bytes + prefix_result.quantized_value_residual_bytes,
        scales_bytes=prefix_result.scale_bytes,
        metadata_bytes=wrapper_header_bytes + prefix_result.container_header_bytes,
        indices_bytes=prefix_result.container_index_bytes,
        block_page_metadata_bytes=prefix_result.certificate_metadata_bytes + prefix_result.block_header_bytes + prefix_result.padding_alignment_bytes,
        recent_window_bytes=len(recent_key_payload) + len(recent_value_payload),
        total_serialized_bytes=len(payload),
    )
    artifact = SerializedBaselineArtifact(
        definition=BASELINE_DEFINITIONS["rack_kv"],
        mode=mode,
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        total_serialized_bytes=len(payload),
        byte_breakdown=breakdown,
        parameters=parameters,
        budget_tunable=BASELINE_DEFINITIONS["rack_kv"].budget_tunable,
        budget_match_attempted=False,
        budget_target_bytes=None,
        budget_abs_diff_bytes=None,
        budget_rel_diff_fraction=None,
        budget_match_possible=None,
    )
    return artifact, prefix_result


def _decode_rack_kv(
    artifact: SerializedBaselineArtifact,
    context: Stage4CaseContext,
    *,
    native_prefix_result: PrefixCompressionResult,
) -> tuple[DecodedBaselineState, PrefixCompressionResult]:
    header, sections = _parse_payload(artifact.payload)
    shapes = {str(section["name"]): section.get("shape") for section in header["sections"]}
    recent_keys = _bf16_bytes_to_numpy(sections["recent_keys_bf16"], shape=shapes["recent_keys_bf16"])
    recent_values = _bf16_bytes_to_numpy(sections["recent_values_bf16"], shape=shapes["recent_values_bf16"])
    container_bytes = sections["historical_block_container"]
    blocks: list[CompressedBlock] = []
    container: SerializedBlockContainer | None = None
    reconstructed_historical_keys: list[np.ndarray] = []
    reconstructed_historical_values: list[np.ndarray] = []
    if container_bytes:
        container = deserialize_block_container(container_bytes)
        for block_index in range(container.block_count):
            block = container.deserialize_block(block_index)
            blocks.append(block)
            block_keys, block_values = block.decode_block()
            reconstructed_historical_keys.append(block_keys)
            reconstructed_historical_values.append(block_values)
    reconstructed_keys = np.vstack([*(reconstructed_historical_keys or []), recent_keys]) if context.visible_length else np.zeros((0, context.head_dim), dtype=np.float64)
    reconstructed_values = np.vstack([*(reconstructed_historical_values or []), recent_values]) if context.visible_length else np.zeros((0, recent_values.shape[1]), dtype=np.float64)
    key_diff = reconstructed_keys - context.prefix_keys
    value_diff = reconstructed_values - context.prefix_values
    authoritative_prefix = replace(
        native_prefix_result,
        blocks=tuple(blocks),
        serialized_container=container,
        original_keys=context.prefix_keys,
        original_values=context.prefix_values,
        reconstructed_keys=reconstructed_keys,
        reconstructed_values=reconstructed_values,
        recent_keys=recent_keys,
        recent_values=recent_values,
        compressed_historical_bytes=len(container_bytes),
        recent_exact_bytes=len(sections["recent_keys_bf16"]) + len(sections["recent_values_bf16"]),
        total_compressed_bytes=artifact.total_serialized_bytes,
        bytes_per_historical_token=float(len(container_bytes) / context.historical_tokens) if context.historical_tokens > 0 else 0.0,
        compression_ratio=float(context.full_kv_reference_bytes / artifact.total_serialized_bytes),
        max_key_abs_error=float(np.max(np.abs(key_diff), initial=0.0)),
        key_rmse=_rmse(key_diff),
        max_value_abs_error=float(np.max(np.abs(value_diff), initial=0.0)),
        value_rmse=_rmse(value_diff),
    )
    total_blocks = len(blocks)
    decoded = DecodedBaselineState(
        attention_keys=reconstructed_keys,
        attention_values=reconstructed_values,
        reconstructed_full_keys=reconstructed_keys,
        reconstructed_full_values=reconstructed_values,
        retained_token_fraction=1.0,
        retained_block_fraction=1.0 if total_blocks > 0 else None,
        key_max_abs_error=float(np.max(np.abs(key_diff), initial=0.0)),
        key_rmse=_rmse(key_diff),
        value_max_abs_error=float(np.max(np.abs(value_diff), initial=0.0)),
        value_rmse=_rmse(value_diff),
        retained_old_token_count=context.historical_tokens,
        retained_old_block_count=total_blocks,
        total_old_block_count=total_blocks,
        extra={},
    )
    return decoded, authoritative_prefix


def _decode_state_output(
    context: Stage4CaseContext,
    state: DecodedBaselineState,
    *,
    precision: int,
) -> tuple[tuple[gmpy2.mpfr, ...], np.ndarray]:
    output_mpfr = tuple(
        exact_reference_output_mpfr(
            context.query,
            state.attention_keys,
            state.attention_values,
            precision=precision,
            attention_scale=context.attention_scale,
        )
    )
    return output_mpfr, _mpfr_vector_to_numpy(output_mpfr)


def _budget_metrics(target: int | None, actual: int, *, possible: bool) -> tuple[int | None, float | None, bool | None]:
    if target is None:
        return None, None, None
    abs_diff = abs(int(actual) - int(target))
    rel_diff = float(abs_diff / target) if target != 0 else math.inf
    return abs_diff, rel_diff, (rel_diff <= 0.02 if possible else None)


def _evaluate_generic_case(
    *,
    context: Stage4CaseContext,
    artifact: SerializedBaselineArtifact,
    decoded_state: DecodedBaselineState,
    precision: int,
    payload_relative_path: str | None,
) -> BaselineCaseResult:
    output_mpfr, output = _decode_state_output(context, decoded_state, precision=precision)
    attention_error = _norm(context.original_reference_output - output)
    model_reference_gap = _norm(context.captured_model_output - context.original_reference_output)
    captured_model_total_gap = _norm(context.captured_model_output - output)
    full_kv_bytes = context.full_kv_reference_bytes
    compression_ratio = float(full_kv_bytes / artifact.total_serialized_bytes) if artifact.total_serialized_bytes > 0 else math.inf
    memory_saving = float(1.0 - (artifact.total_serialized_bytes / full_kv_bytes)) if full_kv_bytes > 0 else 0.0
    budget_abs_diff, budget_rel_diff, budget_within = _budget_metrics(
        artifact.budget_target_bytes,
        artifact.total_serialized_bytes,
        possible=artifact.budget_match_possible,
    )
    return BaselineCaseResult(
        case_key=_build_case_key(context),
        method_name=artifact.definition.name,
        mode=artifact.mode,
        method_version=STAGE4_METHOD_VERSION,
        category=artifact.definition.category,
        citation_key=artifact.definition.citation_key,
        fidelity_status=artifact.definition.fidelity_status,
        source_label=artifact.definition.source_label,
        deviations=artifact.definition.deviations,
        layer_index=context.layer_index,
        record_index=context.record_index,
        query_position=context.query_position,
        query_local_index=context.query_local_index,
        query_head_global=context.query_head_global,
        kv_head_global=context.kv_head_global,
        visible_length=context.visible_length,
        historical_tokens=context.historical_tokens,
        recent_exact_tokens=context.recent_exact_tokens,
        attention_scale=context.attention_scale,
        full_kv_reference_bytes=full_kv_bytes,
        payload_sha256=artifact.payload_sha256,
        total_serialized_bytes=artifact.total_serialized_bytes,
        encoded_key_bytes=artifact.byte_breakdown.encoded_key_bytes,
        encoded_value_bytes=artifact.byte_breakdown.encoded_value_bytes,
        scales_bytes=artifact.byte_breakdown.scales_bytes,
        metadata_bytes=artifact.byte_breakdown.metadata_bytes,
        indices_bytes=artifact.byte_breakdown.indices_bytes,
        block_page_metadata_bytes=artifact.byte_breakdown.block_page_metadata_bytes,
        recent_window_bytes=artifact.byte_breakdown.recent_window_bytes,
        compression_ratio_vs_full_kv=compression_ratio,
        memory_saving_fraction_vs_full_kv=memory_saving,
        budget_tunable=artifact.budget_tunable,
        budget_match_attempted=artifact.budget_match_attempted,
        budget_target_bytes=artifact.budget_target_bytes,
        budget_abs_diff_bytes=budget_abs_diff,
        budget_rel_diff_fraction=budget_rel_diff,
        budget_match_possible=artifact.budget_match_possible,
        budget_within_two_percent=budget_within,
        retained_token_fraction=decoded_state.retained_token_fraction,
        retained_block_fraction=decoded_state.retained_block_fraction,
        attention_output_l2_error=attention_error,
        relative_l2_error=_relative_l2_error(context.original_reference_output, output),
        max_abs_component_error=_max_abs_component_error(context.original_reference_output, output),
        cosine_similarity=_cosine_similarity(context.original_reference_output, output),
        model_reference_gap=model_reference_gap,
        captured_model_total_gap=captured_model_total_gap,
        key_max_abs_error=decoded_state.key_max_abs_error,
        key_rmse=decoded_state.key_rmse,
        value_max_abs_error=decoded_state.value_max_abs_error,
        value_rmse=decoded_state.value_rmse,
        candidate_blocks=None,
        certified_skipped_blocks=None,
        certificate_upper_bound=None,
        observed_skipping_error=None,
        decoded_blocks=None,
        false_safe_count=None,
        rigorous_interval_violations=None,
        approximate_observed_violations=None,
        decomposition_violations=None,
        reference_decomposition_violations=None,
        model_relative_decomposition_violations=None,
        numerical_fallbacks=None,
        payload_relative_path=payload_relative_path,
        parameters=dict(artifact.parameters),
        extra={**decoded_state.extra},
    )


def _evaluate_rack_kv_case(
    *,
    trace: ValidatedCompactTrace,
    context: Stage4CaseContext,
    artifact: SerializedBaselineArtifact,
    authoritative_prefix_result: PrefixCompressionResult,
    precision: int,
    payload_relative_path: str | None,
    tolerance: float,
) -> BaselineCaseResult:
    base_case = _build_query_case_base(
        trace=trace,
        prefix_result=authoritative_prefix_result,
        query_local_index=context.query_local_index,
        precision=precision,
    )
    bundled = _run_query_case_tolerance_bundle(
        trace=trace,
        prefix_result=authoritative_prefix_result,
        query_local_index=context.query_local_index,
        tolerances=(tolerance,),
        precision=precision,
        base_case=base_case,
    )
    if len(bundled) != 1:
        raise AssertionError("RACK-KV bundled evaluation returned an unexpected result count.")
    case = bundled[0]
    output_mpfr, output = _decode_state_output(
        context,
        DecodedBaselineState(
            attention_keys=authoritative_prefix_result.reconstructed_keys,
            attention_values=authoritative_prefix_result.reconstructed_values,
            reconstructed_full_keys=authoritative_prefix_result.reconstructed_keys,
            reconstructed_full_values=authoritative_prefix_result.reconstructed_values,
            retained_token_fraction=1.0,
            retained_block_fraction=1.0 if len(authoritative_prefix_result.blocks) > 0 else None,
            key_max_abs_error=authoritative_prefix_result.max_key_abs_error,
            key_rmse=authoritative_prefix_result.key_rmse,
            value_max_abs_error=authoritative_prefix_result.max_value_abs_error,
            value_rmse=authoritative_prefix_result.value_rmse,
            retained_old_token_count=context.historical_tokens,
            retained_old_block_count=len(authoritative_prefix_result.blocks),
            total_old_block_count=len(authoritative_prefix_result.blocks),
            extra={},
        ),
        precision=precision,
    )
    full_kv_bytes = context.full_kv_reference_bytes
    compression_ratio = float(full_kv_bytes / artifact.total_serialized_bytes) if artifact.total_serialized_bytes > 0 else math.inf
    memory_saving = float(1.0 - (artifact.total_serialized_bytes / full_kv_bytes)) if full_kv_bytes > 0 else 0.0
    budget_abs_diff, budget_rel_diff, budget_within = _budget_metrics(
        artifact.budget_target_bytes,
        artifact.total_serialized_bytes,
        possible=artifact.budget_match_possible,
    )
    reference_decomposition_violation = int(
        case.reference_decomposition_lhs > case.reference_decomposition_rhs + 1e-9
    )
    model_relative_decomposition_violation = int(
        case.model_relative_decomposition_lhs > case.model_relative_decomposition_rhs + 1e-9
    )
    decomposition_violations = int(
        reference_decomposition_violation or model_relative_decomposition_violation
    )
    return BaselineCaseResult(
        case_key=_build_case_key(context),
        method_name=artifact.definition.name,
        mode=artifact.mode,
        method_version=STAGE4_METHOD_VERSION,
        category=artifact.definition.category,
        citation_key=artifact.definition.citation_key,
        fidelity_status=artifact.definition.fidelity_status,
        source_label=artifact.definition.source_label,
        deviations=artifact.definition.deviations,
        layer_index=context.layer_index,
        record_index=context.record_index,
        query_position=context.query_position,
        query_local_index=context.query_local_index,
        query_head_global=context.query_head_global,
        kv_head_global=context.kv_head_global,
        visible_length=context.visible_length,
        historical_tokens=context.historical_tokens,
        recent_exact_tokens=context.recent_exact_tokens,
        attention_scale=context.attention_scale,
        full_kv_reference_bytes=full_kv_bytes,
        payload_sha256=artifact.payload_sha256,
        total_serialized_bytes=artifact.total_serialized_bytes,
        encoded_key_bytes=artifact.byte_breakdown.encoded_key_bytes,
        encoded_value_bytes=artifact.byte_breakdown.encoded_value_bytes,
        scales_bytes=artifact.byte_breakdown.scales_bytes,
        metadata_bytes=artifact.byte_breakdown.metadata_bytes,
        indices_bytes=artifact.byte_breakdown.indices_bytes,
        block_page_metadata_bytes=artifact.byte_breakdown.block_page_metadata_bytes,
        recent_window_bytes=artifact.byte_breakdown.recent_window_bytes,
        compression_ratio_vs_full_kv=compression_ratio,
        memory_saving_fraction_vs_full_kv=memory_saving,
        budget_tunable=artifact.budget_tunable,
        budget_match_attempted=artifact.budget_match_attempted,
        budget_target_bytes=artifact.budget_target_bytes,
        budget_abs_diff_bytes=budget_abs_diff,
        budget_rel_diff_fraction=budget_rel_diff,
        budget_match_possible=artifact.budget_match_possible,
        budget_within_two_percent=budget_within,
        retained_token_fraction=1.0,
        retained_block_fraction=1.0 if len(authoritative_prefix_result.blocks) > 0 else None,
        attention_output_l2_error=case.reference_total_error,
        relative_l2_error=_relative_l2_error(context.original_reference_output, output),
        max_abs_component_error=_max_abs_component_error(context.original_reference_output, output),
        cosine_similarity=_cosine_similarity(context.original_reference_output, output),
        model_reference_gap=case.model_reference_gap,
        captured_model_total_gap=case.captured_model_total_gap,
        key_max_abs_error=authoritative_prefix_result.max_key_abs_error,
        key_rmse=authoritative_prefix_result.key_rmse,
        value_max_abs_error=authoritative_prefix_result.max_value_abs_error,
        value_rmse=authoritative_prefix_result.value_rmse,
        candidate_blocks=case.candidate_blocks,
        certified_skipped_blocks=case.certified_skipped_blocks,
        certificate_upper_bound=case.certificate_bound_upper_float,
        observed_skipping_error=case.observed_skip_error,
        decoded_blocks=case.decoded_blocks,
        false_safe_count=int(case.rigorous_interval_violation),
        rigorous_interval_violations=int(case.rigorous_interval_violation),
        approximate_observed_violations=int(case.approximate_observed_violation),
        decomposition_violations=decomposition_violations,
        reference_decomposition_violations=reference_decomposition_violation,
        model_relative_decomposition_violations=model_relative_decomposition_violation,
        numerical_fallbacks=int(case.numerical_fallback_used),
        payload_relative_path=payload_relative_path,
        parameters={**artifact.parameters, "tolerance": tolerance},
        extra={
            "certificate_name": case.certificate_name,
            "decoded_block_starts": list(case.decoded_block_starts),
            "skipped_block_starts": list(case.skipped_block_starts),
            "certificate_bound_text": case.certificate_bound_text,
            "rigorous_skip_error_upper_text": case.rigorous_skip_error_upper_text,
            "z_k_lower_text": case.z_k_lower_text,
            "u_s_upper_text": case.u_s_upper_text,
            "nu_s_upper_text": case.nu_s_upper_text,
            "kept_output_norm_upper_text": case.kept_output_norm_upper_text,
        },
    )


def _snap_native_keep_count(context: Stage4CaseContext) -> int:
    if context.historical_tokens <= 0:
        return 0
    return max(1, int(math.ceil(context.historical_tokens * STAGE4_SNAP_KEEP_FRACTION)))


def _quest_native_keep_block_count(context: Stage4CaseContext, block_size: int) -> int:
    blocks = _historical_blocks(context, block_size)
    if not blocks:
        return 0
    return max(1, int(math.ceil(len(blocks) * STAGE4_QUEST_KEEP_FRACTION)))


def _find_best_budget_match(
    *,
    candidate_builder: Callable[[int], SerializedBaselineArtifact],
    max_candidate: int,
    target_bytes: int,
) -> SerializedBaselineArtifact:
    best: SerializedBaselineArtifact | None = None
    best_abs_diff: int | None = None
    for candidate in range(0, max_candidate + 1):
        artifact = candidate_builder(candidate)
        abs_diff = abs(artifact.total_serialized_bytes - target_bytes)
        if best is None or best_abs_diff is None or abs_diff < best_abs_diff:
            best = artifact
            best_abs_diff = abs_diff
    if best is None:
        raise ValueError("No budget-match candidate was produced.")
    return best


def run_baseline_case(
    *,
    context: Stage4CaseContext,
    method_name: str,
    mode: str,
    trace: ValidatedCompactTrace | None = None,
    recent_window: int = STAGE4_RECENT_WINDOW,
    block_size: int = STAGE4_BLOCK_SIZE,
    tolerance: float = STAGE4_RACK_TOLERANCE,
    precision: int = STAGE4_PRECISION,
    payload_relative_path: str | None = None,
    rack_target_bytes: int | None = None,
) -> tuple[BaselineCaseResult, bytes]:
    if method_name not in BASELINE_DEFINITIONS:
        raise KeyError(f"Unknown baseline method: {method_name}")
    if mode not in {"native", "matched_budget"}:
        raise ValueError("mode must be 'native' or 'matched_budget'.")
    definition = BASELINE_DEFINITIONS[method_name]
    if mode == "matched_budget" and not definition.budget_tunable:
        raise ValueError(f"Method {method_name} does not support matched_budget mode.")

    if method_name == "full_kv":
        artifact = _serialize_full_kv(context, mode=mode)
        decoded = _decode_full_kv(artifact, context)
        return _evaluate_generic_case(
            context=context,
            artifact=artifact,
            decoded_state=decoded,
            precision=precision,
            payload_relative_path=payload_relative_path,
        ), artifact.payload

    if method_name == "rack_kv":
        active_trace = trace
        if active_trace is None:
            active_trace = validate_compact_trace(context.trace_path, allow_nonzero_layer=True)
        artifact, native_prefix = _serialize_rack_kv(
            context,
            mode=mode,
            recent_window=recent_window,
            block_size=block_size,
            precision=precision,
        )
        decoded, authoritative_prefix = _decode_rack_kv(
            artifact,
            context,
            native_prefix_result=native_prefix,
        )
        _ = decoded
        return _evaluate_rack_kv_case(
            trace=active_trace,
            context=context,
            artifact=artifact,
            authoritative_prefix_result=authoritative_prefix,
            precision=precision,
            payload_relative_path=payload_relative_path,
            tolerance=tolerance,
        ), artifact.payload

    if method_name == "uniform_int8_kv":
        artifact = _serialize_uniform_int8_kv(context, mode=mode)
        decoded = _decode_uniform_int8_kv(artifact, context)
        return _evaluate_generic_case(
            context=context,
            artifact=artifact,
            decoded_state=decoded,
            precision=precision,
            payload_relative_path=payload_relative_path,
        ), artifact.payload

    if method_name == "kivi_style":
        artifact = _serialize_kivi_style(context, mode=mode)
        decoded = _decode_kivi_style(artifact, context)
        return _evaluate_generic_case(
            context=context,
            artifact=artifact,
            decoded_state=decoded,
            precision=precision,
            payload_relative_path=payload_relative_path,
        ), artifact.payload

    if method_name == "snapkv_style":
        if mode == "matched_budget" and rack_target_bytes is None:
            raise ValueError("snapkv_style matched_budget mode requires rack_target_bytes.")
        if mode == "native" or rack_target_bytes is None:
            keep_count = _snap_native_keep_count(context)
            artifact = _serialize_snapkv_style(
                context,
                mode=mode,
                keep_count=keep_count,
                budget_target_bytes=rack_target_bytes if mode == "matched_budget" else None,
            )
        else:
            artifact = _find_best_budget_match(
                candidate_builder=lambda keep_count: _serialize_snapkv_style(
                    context,
                    mode=mode,
                    keep_count=keep_count,
                    budget_target_bytes=rack_target_bytes,
                ),
                max_candidate=context.historical_tokens,
                target_bytes=int(rack_target_bytes),
            )
        decoded = _decode_snapkv_style(artifact, context)
        return _evaluate_generic_case(
            context=context,
            artifact=artifact,
            decoded_state=decoded,
            precision=precision,
            payload_relative_path=payload_relative_path,
        ), artifact.payload

    if method_name == "quest_style":
        if mode == "matched_budget" and rack_target_bytes is None:
            raise ValueError("quest_style matched_budget mode requires rack_target_bytes.")
        total_blocks = len(_historical_blocks(context, block_size))
        if mode == "native" or rack_target_bytes is None:
            keep_block_count = _quest_native_keep_block_count(context, block_size)
            artifact = _serialize_quest_style(
                context,
                mode=mode,
                block_size=block_size,
                keep_block_count=keep_block_count,
                budget_target_bytes=rack_target_bytes if mode == "matched_budget" else None,
            )
        else:
            artifact = _find_best_budget_match(
                candidate_builder=lambda keep_block_count: _serialize_quest_style(
                    context,
                    mode=mode,
                    block_size=block_size,
                    keep_block_count=keep_block_count,
                    budget_target_bytes=rack_target_bytes,
                ),
                max_candidate=total_blocks,
                target_bytes=int(rack_target_bytes),
            )
        decoded = _decode_quest_style(artifact, context)
        return _evaluate_generic_case(
            context=context,
            artifact=artifact,
            decoded_state=decoded,
            precision=precision,
            payload_relative_path=payload_relative_path,
        ), artifact.payload

    raise AssertionError(f"Unhandled baseline method: {method_name}")


def stage4_case_result_to_dict(result: BaselineCaseResult) -> dict[str, Any]:
    return {
        **asdict(result),
        "deviations": list(result.deviations),
    }


def aggregate_baseline_results(results: Sequence[BaselineCaseResult]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[BaselineCaseResult]] = {}
    for result in results:
        grouped.setdefault((result.method_name, result.mode), []).append(result)
    summary: dict[str, Any] = {}
    for (method_name, mode), group in sorted(grouped.items()):
        output_errors = np.asarray([item.attention_output_l2_error for item in group], dtype=np.float64)
        relative_errors = np.asarray([item.relative_l2_error for item in group], dtype=np.float64)
        max_component_errors = np.asarray([item.max_abs_component_error for item in group], dtype=np.float64)
        cosine_similarities = np.asarray([item.cosine_similarity for item in group], dtype=np.float64)
        bytes_used = np.asarray([item.total_serialized_bytes for item in group], dtype=np.float64)
        compression_ratios = np.asarray([item.compression_ratio_vs_full_kv for item in group], dtype=np.float64)
        memory_savings = np.asarray([item.memory_saving_fraction_vs_full_kv for item in group], dtype=np.float64)
        token_fractions = np.asarray([item.retained_token_fraction for item in group], dtype=np.float64)
        non_null_block_fractions = [item.retained_block_fraction for item in group if item.retained_block_fraction is not None]
        key_max_errors = [item.key_max_abs_error for item in group if item.key_max_abs_error is not None]
        key_rmses = [item.key_rmse for item in group if item.key_rmse is not None]
        value_max_errors = [item.value_max_abs_error for item in group if item.value_max_abs_error is not None]
        value_rmses = [item.value_rmse for item in group if item.value_rmse is not None]
        budget_abs_diffs = [item.budget_abs_diff_bytes for item in group if item.budget_abs_diff_bytes is not None]
        budget_rel_diffs = [item.budget_rel_diff_fraction for item in group if item.budget_rel_diff_fraction is not None]
        rack_candidates = [int(item.candidate_blocks or 0) for item in group if item.method_name == "rack_kv"]
        rack_skipped = [int(item.certified_skipped_blocks or 0) for item in group if item.method_name == "rack_kv"]
        rack_skip_fractions = [
            (skipped / candidate) for skipped, candidate in zip(rack_skipped, rack_candidates) if candidate > 0
        ]
        summary[f"{method_name}:{mode}"] = {
            "method_name": method_name,
            "mode": mode,
            "case_count": len(group),
            "budget_tunable": group[0].budget_tunable,
            "budget_match_attempted": any(item.budget_match_attempted for item in group),
            "mean_attention_output_l2_error": float(np.mean(output_errors)) if output_errors.size else 0.0,
            "median_attention_output_l2_error": float(np.median(output_errors)) if output_errors.size else 0.0,
            "p95_attention_output_l2_error": float(np.quantile(output_errors, 0.95)) if output_errors.size else 0.0,
            "max_attention_output_l2_error": float(np.max(output_errors)) if output_errors.size else 0.0,
            "mean_relative_l2_error": float(np.mean(relative_errors)) if relative_errors.size else 0.0,
            "median_relative_l2_error": float(np.median(relative_errors)) if relative_errors.size else 0.0,
            "p95_relative_l2_error": float(np.quantile(relative_errors, 0.95)) if relative_errors.size else 0.0,
            "max_relative_l2_error": float(np.max(relative_errors)) if relative_errors.size else 0.0,
            "mean_max_abs_component_error": float(np.mean(max_component_errors)) if max_component_errors.size else 0.0,
            "median_max_abs_component_error": float(np.median(max_component_errors)) if max_component_errors.size else 0.0,
            "p95_max_abs_component_error": float(np.quantile(max_component_errors, 0.95)) if max_component_errors.size else 0.0,
            "max_max_abs_component_error": float(np.max(max_component_errors)) if max_component_errors.size else 0.0,
            "mean_cosine_similarity": float(np.mean(cosine_similarities)) if cosine_similarities.size else 0.0,
            "median_cosine_similarity": float(np.median(cosine_similarities)) if cosine_similarities.size else 0.0,
            "p95_cosine_similarity": float(np.quantile(cosine_similarities, 0.95)) if cosine_similarities.size else 0.0,
            "min_cosine_similarity": float(np.min(cosine_similarities)) if cosine_similarities.size else 0.0,
            "mean_serialized_bytes": float(np.mean(bytes_used)) if bytes_used.size else 0.0,
            "median_serialized_bytes": float(np.median(bytes_used)) if bytes_used.size else 0.0,
            "p95_serialized_bytes": float(np.quantile(bytes_used, 0.95)) if bytes_used.size else 0.0,
            "max_serialized_bytes": float(np.max(bytes_used)) if bytes_used.size else 0.0,
            "mean_compression_ratio_vs_full_kv": float(np.mean(compression_ratios)) if compression_ratios.size else 0.0,
            "median_compression_ratio_vs_full_kv": float(np.median(compression_ratios)) if compression_ratios.size else 0.0,
            "p95_compression_ratio_vs_full_kv": float(np.quantile(compression_ratios, 0.95)) if compression_ratios.size else 0.0,
            "max_compression_ratio_vs_full_kv": float(np.max(compression_ratios)) if compression_ratios.size else 0.0,
            "mean_memory_saving_fraction_vs_full_kv": float(np.mean(memory_savings)) if memory_savings.size else 0.0,
            "median_memory_saving_fraction_vs_full_kv": float(np.median(memory_savings)) if memory_savings.size else 0.0,
            "p95_memory_saving_fraction_vs_full_kv": float(np.quantile(memory_savings, 0.95)) if memory_savings.size else 0.0,
            "max_memory_saving_fraction_vs_full_kv": float(np.max(memory_savings)) if memory_savings.size else 0.0,
            "mean_retained_token_fraction": float(np.mean(token_fractions)) if token_fractions.size else 0.0,
            "median_retained_token_fraction": float(np.median(token_fractions)) if token_fractions.size else 0.0,
            "p95_retained_token_fraction": float(np.quantile(token_fractions, 0.95)) if token_fractions.size else 0.0,
            "max_retained_token_fraction": float(np.max(token_fractions)) if token_fractions.size else 0.0,
            "mean_retained_block_fraction": float(np.mean(non_null_block_fractions)) if non_null_block_fractions else None,
            "median_retained_block_fraction": float(np.median(non_null_block_fractions)) if non_null_block_fractions else None,
            "p95_retained_block_fraction": float(np.quantile(non_null_block_fractions, 0.95)) if non_null_block_fractions else None,
            "max_retained_block_fraction": float(np.max(non_null_block_fractions)) if non_null_block_fractions else None,
            "mean_key_max_abs_error": float(np.mean(key_max_errors)) if key_max_errors else None,
            "median_key_max_abs_error": float(np.median(key_max_errors)) if key_max_errors else None,
            "p95_key_max_abs_error": float(np.quantile(key_max_errors, 0.95)) if key_max_errors else None,
            "max_key_max_abs_error": float(np.max(key_max_errors)) if key_max_errors else None,
            "mean_key_rmse": float(np.mean(key_rmses)) if key_rmses else None,
            "median_key_rmse": float(np.median(key_rmses)) if key_rmses else None,
            "p95_key_rmse": float(np.quantile(key_rmses, 0.95)) if key_rmses else None,
            "max_key_rmse": float(np.max(key_rmses)) if key_rmses else None,
            "mean_value_max_abs_error": float(np.mean(value_max_errors)) if value_max_errors else None,
            "median_value_max_abs_error": float(np.median(value_max_errors)) if value_max_errors else None,
            "p95_value_max_abs_error": float(np.quantile(value_max_errors, 0.95)) if value_max_errors else None,
            "max_value_max_abs_error": float(np.max(value_max_errors)) if value_max_errors else None,
            "mean_value_rmse": float(np.mean(value_rmses)) if value_rmses else None,
            "median_value_rmse": float(np.median(value_rmses)) if value_rmses else None,
            "p95_value_rmse": float(np.quantile(value_rmses, 0.95)) if value_rmses else None,
            "max_value_rmse": float(np.max(value_rmses)) if value_rmses else None,
            "mean_budget_abs_diff_bytes": float(np.mean(budget_abs_diffs)) if budget_abs_diffs else None,
            "median_budget_abs_diff_bytes": float(np.median(budget_abs_diffs)) if budget_abs_diffs else None,
            "max_budget_abs_diff_bytes": float(np.max(budget_abs_diffs)) if budget_abs_diffs else None,
            "mean_budget_rel_diff_fraction": float(np.mean(budget_rel_diffs)) if budget_rel_diffs else None,
            "median_budget_rel_diff_fraction": float(np.median(budget_rel_diffs)) if budget_rel_diffs else None,
            "max_budget_rel_diff_fraction": float(np.max(budget_rel_diffs)) if budget_rel_diffs else None,
            "total_candidate_blocks": int(sum(rack_candidates)) if method_name == "rack_kv" else None,
            "total_certified_skipped_blocks": int(sum(rack_skipped)) if method_name == "rack_kv" else None,
            "weighted_certified_skip_fraction": float(sum(rack_skipped) / sum(rack_candidates)) if method_name == "rack_kv" and sum(rack_candidates) > 0 else (0.0 if method_name == "rack_kv" else None),
            "cases_with_any_certified_skipping": int(sum(1 for item in group if (item.certified_skipped_blocks or 0) > 0)) if method_name == "rack_kv" else None,
            "max_case_skip_fraction": float(max(rack_skip_fractions)) if rack_skip_fractions else (0.0 if method_name == "rack_kv" else None),
            "max_certificate_upper_bound": float(max((item.certificate_upper_bound or 0.0) for item in group)) if method_name == "rack_kv" else None,
            "max_observed_skipping_error": float(max((item.observed_skipping_error or 0.0) for item in group)) if method_name == "rack_kv" else None,
            "rigorous_interval_violation_count": int(sum((item.rigorous_interval_violations or 0) for item in group)) if method_name == "rack_kv" else None,
            "approximate_observed_violation_count": int(sum((item.approximate_observed_violations or 0) for item in group)) if method_name == "rack_kv" else None,
            "false_safe_count": int(sum((item.false_safe_count or 0) for item in group)) if method_name == "rack_kv" else None,
            "reference_decomposition_violation_count": int(sum((item.reference_decomposition_violations or 0) for item in group)) if method_name == "rack_kv" else None,
            "model_relative_decomposition_violation_count": int(sum((item.model_relative_decomposition_violations or 0) for item in group)) if method_name == "rack_kv" else None,
            "numerical_fallback_count": int(sum((item.numerical_fallbacks or 0) for item in group)) if method_name == "rack_kv" else None,
            "case_results": [stage4_case_result_to_dict(item) for item in group],
        }
    return summary
