from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import gmpy2
import numpy as np
import torch

from .certificate import (
    CertificationResult,
    ProgressiveCertificationStep,
    certify_progressive_skipping,
    exact_reference_output_mpfr,
    progressive_certification_steps,
    rigorous_attention_output_interval,
    rigorous_output_error_norm,
    rigorous_output_error_upper_from_intervals,
)
from .codec import (
    CompressedBlock,
    SerializedBlockContainer,
    deserialize_block_container,
    encode_block,
    serialize_block_container,
)
from .llama_trace import load_attention_trace, query_head_to_kv_head
from .rigorous import DEFAULT_PRECISION, exact_mpfr
from .types import CertificateMode, DecodeSchedule

TRACE_SCHEMA_COMPACT_V1 = "compact_final_cache_v1"
REQUIRED_TRACE_TENSORS = frozenset(
    {
        "queries",
        "final_keys",
        "final_values",
        "model_head_outputs",
        "visible_lengths",
        "record_token_ids",
        "record_token_positions",
        "record_layer_indices",
        "selected_query_heads",
        "selected_kv_heads",
    }
)


@dataclass(frozen=True)
class ValidatedCompactTrace:
    trace_path: Path
    trace_sha256: str
    trace_sha256_from_report: str | None
    checkpoint_repo: str
    checkpoint_revision: str
    trace_schema: str
    selected_query_heads: tuple[int, ...]
    selected_kv_heads: tuple[int, ...]
    query_to_kv_heads: tuple[int, ...]
    visible_lengths: tuple[int, ...]
    query_positions: tuple[int, ...]
    token_ids: tuple[int, ...]
    layer_indices: tuple[int, ...]
    num_attention_heads: int
    num_key_value_heads: int
    gqa_group_size: int
    head_dim: int
    scaling: float
    source_dtype: str
    storage_dtype: str
    queries: torch.Tensor
    final_keys: torch.Tensor
    final_values: torch.Tensor
    model_head_outputs: torch.Tensor
    metadata: dict[str, Any]

    @property
    def query_records(self) -> int:
        return int(self.queries.shape[0])

    @property
    def layer_index(self) -> int:
        return int(self.layer_indices[0])

    @property
    def query_head_count(self) -> int:
        return int(self.queries.shape[1])

    @property
    def value_dim(self) -> int:
        return int(self.model_head_outputs.shape[2])

    @property
    def sequence_length(self) -> int:
        return int(self.final_keys.shape[1])

    @property
    def kv_head_to_local(self) -> dict[int, int]:
        return {kv_head: index for index, kv_head in enumerate(self.selected_kv_heads)}

    def query_case_prefix(
        self,
        *,
        record_index: int,
        query_local_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visible_len = self.visible_lengths[record_index]
        kv_global = self.query_to_kv_heads[query_local_index]
        kv_local = self.kv_head_to_local[kv_global]
        return (
            self.final_keys[kv_local, :visible_len, :].clone(),
            self.final_values[kv_local, :visible_len, :].clone(),
        )

    def kv_prefix(
        self,
        *,
        record_index: int,
        kv_head_global: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visible_len = self.visible_lengths[record_index]
        kv_local = self.kv_head_to_local[kv_head_global]
        return (
            self.final_keys[kv_local, :visible_len, :].clone(),
            self.final_values[kv_local, :visible_len, :].clone(),
        )


@dataclass(frozen=True)
class PrefixCompressionResult:
    record_index: int
    kv_head_global: int
    visible_length: int
    recent_window: int
    block_size: int
    historical_tokens: int
    recent_exact_tokens: int
    blocks: tuple[CompressedBlock, ...]
    serialized_container: SerializedBlockContainer | None
    authoritative_serialized_roundtrip_used: bool
    original_keys: np.ndarray
    original_values: np.ndarray
    reconstructed_keys: np.ndarray
    reconstructed_values: np.ndarray
    recent_keys: np.ndarray
    recent_values: np.ndarray
    original_kv_bytes: int
    recent_exact_bytes: int
    compressed_historical_bytes: int
    total_compressed_bytes: int
    bytes_per_historical_token: float
    compression_ratio: float
    key_anchor_bytes: int
    value_anchor_bytes: int
    quantized_key_residual_bytes: int
    quantized_value_residual_bytes: int
    scale_bytes: int
    certificate_metadata_bytes: int
    block_header_bytes: int
    container_header_bytes: int
    container_index_bytes: int
    padding_alignment_bytes: int
    independent_blocks_tested: int
    independent_decode_max_diff: float
    unaffected_block_decode_tested: bool
    unaffected_block_decode_passed: bool
    max_key_abs_error: float
    key_rmse: float
    max_value_abs_error: float
    value_rmse: float


@dataclass(frozen=True)
class QueryCaseBase:
    query: np.ndarray
    captured_model_output: np.ndarray
    original_reference_output_mpfr: tuple[gmpy2.mpfr, ...]
    original_reference_output: np.ndarray
    reconstructed_full_output_mpfr: tuple[gmpy2.mpfr, ...]
    reconstructed_full_output: np.ndarray
    model_reference_gap: float
    compression_error: float
    reconstructed_full_interval: tuple[Any, ...] | None


@dataclass(frozen=True)
class QueryCaseResult:
    record_index: int
    query_local_index: int
    query_head_global: int
    kv_head_global: int
    visible_length: int
    query_position: int
    historical_tokens: int
    recent_exact_tokens: int
    candidate_blocks: int
    certified_skipped_blocks: int
    decoded_blocks: int
    decoded_block_starts: tuple[int, ...]
    skipped_block_starts: tuple[int, ...]
    certificate_name: str
    z_k_lower_text: str | None
    z_k_lower_upper_float: float | None
    u_s_upper_text: str | None
    u_s_upper_float: float | None
    nu_s_upper_text: str | None
    nu_s_upper_float: float | None
    kept_output_norm_upper_text: str | None
    kept_output_norm_upper_float: float | None
    certificate_bound_text: str
    certificate_bound_upper_float: float
    observed_skip_error: float
    rigorous_skip_error_upper_text: str
    rigorous_skip_error_upper_float: float
    bound_to_observed_ratio: float | None
    approximate_observed_violation: bool
    rigorous_interval_violation: bool
    numerical_fallback_used: bool
    model_reference_gap: float
    compression_error: float
    reference_total_error: float
    captured_model_total_gap: float
    reference_decomposition_lhs: float
    reference_decomposition_rhs: float
    model_relative_decomposition_lhs: float
    model_relative_decomposition_rhs: float


@dataclass(frozen=True)
class ExperimentConfig:
    recent_window: int
    block_size: int
    tolerance: float


@dataclass(frozen=True)
class ExperimentConfigSummary:
    config: ExperimentConfig
    evaluated_query_cases: int
    eligible_query_cases: int
    unique_prefix_slices: int
    unique_prefix_slices_with_history: int
    aggregate_prefix_original_bytes: int
    aggregate_prefix_serialized_bytes: int
    aggregate_prefix_historical_serialized_bytes: int
    aggregate_prefix_compression_ratio: float
    min_prefix_compression_ratio: float
    mean_prefix_compression_ratio: float
    median_prefix_compression_ratio: float
    max_prefix_compression_ratio: float
    final_prefix_original_bytes: int
    final_prefix_serialized_bytes: int
    final_prefix_historical_serialized_bytes: int
    final_prefix_compression_ratio: float
    final_prefix_values_per_selected_kv_head: int
    final_prefix_total_values_selected_kv_heads: int
    bytes_per_historical_token: float
    max_key_abs_error: float
    key_rmse: float
    max_value_abs_error: float
    value_rmse: float
    number_of_blocks: int
    independently_decoded_blocks_tested: int
    independent_versus_full_decode_max_difference: float
    unaffected_block_decoding_tests: int
    unaffected_block_decoding_passed: bool
    serializer_roundtrip_checks_passed: bool
    total_candidate_blocks: int
    certified_skipped_blocks: int
    decoded_blocks: int
    skipped_block_fraction: float
    decoded_block_fraction: float
    max_certified_bound: float
    max_observed_reconstructed_skipping_error: float
    max_rigorous_skip_error_upper: float
    min_bound_to_observed_ratio_nonzero: float | None
    rigorous_interval_violation_count: int
    approximate_observed_violation_count: int
    false_safe_violation_count: int
    fallback_count: int
    max_model_reference_gap: float
    max_true_compression_error: float
    max_reference_total_error: float
    max_captured_model_total_gap: float
    reference_decomposition_violation_count: int
    model_relative_decomposition_violation_count: int
    query_case_results: tuple[QueryCaseResult, ...]


@dataclass(frozen=True)
class Stage2SmokeResult:
    trace: ValidatedCompactTrace
    precision: int
    random_seed: int
    global_prefix_ratio_min: float
    global_prefix_ratio_mean: float
    global_prefix_ratio_median: float
    global_prefix_ratio_max: float
    configs: tuple[ExperimentConfigSummary, ...]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _ensure_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype.is_floating_point:
        if not torch.isfinite(tensor.to(torch.float32)).all():
            raise ValueError(f"{name} contains non-finite values.")


def _float64_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float64).numpy()


def _load_capture_report(trace_path: Path) -> tuple[str | None, dict[str, Any] | None]:
    report_path = trace_path.with_name("capture_report.json")
    if not report_path.exists():
        return None, None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report.get("trace_sha256"), report


def validate_compact_trace(
    trace_path: str | Path,
    *,
    allow_nonzero_layer: bool = False,
) -> ValidatedCompactTrace:
    trace_file = Path(trace_path)
    if not trace_file.exists():
        raise FileNotFoundError(f"Trace file does not exist: {trace_file}")

    tensors, metadata = load_attention_trace(trace_file)
    missing = REQUIRED_TRACE_TENSORS.difference(tensors.keys())
    if missing:
        raise ValueError(f"Trace is missing required tensors: {sorted(missing)}")
    if metadata.get("trace_schema") != TRACE_SCHEMA_COMPACT_V1:
        raise ValueError(f"Unexpected trace schema: {metadata.get('trace_schema')!r}")

    queries = tensors["queries"].detach().cpu()
    final_keys = tensors["final_keys"].detach().cpu()
    final_values = tensors["final_values"].detach().cpu()
    model_head_outputs = tensors["model_head_outputs"].detach().cpu()
    visible_lengths = tuple(int(value) for value in tensors["visible_lengths"].tolist())
    query_positions = tuple(int(value) for value in tensors["record_token_positions"].tolist())
    token_ids = tuple(int(value) for value in tensors["record_token_ids"].tolist())
    layer_indices = tuple(int(value) for value in tensors["record_layer_indices"].tolist())
    selected_query_heads_tensor = tuple(int(value) for value in tensors["selected_query_heads"].tolist())
    selected_kv_heads_tensor = tuple(int(value) for value in tensors["selected_kv_heads"].tolist())
    selected_query_heads_meta = tuple(int(value) for value in metadata["selected_query_heads"])
    selected_kv_heads_meta = tuple(int(value) for value in metadata["selected_kv_heads"])
    query_to_kv_heads = tuple(int(value) for value in metadata["query_to_kv_heads"])
    num_attention_heads = int(metadata["num_attention_heads"])
    num_key_value_heads = int(metadata["num_key_value_heads"])
    gqa_group_size = int(metadata["gqa_group_size"])
    head_dim = int(metadata["head_dim"])
    scaling = float(metadata["scaling"])
    source_dtype = str(metadata["source_dtype"])
    storage_dtype = str(metadata["storage_dtype"])

    if selected_query_heads_tensor != selected_query_heads_meta:
        raise ValueError("selected_query_heads tensor does not match trace metadata.")
    if selected_kv_heads_tensor != selected_kv_heads_meta:
        raise ValueError("selected_kv_heads tensor does not match trace metadata.")
    if len(query_to_kv_heads) != len(selected_query_heads_tensor):
        raise ValueError("query_to_kv_heads length does not match the selected query heads.")
    if len(set(selected_query_heads_tensor)) != len(selected_query_heads_tensor):
        raise ValueError("selected_query_heads must be unique.")
    if len(set(selected_kv_heads_tensor)) != len(selected_kv_heads_tensor):
        raise ValueError("selected_kv_heads must be unique.")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("Trace GQA geometry is invalid: query heads are not divisible by KV heads.")
    if gqa_group_size != num_attention_heads // num_key_value_heads:
        raise ValueError("Trace gqa_group_size does not match the saved head geometry.")
    for query_head, kv_head in zip(selected_query_heads_tensor, query_to_kv_heads):
        expected = query_head_to_kv_head(
            query_head_index=query_head,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
        if kv_head != expected:
            raise ValueError("Trace query_to_kv_heads mapping does not match the saved GQA geometry.")
        if kv_head not in selected_kv_heads_tensor:
            raise ValueError("Trace query_to_kv_heads contains a KV head outside selected_kv_heads.")

    if queries.ndim != 3 or model_head_outputs.ndim != 3:
        raise ValueError("Trace query tensors must be rank-3.")
    if final_keys.ndim != 3 or final_values.ndim != 3:
        raise ValueError("Trace final K/V tensors must be rank-3.")
    if queries.shape != model_head_outputs.shape:
        raise ValueError("queries and model_head_outputs must have matching shapes.")
    if final_keys.shape != final_values.shape:
        raise ValueError("final_keys and final_values must have matching shapes.")
    if int(queries.shape[1]) != len(selected_query_heads_tensor):
        raise ValueError("Query-head axis does not match selected_query_heads.")
    if int(final_keys.shape[0]) != len(selected_kv_heads_tensor):
        raise ValueError("KV-head axis does not match selected_kv_heads.")
    if int(queries.shape[2]) != head_dim or int(final_keys.shape[2]) != head_dim:
        raise ValueError("Trace head_dim does not match the stored tensor widths.")
    if len(visible_lengths) != int(queries.shape[0]):
        raise ValueError("visible_lengths does not match the number of query records.")
    if len(query_positions) != int(queries.shape[0]):
        raise ValueError("record_token_positions does not match the number of query records.")
    if len(token_ids) != int(queries.shape[0]):
        raise ValueError("record_token_ids does not match the number of query records.")
    if len(layer_indices) != int(queries.shape[0]):
        raise ValueError("record_layer_indices does not match the number of query records.")

    sequence_length = int(final_keys.shape[1])
    if sequence_length <= 0:
        raise ValueError("Trace final cache sequence length must be positive.")
    if not layer_indices or len(set(layer_indices)) != 1:
        raise ValueError("Trace must contain one consistent layer index.")
    if layer_indices[0] != int(metadata["layer_index"]):
        raise ValueError("record_layer_indices do not match metadata layer_index.")
    if not allow_nonzero_layer and layer_indices[0] != 0:
        raise ValueError("This validation path only accepts validated layer-0 traces unless allow_nonzero_layer=True.")
    if list(query_positions) != sorted(query_positions):
        raise ValueError("record_token_positions must be monotonic nondecreasing.")
    if any(position < 0 or position >= sequence_length for position in query_positions):
        raise ValueError("record_token_positions fall outside the saved final cache range.")
    if any(visible_len <= 0 or visible_len > sequence_length for visible_len in visible_lengths):
        raise ValueError("visible_lengths fall outside the saved final cache range.")
    if any(visible_len != position + 1 for visible_len, position in zip(visible_lengths, query_positions)):
        raise ValueError("visible_lengths must equal query_position + 1 for the saved causal trace.")
    if len(token_ids) != len(query_positions):
        raise ValueError("record_token_ids length does not match record_token_positions.")
    if not math.isfinite(scaling) or scaling <= 0.0:
        raise ValueError("Trace scaling must be finite and positive.")

    _ensure_finite_tensor("queries", queries)
    _ensure_finite_tensor("final_keys", final_keys)
    _ensure_finite_tensor("final_values", final_values)
    _ensure_finite_tensor("model_head_outputs", model_head_outputs)

    trace_sha256 = _sha256_file(trace_file)
    report_trace_sha256, capture_report = _load_capture_report(trace_file)
    if report_trace_sha256 is not None and report_trace_sha256 != trace_sha256:
        raise ValueError("Trace SHA-256 does not match the saved capture provenance report.")
    if capture_report is not None:
        report_selected_query_heads = tuple(int(value) for value in capture_report["selected_query_heads"])
        report_selected_kv_heads = tuple(int(value) for value in capture_report["selected_kv_heads"])
        report_query_to_kv_heads = tuple(int(value) for value in capture_report["query_to_kv_heads"])
        if report_selected_query_heads != selected_query_heads_tensor:
            raise ValueError("Capture report selected_query_heads does not match the trace.")
        if report_selected_kv_heads != selected_kv_heads_tensor:
            raise ValueError("Capture report selected_kv_heads does not match the trace.")
        if report_query_to_kv_heads != query_to_kv_heads:
            raise ValueError("Capture report query_to_kv_heads does not match the trace.")

    return ValidatedCompactTrace(
        trace_path=trace_file,
        trace_sha256=trace_sha256,
        trace_sha256_from_report=report_trace_sha256,
        checkpoint_repo=str(metadata["checkpoint_repo"]),
        checkpoint_revision=str(metadata["checkpoint_revision"]),
        trace_schema=str(metadata["trace_schema"]),
        selected_query_heads=selected_query_heads_tensor,
        selected_kv_heads=selected_kv_heads_tensor,
        query_to_kv_heads=query_to_kv_heads,
        visible_lengths=visible_lengths,
        query_positions=query_positions,
        token_ids=token_ids,
        layer_indices=layer_indices,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        gqa_group_size=gqa_group_size,
        head_dim=head_dim,
        scaling=scaling,
        source_dtype=source_dtype,
        storage_dtype=storage_dtype,
        queries=queries,
        final_keys=final_keys,
        final_values=final_values,
        model_head_outputs=model_head_outputs,
        metadata=dict(metadata),
    )


def _block_slices(historical_tokens: int, block_size: int) -> list[tuple[int, int]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    return [
        (block_start, min(block_start + block_size, historical_tokens))
        for block_start in range(0, historical_tokens, block_size)
    ]


def _rmse(diff: np.ndarray) -> float:
    if diff.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(diff, dtype=np.float64), dtype=np.float64)))


def _float16_bits(value: np.float16) -> int:
    return int(np.asarray([value], dtype=np.float16).view(np.uint16)[0])


def _float32_bits(value: np.float32) -> int:
    return int(np.asarray([value], dtype=np.float32).view(np.uint32)[0])


def _assert_block_roundtrip_equal(reference: CompressedBlock, roundtrip: CompressedBlock) -> None:
    if reference.header.block_start != roundtrip.header.block_start:
        raise AssertionError("Round-tripped block_start differs from the original serialized block.")
    if reference.header.block_len != roundtrip.header.block_len:
        raise AssertionError("Round-tripped block_len differs from the original serialized block.")
    if reference.header.anchor_key.tobytes(order="C") != roundtrip.header.anchor_key.tobytes(order="C"):
        raise AssertionError("Round-tripped anchor_key bytes differ from the original serialized block.")
    if reference.header.anchor_value.tobytes(order="C") != roundtrip.header.anchor_value.tobytes(order="C"):
        raise AssertionError("Round-tripped anchor_value bytes differ from the original serialized block.")
    if _float16_bits(reference.header.key_scale) != _float16_bits(roundtrip.header.key_scale):
        raise AssertionError("Round-tripped key_scale bits differ from the original serialized block.")
    if _float16_bits(reference.header.value_scale) != _float16_bits(roundtrip.header.value_scale):
        raise AssertionError("Round-tripped value_scale bits differ from the original serialized block.")
    if _float32_bits(reference.header.rho_upper) != _float32_bits(roundtrip.header.rho_upper):
        raise AssertionError("Round-tripped rho_upper bits differ from the original serialized block.")
    if _float32_bits(reference.header.nu_upper) != _float32_bits(roundtrip.header.nu_upper):
        raise AssertionError("Round-tripped nu_upper bits differ from the original serialized block.")
    if reference.key_residuals.tobytes(order="C") != roundtrip.key_residuals.tobytes(order="C"):
        raise AssertionError("Round-tripped key residual payload differs from the original serialized block.")
    if reference.value_residuals.tobytes(order="C") != roundtrip.value_residuals.tobytes(order="C"):
        raise AssertionError("Round-tripped value residual payload differs from the original serialized block.")

    reference_keys, reference_values = reference.decode_block()
    roundtrip_keys, roundtrip_values = roundtrip.decode_block()
    if not np.array_equal(reference_keys, roundtrip_keys):
        raise AssertionError("Round-tripped reconstructed keys differ from the original serialized block.")
    if not np.array_equal(reference_values, roundtrip_values):
        raise AssertionError("Round-tripped reconstructed values differ from the original serialized block.")


def _build_prefix_result(
    *,
    record_index: int,
    kv_head_global: int,
    prefix_keys: torch.Tensor,
    prefix_values: torch.Tensor,
    recent_window: int,
    block_size: int,
    precision: int,
) -> PrefixCompressionResult:
    visible_length = int(prefix_keys.shape[0])
    recent_exact_tokens = min(recent_window, visible_length)
    historical_tokens = max(visible_length - recent_exact_tokens, 0)

    original_keys = _float64_numpy(prefix_keys)
    original_values = _float64_numpy(prefix_values)
    recent_keys_t = prefix_keys[historical_tokens:, :]
    recent_values_t = prefix_values[historical_tokens:, :]
    recent_keys = _float64_numpy(recent_keys_t)
    recent_values = _float64_numpy(recent_values_t)

    encoded_blocks: list[CompressedBlock] = []
    for block_start, block_end in _block_slices(historical_tokens, block_size):
        block_keys = original_keys[block_start:block_end, :]
        block_values = original_values[block_start:block_end, :]
        encoded_blocks.append(encode_block(block_keys, block_values, block_start=block_start, precision=precision))

    serialized_container: SerializedBlockContainer | None = None
    authoritative_serialized_roundtrip_used = True
    key_anchor_bytes = 0
    value_anchor_bytes = 0
    quantized_key_residual_bytes = 0
    quantized_value_residual_bytes = 0
    scale_bytes = 0
    certificate_metadata_bytes = 0
    block_header_bytes = 0
    container_header_bytes = 0
    container_index_bytes = 0
    padding_alignment_bytes = 0
    independent_blocks_tested = 0
    independent_decode_max_diff = 0.0
    unaffected_block_decode_tested = False
    unaffected_block_decode_passed = True
    reconstructed_historical_keys: list[np.ndarray] = []
    reconstructed_historical_values: list[np.ndarray] = []
    blocks: list[CompressedBlock] = []

    if encoded_blocks:
        emitted_container = serialize_block_container(encoded_blocks)
        serialized_container = deserialize_block_container(emitted_container.buffer)
        if serialized_container.total_bytes != len(serialized_container.buffer):
            raise ValueError("Serialized block container length is inconsistent with its buffer.")
        block_count = len(encoded_blocks)
        key_dim = int(encoded_blocks[0].header.anchor_key.shape[0])
        value_dim = int(encoded_blocks[0].header.anchor_value.shape[0])
        key_anchor_bytes = block_count * 2 * key_dim
        value_anchor_bytes = block_count * 2 * value_dim
        quantized_key_residual_bytes = sum(max(block.block_len - 1, 0) * key_dim for block in encoded_blocks)
        quantized_value_residual_bytes = sum(max(block.block_len - 1, 0) * value_dim for block in encoded_blocks)
        scale_bytes = block_count * (2 + 2)
        certificate_metadata_bytes = block_count * (4 + 4)
        block_header_bytes = block_count * 8
        container_header_bytes = serialized_container.header_bytes
        container_index_bytes = serialized_container.index_bytes
        payload_total = 0
        for block_index, encoded_block in enumerate(encoded_blocks):
            payload = serialized_container.payload_for_block(block_index)
            payload_total += len(payload)
            authoritative_block = serialized_container.deserialize_block(block_index)
            _assert_block_roundtrip_equal(encoded_block, authoritative_block)
            blocks.append(authoritative_block)
            authoritative_keys, authoritative_values = authoritative_block.decode_block()
            reconstructed_historical_keys.append(authoritative_keys)
            reconstructed_historical_values.append(authoritative_values)
            independently_decoded = CompressedBlock.deserialize(payload, key_dim=key_dim, value_dim=value_dim)
            _assert_block_roundtrip_equal(authoritative_block, independently_decoded)
            independent_keys = independently_decoded.decode_key_block()
            independent_values = independently_decoded.decode_value_block()
            independent_blocks_tested += 1
            independent_decode_max_diff = max(
                independent_decode_max_diff,
                float(np.max(np.abs(independent_keys - authoritative_keys), initial=0.0)),
                float(np.max(np.abs(independent_values - authoritative_values), initial=0.0)),
            )
            if not np.array_equal(independent_keys, authoritative_keys) or not np.array_equal(independent_values, authoritative_values):
                raise AssertionError("Independent block decode does not match the sequential decode result.")
        if payload_total + container_header_bytes + container_index_bytes + padding_alignment_bytes != serialized_container.total_bytes:
            raise AssertionError("Serializer-backed block accounting does not match the actual container length.")
        if len(blocks) > 1:
            unaffected_block_decode_tested = True
            target_index = 1
            corrupted_index = 0
            corrupted = bytearray(serialized_container.buffer)
            corrupted_offset = serialized_container.block_offsets[corrupted_index]
            corrupted[corrupted_offset] ^= 0x01
            corrupted_container = deserialize_block_container(bytes(corrupted))
            corrupted_target = corrupted_container.deserialize_block(target_index)
            expected_target_keys, expected_target_values = blocks[target_index].decode_block()
            corrupted_keys, corrupted_values = corrupted_target.decode_block()
            unaffected_block_decode_passed = bool(
                np.array_equal(corrupted_keys, expected_target_keys)
                and np.array_equal(corrupted_values, expected_target_values)
            )
            if not unaffected_block_decode_passed:
                raise AssertionError("Corrupting one block changed an unrelated block's decode result.")
    else:
        key_dim = int(prefix_keys.shape[1])
        value_dim = int(prefix_values.shape[1])

    reconstructed_keys = np.vstack([*(reconstructed_historical_keys or []), recent_keys]) if visible_length else np.zeros((0, key_dim), dtype=np.float64)
    reconstructed_values = np.vstack([*(reconstructed_historical_values or []), recent_values]) if visible_length else np.zeros((0, value_dim), dtype=np.float64)
    if reconstructed_keys.shape != original_keys.shape or reconstructed_values.shape != original_values.shape:
        raise AssertionError("Reconstructed prefix shapes do not match the original visible prefix.")

    key_diff = reconstructed_keys - original_keys
    value_diff = reconstructed_values - original_values
    compressed_historical_bytes = serialized_container.total_bytes if serialized_container is not None else 0
    recent_exact_bytes = _tensor_bytes(recent_keys_t) + _tensor_bytes(recent_values_t)
    original_kv_bytes = _tensor_bytes(prefix_keys) + _tensor_bytes(prefix_values)
    total_compressed_bytes = recent_exact_bytes + compressed_historical_bytes
    if total_compressed_bytes <= 0:
        raise ValueError("Total compressed bytes must stay positive for a non-empty visible prefix.")
    compression_ratio = float(original_kv_bytes / total_compressed_bytes)
    bytes_per_historical_token = (
        float(compressed_historical_bytes / historical_tokens) if historical_tokens > 0 else 0.0
    )

    return PrefixCompressionResult(
        record_index=record_index,
        kv_head_global=kv_head_global,
        visible_length=visible_length,
        recent_window=recent_window,
        block_size=block_size,
        historical_tokens=historical_tokens,
        recent_exact_tokens=recent_exact_tokens,
        blocks=tuple(blocks),
        serialized_container=serialized_container,
        authoritative_serialized_roundtrip_used=authoritative_serialized_roundtrip_used,
        original_keys=original_keys,
        original_values=original_values,
        reconstructed_keys=reconstructed_keys,
        reconstructed_values=reconstructed_values,
        recent_keys=recent_keys,
        recent_values=recent_values,
        original_kv_bytes=original_kv_bytes,
        recent_exact_bytes=recent_exact_bytes,
        compressed_historical_bytes=compressed_historical_bytes,
        total_compressed_bytes=total_compressed_bytes,
        bytes_per_historical_token=bytes_per_historical_token,
        compression_ratio=compression_ratio,
        key_anchor_bytes=key_anchor_bytes,
        value_anchor_bytes=value_anchor_bytes,
        quantized_key_residual_bytes=quantized_key_residual_bytes,
        quantized_value_residual_bytes=quantized_value_residual_bytes,
        scale_bytes=scale_bytes,
        certificate_metadata_bytes=certificate_metadata_bytes,
        block_header_bytes=block_header_bytes,
        container_header_bytes=container_header_bytes,
        container_index_bytes=container_index_bytes,
        padding_alignment_bytes=padding_alignment_bytes,
        independent_blocks_tested=independent_blocks_tested,
        independent_decode_max_diff=independent_decode_max_diff,
        unaffected_block_decode_tested=unaffected_block_decode_tested,
        unaffected_block_decode_passed=unaffected_block_decode_passed,
        max_key_abs_error=float(np.max(np.abs(key_diff), initial=0.0)),
        key_rmse=_rmse(key_diff),
        max_value_abs_error=float(np.max(np.abs(value_diff), initial=0.0)),
        value_rmse=_rmse(value_diff),
    )


def _kept_from_decoded_block_starts(prefix_result: PrefixCompressionResult, decoded_block_starts: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    decoded = set(decoded_block_starts)
    kept_keys: list[np.ndarray] = []
    kept_values: list[np.ndarray] = []
    for block in prefix_result.blocks:
        if block.header.block_start in decoded:
            block_keys, block_values = block.decode_block()
            kept_keys.append(block_keys)
            kept_values.append(block_values)
    kept_keys.append(prefix_result.recent_keys)
    kept_values.append(prefix_result.recent_values)
    return np.vstack(kept_keys), np.vstack(kept_values)


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(value.astype(np.float64), ord=2))


def _mpfr_vector_to_numpy(values: Sequence[gmpy2.mpfr]) -> np.ndarray:
    return np.asarray([float(component) for component in values], dtype=np.float64)


def _canonical_attention_scale(head_dim: int) -> float:
    return float(1.0 / math.sqrt(head_dim))


def _skip_validation_flags(
    *,
    certificate_value: gmpy2.mpfr,
    observed_skip_error_mpfr: gmpy2.mpfr,
    rigorous_skip_error_upper: gmpy2.mpfr,
) -> tuple[bool, bool]:
    approximate_observed_violation = observed_skip_error_mpfr > certificate_value
    rigorous_interval_violation = rigorous_skip_error_upper > certificate_value
    return approximate_observed_violation, rigorous_interval_violation


def _select_progressive_step(
    *,
    steps: Sequence[ProgressiveCertificationStep],
    tolerance: float,
    precision: int,
) -> ProgressiveCertificationStep:
    if not steps:
        raise ValueError("At least one certification progress step is required.")
    tolerance_exact = exact_mpfr(tolerance, precision=precision)
    for step in steps:
        if step.certificate_value_mpfr <= tolerance_exact:
            return step
    return steps[-1]


def _run_query_case_tolerance_bundle(
    *,
    trace: ValidatedCompactTrace,
    prefix_result: PrefixCompressionResult,
    query_local_index: int,
    tolerances: Sequence[float],
    precision: int,
    base_case: QueryCaseBase,
) -> tuple[QueryCaseResult, ...]:
    if not tolerances:
        return tuple()

    record_index = prefix_result.record_index
    query_head_global = trace.selected_query_heads[query_local_index]
    query = base_case.query
    captured_model_output = base_case.captured_model_output
    original_reference_output = base_case.original_reference_output
    reconstructed_full_output_mpfr = tuple(base_case.reconstructed_full_output_mpfr)
    reconstructed_full_output = base_case.reconstructed_full_output
    model_reference_gap = base_case.model_reference_gap
    compression_error = base_case.compression_error

    if not prefix_result.blocks:
        reference_total_error = _norm(original_reference_output - reconstructed_full_output)
        captured_model_total_gap = _norm(captured_model_output - reconstructed_full_output)
        reference_decomposition_rhs = compression_error
        model_relative_decomposition_rhs = model_reference_gap + compression_error
        if reference_total_error > reference_decomposition_rhs + 1e-9:
            raise AssertionError("Reference decomposition failed numerically in the zero-history case.")
        if captured_model_total_gap > model_relative_decomposition_rhs + 1e-9:
            raise AssertionError("Model-relative decomposition failed numerically in the zero-history case.")
        zero_result = QueryCaseResult(
            record_index=record_index,
            query_local_index=query_local_index,
            query_head_global=query_head_global,
            kv_head_global=prefix_result.kv_head_global,
            visible_length=prefix_result.visible_length,
            query_position=trace.query_positions[record_index],
            historical_tokens=prefix_result.historical_tokens,
            recent_exact_tokens=prefix_result.recent_exact_tokens,
            candidate_blocks=0,
            certified_skipped_blocks=0,
            decoded_blocks=0,
            decoded_block_starts=tuple(),
            skipped_block_starts=tuple(),
            certificate_name="none",
            z_k_lower_text=None,
            z_k_lower_upper_float=None,
            u_s_upper_text=None,
            u_s_upper_float=None,
            nu_s_upper_text=None,
            nu_s_upper_float=None,
            kept_output_norm_upper_text=None,
            kept_output_norm_upper_float=None,
            certificate_bound_text="0.0",
            certificate_bound_upper_float=0.0,
            observed_skip_error=0.0,
            rigorous_skip_error_upper_text="0.0",
            rigorous_skip_error_upper_float=0.0,
            bound_to_observed_ratio=None,
            approximate_observed_violation=False,
            rigorous_interval_violation=False,
            numerical_fallback_used=False,
            model_reference_gap=model_reference_gap,
            compression_error=compression_error,
            reference_total_error=reference_total_error,
            captured_model_total_gap=captured_model_total_gap,
            reference_decomposition_lhs=reference_total_error,
            reference_decomposition_rhs=reference_decomposition_rhs,
            model_relative_decomposition_lhs=captured_model_total_gap,
            model_relative_decomposition_rhs=model_relative_decomposition_rhs,
        )
        return tuple(zero_result for _ in tolerances)

    progress_steps = progressive_certification_steps(
        query=query,
        recent_keys=prefix_result.recent_keys,
        recent_values=prefix_result.recent_values,
        historical_blocks=prefix_result.blocks,
        schedule=DecodeSchedule.LARGEST_U_TIMES_NU,
        precision=precision,
        attention_scale=trace.scaling,
    )
    if not progress_steps:
        raise AssertionError("Progressive certification produced no steps.")

    full_interval = base_case.reconstructed_full_interval
    kept_state_cache: dict[
        tuple[int, ...],
        tuple[np.ndarray, tuple[gmpy2.mpfr, ...], gmpy2.mpfr, float, tuple[Any, ...]],
    ] = {}
    results: list[QueryCaseResult] = []

    for tolerance in tolerances:
        step = _select_progressive_step(steps=progress_steps, tolerance=float(tolerance), precision=precision)
        if not step.skipped_block_starts:
            reference_total_error = compression_error
            captured_model_total_gap = _norm(captured_model_output - reconstructed_full_output)
            reference_decomposition_rhs = compression_error
            model_relative_decomposition_rhs = model_reference_gap + compression_error
            if reference_total_error > reference_decomposition_rhs + 1e-9:
                raise AssertionError("Reference decomposition failed numerically in the zero-skip case.")
            if captured_model_total_gap > model_relative_decomposition_rhs + 1e-9:
                raise AssertionError("Model-relative decomposition failed numerically in the zero-skip case.")
            results.append(
                QueryCaseResult(
                    record_index=record_index,
                    query_local_index=query_local_index,
                    query_head_global=query_head_global,
                    kv_head_global=prefix_result.kv_head_global,
                    visible_length=prefix_result.visible_length,
                    query_position=trace.query_positions[record_index],
                    historical_tokens=prefix_result.historical_tokens,
                    recent_exact_tokens=prefix_result.recent_exact_tokens,
                    candidate_blocks=len(prefix_result.blocks),
                    certified_skipped_blocks=0,
                    decoded_blocks=len(step.decoded_block_starts),
                    decoded_block_starts=step.decoded_block_starts,
                    skipped_block_starts=tuple(),
                    certificate_name=step.chosen_certificate,
                    z_k_lower_text=step.z_k_lower_text,
                    z_k_lower_upper_float=step.z_k_lower_upper_float,
                    u_s_upper_text=step.u_s_upper_text,
                    u_s_upper_float=step.u_s_upper_float,
                    nu_s_upper_text=step.nu_s_upper_text,
                    nu_s_upper_float=step.nu_s_upper_float,
                    kept_output_norm_upper_text=step.kept_output_norm_upper_text,
                    kept_output_norm_upper_float=step.kept_output_norm_upper_float,
                    certificate_bound_text=step.certificate_value_text,
                    certificate_bound_upper_float=float(step.certificate_value_upper_float),
                    observed_skip_error=0.0,
                    rigorous_skip_error_upper_text="0.0",
                    rigorous_skip_error_upper_float=0.0,
                    bound_to_observed_ratio=None,
                    approximate_observed_violation=False,
                    rigorous_interval_violation=False,
                    numerical_fallback_used=step.numerical_fallback_used,
                    model_reference_gap=model_reference_gap,
                    compression_error=compression_error,
                    reference_total_error=reference_total_error,
                    captured_model_total_gap=captured_model_total_gap,
                    reference_decomposition_lhs=reference_total_error,
                    reference_decomposition_rhs=reference_decomposition_rhs,
                    model_relative_decomposition_lhs=captured_model_total_gap,
                    model_relative_decomposition_rhs=model_relative_decomposition_rhs,
                )
            )
            continue

        cache_key = step.decoded_block_starts
        if cache_key not in kept_state_cache:
            kept_keys, kept_values = _kept_from_decoded_block_starts(prefix_result, cache_key)
            kept_output_mpfr = tuple(
                exact_reference_output_mpfr(
                    query,
                    kept_keys,
                    kept_values,
                    precision=precision,
                    attention_scale=trace.scaling,
                )
            )
            kept_output = _mpfr_vector_to_numpy(kept_output_mpfr)
            observed_skip_error_mpfr = rigorous_output_error_norm(
                reconstructed_full_output_mpfr,
                kept_output_mpfr,
                precision=precision,
            )
            observed_skip_error = float(observed_skip_error_mpfr)
            if full_interval is None:
                full_interval, _ = rigorous_attention_output_interval(
                    query,
                    prefix_result.reconstructed_keys,
                    prefix_result.reconstructed_values,
                    precision=precision,
                    attention_scale=trace.scaling,
                )
            kept_interval, _ = rigorous_attention_output_interval(
                query,
                kept_keys,
                kept_values,
                precision=precision,
                attention_scale=trace.scaling,
            )
            kept_state_cache[cache_key] = (
                kept_output,
                kept_output_mpfr,
                observed_skip_error_mpfr,
                observed_skip_error,
                kept_interval,
            )

        kept_output, kept_output_mpfr, observed_skip_error_mpfr, observed_skip_error, kept_interval = kept_state_cache[cache_key]
        rigorous_skip_error_upper = rigorous_output_error_upper_from_intervals(full_interval, kept_interval, precision=precision)
        approximate_observed_violation, rigorous_interval_violation = _skip_validation_flags(
            certificate_value=step.certificate_value_mpfr,
            observed_skip_error_mpfr=observed_skip_error_mpfr,
            rigorous_skip_error_upper=rigorous_skip_error_upper,
        )
        if rigorous_interval_violation:
            raise AssertionError("Rigorous skipped-output interval upper bound exceeds the rigorous certificate.")

        reference_total_error = _norm(original_reference_output - kept_output)
        captured_model_total_gap = _norm(captured_model_output - kept_output)
        reference_decomposition_rhs = compression_error + observed_skip_error
        model_relative_decomposition_rhs = model_reference_gap + compression_error + observed_skip_error
        if reference_total_error > reference_decomposition_rhs + 1e-9:
            raise AssertionError("Reference decomposition failed numerically for the Stage 2 case.")
        if captured_model_total_gap > model_relative_decomposition_rhs + 1e-9:
            raise AssertionError("Model-relative decomposition failed numerically for the Stage 2 case.")

        bound_to_observed_ratio = None
        if observed_skip_error > 0.0:
            bound_to_observed_ratio = float(step.certificate_value_upper_float / observed_skip_error)

        results.append(
            QueryCaseResult(
                record_index=record_index,
                query_local_index=query_local_index,
                query_head_global=query_head_global,
                kv_head_global=prefix_result.kv_head_global,
                visible_length=prefix_result.visible_length,
                query_position=trace.query_positions[record_index],
                historical_tokens=prefix_result.historical_tokens,
                recent_exact_tokens=prefix_result.recent_exact_tokens,
                candidate_blocks=len(prefix_result.blocks),
                certified_skipped_blocks=len(step.skipped_block_starts),
                decoded_blocks=len(step.decoded_block_starts),
                decoded_block_starts=step.decoded_block_starts,
                skipped_block_starts=step.skipped_block_starts,
                certificate_name=step.chosen_certificate,
                z_k_lower_text=step.z_k_lower_text,
                z_k_lower_upper_float=step.z_k_lower_upper_float,
                u_s_upper_text=step.u_s_upper_text,
                u_s_upper_float=step.u_s_upper_float,
                nu_s_upper_text=step.nu_s_upper_text,
                nu_s_upper_float=step.nu_s_upper_float,
                kept_output_norm_upper_text=step.kept_output_norm_upper_text,
                kept_output_norm_upper_float=step.kept_output_norm_upper_float,
                certificate_bound_text=step.certificate_value_text,
                certificate_bound_upper_float=float(step.certificate_value_upper_float),
                observed_skip_error=observed_skip_error,
                rigorous_skip_error_upper_text=str(rigorous_skip_error_upper),
                rigorous_skip_error_upper_float=float(rigorous_skip_error_upper),
                bound_to_observed_ratio=bound_to_observed_ratio,
                approximate_observed_violation=approximate_observed_violation,
                rigorous_interval_violation=rigorous_interval_violation,
                numerical_fallback_used=step.numerical_fallback_used,
                model_reference_gap=model_reference_gap,
                compression_error=compression_error,
                reference_total_error=reference_total_error,
                captured_model_total_gap=captured_model_total_gap,
                reference_decomposition_lhs=reference_total_error,
                reference_decomposition_rhs=reference_decomposition_rhs,
                model_relative_decomposition_lhs=captured_model_total_gap,
                model_relative_decomposition_rhs=model_relative_decomposition_rhs,
            )
        )

    return tuple(results)


def _run_query_case(
    *,
    trace: ValidatedCompactTrace,
    prefix_result: PrefixCompressionResult,
    query_local_index: int,
    tolerance: float,
    precision: int,
    base_case: QueryCaseBase,
) -> QueryCaseResult:
    record_index = prefix_result.record_index
    query_head_global = trace.selected_query_heads[query_local_index]
    query = base_case.query
    captured_model_output = base_case.captured_model_output
    original_reference_output = base_case.original_reference_output
    reconstructed_full_output_mpfr = list(base_case.reconstructed_full_output_mpfr)
    reconstructed_full_output = base_case.reconstructed_full_output
    model_reference_gap = base_case.model_reference_gap
    compression_error = base_case.compression_error

    if not prefix_result.blocks:
        reference_total_error = _norm(original_reference_output - reconstructed_full_output)
        captured_model_total_gap = _norm(captured_model_output - reconstructed_full_output)
        reference_decomposition_rhs = compression_error
        model_relative_decomposition_rhs = model_reference_gap + compression_error
        if reference_total_error > reference_decomposition_rhs + 1e-9:
            raise AssertionError("Reference decomposition failed numerically in the zero-history case.")
        if captured_model_total_gap > model_relative_decomposition_rhs + 1e-9:
            raise AssertionError("Model-relative decomposition failed numerically in the zero-history case.")
        return QueryCaseResult(
            record_index=record_index,
            query_local_index=query_local_index,
            query_head_global=query_head_global,
            kv_head_global=prefix_result.kv_head_global,
            visible_length=prefix_result.visible_length,
            query_position=trace.query_positions[record_index],
            historical_tokens=prefix_result.historical_tokens,
            recent_exact_tokens=prefix_result.recent_exact_tokens,
            candidate_blocks=0,
            certified_skipped_blocks=0,
            decoded_blocks=0,
            decoded_block_starts=tuple(),
            skipped_block_starts=tuple(),
            certificate_name="none",
            z_k_lower_text=None,
            z_k_lower_upper_float=None,
            u_s_upper_text=None,
            u_s_upper_float=None,
            nu_s_upper_text=None,
            nu_s_upper_float=None,
            kept_output_norm_upper_text=None,
            kept_output_norm_upper_float=None,
            certificate_bound_text="0.0",
            certificate_bound_upper_float=0.0,
            observed_skip_error=0.0,
            rigorous_skip_error_upper_text="0.0",
            rigorous_skip_error_upper_float=0.0,
            bound_to_observed_ratio=None,
            approximate_observed_violation=False,
            rigorous_interval_violation=False,
            numerical_fallback_used=False,
            model_reference_gap=model_reference_gap,
            compression_error=compression_error,
            reference_total_error=reference_total_error,
            captured_model_total_gap=captured_model_total_gap,
            reference_decomposition_lhs=reference_total_error,
            reference_decomposition_rhs=reference_decomposition_rhs,
            model_relative_decomposition_lhs=captured_model_total_gap,
            model_relative_decomposition_rhs=model_relative_decomposition_rhs,
        )

    certification = certify_progressive_skipping(
        query=query,
        recent_keys=prefix_result.recent_keys,
        recent_values=prefix_result.recent_values,
        historical_blocks=prefix_result.blocks,
        tolerance=tolerance,
        mode=CertificateMode.RIGOROUS_REFERENCE,
        schedule=DecodeSchedule.LARGEST_U_TIMES_NU,
        precision=precision,
        attention_scale=trace.scaling,
    )
    if not certification.certified or certification.certificate_value_mpfr is None or certification.certificate_value_upper_float is None:
        raise AssertionError("Rigorous certification unexpectedly failed in the Stage 2 smoke experiment.")

    if not certification.skipped_block_starts:
        reference_total_error = compression_error
        captured_model_total_gap = _norm(captured_model_output - reconstructed_full_output)
        reference_decomposition_rhs = compression_error
        model_relative_decomposition_rhs = model_reference_gap + compression_error
        if reference_total_error > reference_decomposition_rhs + 1e-9:
            raise AssertionError("Reference decomposition failed numerically in the zero-skip case.")
        if captured_model_total_gap > model_relative_decomposition_rhs + 1e-9:
            raise AssertionError("Model-relative decomposition failed numerically in the zero-skip case.")
        return QueryCaseResult(
            record_index=record_index,
            query_local_index=query_local_index,
            query_head_global=query_head_global,
            kv_head_global=prefix_result.kv_head_global,
            visible_length=prefix_result.visible_length,
            query_position=trace.query_positions[record_index],
            historical_tokens=prefix_result.historical_tokens,
            recent_exact_tokens=prefix_result.recent_exact_tokens,
            candidate_blocks=len(prefix_result.blocks),
            certified_skipped_blocks=0,
            decoded_blocks=len(certification.decoded_block_starts),
            decoded_block_starts=certification.decoded_block_starts,
            skipped_block_starts=tuple(),
            certificate_name=certification.chosen_certificate,
            z_k_lower_text=certification.z_k_lower_text,
            z_k_lower_upper_float=certification.z_k_lower_upper_float,
            u_s_upper_text=certification.u_s_upper_text,
            u_s_upper_float=certification.u_s_upper_float,
            nu_s_upper_text=certification.nu_s_upper_text,
            nu_s_upper_float=certification.nu_s_upper_float,
            kept_output_norm_upper_text=certification.kept_output_norm_upper_text,
            kept_output_norm_upper_float=certification.kept_output_norm_upper_float,
            certificate_bound_text=certification.certificate_value_text or "0.0",
            certificate_bound_upper_float=float(certification.certificate_value_upper_float),
            observed_skip_error=0.0,
            rigorous_skip_error_upper_text="0.0",
            rigorous_skip_error_upper_float=0.0,
            bound_to_observed_ratio=None,
            approximate_observed_violation=False,
            rigorous_interval_violation=False,
            numerical_fallback_used=certification.numerical_fallback_used,
            model_reference_gap=model_reference_gap,
            compression_error=compression_error,
            reference_total_error=reference_total_error,
            captured_model_total_gap=captured_model_total_gap,
            reference_decomposition_lhs=reference_total_error,
            reference_decomposition_rhs=reference_decomposition_rhs,
            model_relative_decomposition_lhs=captured_model_total_gap,
            model_relative_decomposition_rhs=model_relative_decomposition_rhs,
        )

    kept_keys, kept_values = _kept_from_decoded_block_starts(prefix_result, certification.decoded_block_starts)
    kept_output_mpfr = exact_reference_output_mpfr(
        query,
        kept_keys,
        kept_values,
        precision=precision,
        attention_scale=trace.scaling,
    )
    kept_output = _mpfr_vector_to_numpy(kept_output_mpfr)
    observed_skip_error_mpfr = rigorous_output_error_norm(
        reconstructed_full_output_mpfr,
        kept_output_mpfr,
        precision=precision,
    )
    observed_skip_error = float(observed_skip_error_mpfr)
    full_interval = base_case.reconstructed_full_interval
    if full_interval is None:
        full_interval, _ = rigorous_attention_output_interval(
            query,
            prefix_result.reconstructed_keys,
            prefix_result.reconstructed_values,
            precision=precision,
            attention_scale=trace.scaling,
        )
    kept_interval, _ = rigorous_attention_output_interval(
        query,
        kept_keys,
        kept_values,
        precision=precision,
        attention_scale=trace.scaling,
    )
    rigorous_skip_error_upper = rigorous_output_error_upper_from_intervals(full_interval, kept_interval, precision=precision)
    approximate_observed_violation, rigorous_interval_violation = _skip_validation_flags(
        certificate_value=certification.certificate_value_mpfr,
        observed_skip_error_mpfr=observed_skip_error_mpfr,
        rigorous_skip_error_upper=rigorous_skip_error_upper,
    )
    if rigorous_interval_violation:
        raise AssertionError("Rigorous skipped-output interval upper bound exceeds the rigorous certificate.")

    reference_total_error = _norm(original_reference_output - kept_output)
    captured_model_total_gap = _norm(captured_model_output - kept_output)
    reference_decomposition_rhs = compression_error + observed_skip_error
    model_relative_decomposition_rhs = model_reference_gap + compression_error + observed_skip_error
    if reference_total_error > reference_decomposition_rhs + 1e-9:
        raise AssertionError("Reference decomposition failed numerically for the Stage 2 case.")
    if captured_model_total_gap > model_relative_decomposition_rhs + 1e-9:
        raise AssertionError("Model-relative decomposition failed numerically for the Stage 2 case.")
    bound_to_observed_ratio = None
    if observed_skip_error > 0.0:
        bound_to_observed_ratio = float(certification.certificate_value_upper_float / observed_skip_error)

    return QueryCaseResult(
        record_index=record_index,
        query_local_index=query_local_index,
        query_head_global=query_head_global,
        kv_head_global=prefix_result.kv_head_global,
        visible_length=prefix_result.visible_length,
        query_position=trace.query_positions[record_index],
        historical_tokens=prefix_result.historical_tokens,
        recent_exact_tokens=prefix_result.recent_exact_tokens,
        candidate_blocks=len(prefix_result.blocks),
        certified_skipped_blocks=len(certification.skipped_block_starts),
        decoded_blocks=len(certification.decoded_block_starts),
        decoded_block_starts=certification.decoded_block_starts,
        skipped_block_starts=certification.skipped_block_starts,
        certificate_name=certification.chosen_certificate,
        z_k_lower_text=certification.z_k_lower_text,
        z_k_lower_upper_float=certification.z_k_lower_upper_float,
        u_s_upper_text=certification.u_s_upper_text,
        u_s_upper_float=certification.u_s_upper_float,
        nu_s_upper_text=certification.nu_s_upper_text,
        nu_s_upper_float=certification.nu_s_upper_float,
        kept_output_norm_upper_text=certification.kept_output_norm_upper_text,
        kept_output_norm_upper_float=certification.kept_output_norm_upper_float,
        certificate_bound_text=certification.certificate_value_text or "0.0",
        certificate_bound_upper_float=float(certification.certificate_value_upper_float),
        observed_skip_error=observed_skip_error,
        rigorous_skip_error_upper_text=str(rigorous_skip_error_upper),
        rigorous_skip_error_upper_float=float(rigorous_skip_error_upper),
        bound_to_observed_ratio=bound_to_observed_ratio,
        approximate_observed_violation=approximate_observed_violation,
        rigorous_interval_violation=rigorous_interval_violation,
        numerical_fallback_used=certification.numerical_fallback_used,
        model_reference_gap=model_reference_gap,
        compression_error=compression_error,
        reference_total_error=reference_total_error,
        captured_model_total_gap=captured_model_total_gap,
        reference_decomposition_lhs=reference_total_error,
        reference_decomposition_rhs=reference_decomposition_rhs,
        model_relative_decomposition_lhs=captured_model_total_gap,
        model_relative_decomposition_rhs=model_relative_decomposition_rhs,
    )


def _build_query_case_base(
    *,
    trace: ValidatedCompactTrace,
    prefix_result: PrefixCompressionResult,
    query_local_index: int,
    precision: int,
) -> QueryCaseBase:
    record_index = prefix_result.record_index
    query = _float64_numpy(trace.queries[record_index, query_local_index, :])
    captured_model_output = _float64_numpy(trace.model_head_outputs[record_index, query_local_index, :])
    original_reference_output_mpfr = tuple(
        exact_reference_output_mpfr(
            query,
            prefix_result.original_keys,
            prefix_result.original_values,
            precision=precision,
            attention_scale=trace.scaling,
        )
    )
    original_reference_output = _mpfr_vector_to_numpy(original_reference_output_mpfr)
    reconstructed_full_output_mpfr = tuple(
        exact_reference_output_mpfr(
            query,
            prefix_result.reconstructed_keys,
            prefix_result.reconstructed_values,
            precision=precision,
            attention_scale=trace.scaling,
        )
    )
    reconstructed_full_output = _mpfr_vector_to_numpy(reconstructed_full_output_mpfr)
    model_reference_gap = _norm(captured_model_output - original_reference_output)
    compression_error = _norm(original_reference_output - reconstructed_full_output)
    reconstructed_full_interval = None
    if prefix_result.blocks:
        reconstructed_full_interval, _ = rigorous_attention_output_interval(
            query,
            prefix_result.reconstructed_keys,
            prefix_result.reconstructed_values,
            precision=precision,
            attention_scale=trace.scaling,
        )
    return QueryCaseBase(
        query=query,
        captured_model_output=captured_model_output,
        original_reference_output_mpfr=original_reference_output_mpfr,
        original_reference_output=original_reference_output,
        reconstructed_full_output_mpfr=reconstructed_full_output_mpfr,
        reconstructed_full_output=reconstructed_full_output,
        model_reference_gap=model_reference_gap,
        compression_error=compression_error,
        reconstructed_full_interval=reconstructed_full_interval,
    )


def run_stage2_smoke_experiment(
    *,
    trace_path: str | Path,
    recent_windows: Sequence[int] = (1, 2, 4),
    block_sizes: Sequence[int] = (2, 4),
    tolerances: Sequence[float] = (0.0, 0.01, 0.05, 0.1),
    precision: int = DEFAULT_PRECISION,
    random_seed: int = 0,
) -> Stage2SmokeResult:
    if precision <= 0:
        raise ValueError("precision must be positive.")
    trace = validate_compact_trace(trace_path)
    np.random.seed(random_seed)
    config_results: list[ExperimentConfigSummary] = []
    all_prefix_ratios: list[float] = []

    for recent_window in recent_windows:
        for block_size in block_sizes:
            prefix_cache: dict[tuple[int, int], PrefixCompressionResult] = {}
            for record_index in range(trace.query_records):
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

            prefix_values_for_aggregation = tuple(prefix_cache.values())
            case_bases = {
                (record_index, query_local_index): _build_query_case_base(
                    trace=trace,
                    prefix_result=prefix_cache[(record_index, trace.query_to_kv_heads[query_local_index])],
                    query_local_index=query_local_index,
                    precision=precision,
                )
                for record_index in range(trace.query_records)
                for query_local_index in range(trace.query_head_count)
            }
            unique_prefix_slices = len(prefix_values_for_aggregation)
            unique_prefix_slices_with_history = sum(1 for prefix in prefix_values_for_aggregation if prefix.historical_tokens > 0)
            aggregate_prefix_original_bytes = sum(prefix.original_kv_bytes for prefix in prefix_values_for_aggregation)
            aggregate_prefix_serialized_bytes = sum(prefix.total_compressed_bytes for prefix in prefix_values_for_aggregation)
            aggregate_prefix_historical_serialized_bytes = sum(
                prefix.compressed_historical_bytes for prefix in prefix_values_for_aggregation
            )
            total_historical_tokens = sum(prefix.historical_tokens for prefix in prefix_values_for_aggregation)
            prefix_compression_ratios = np.asarray(
                [prefix.compression_ratio for prefix in prefix_values_for_aggregation],
                dtype=np.float64,
            )
            all_prefix_ratios.extend(prefix_compression_ratios.tolist())
            key_diff_arrays = [prefix.reconstructed_keys - prefix.original_keys for prefix in prefix_values_for_aggregation]
            value_diff_arrays = [prefix.reconstructed_values - prefix.original_values for prefix in prefix_values_for_aggregation]
            combined_key_diff = np.concatenate([diff.reshape(-1) for diff in key_diff_arrays]) if key_diff_arrays else np.zeros((0,), dtype=np.float64)
            combined_value_diff = np.concatenate([diff.reshape(-1) for diff in value_diff_arrays]) if value_diff_arrays else np.zeros((0,), dtype=np.float64)
            final_prefixes = tuple(
                prefix_cache[(trace.query_records - 1, kv_head_global)]
                for kv_head_global in trace.selected_kv_heads
            )
            final_prefix_original_bytes = sum(prefix.original_kv_bytes for prefix in final_prefixes)
            final_prefix_serialized_bytes = sum(prefix.total_compressed_bytes for prefix in final_prefixes)
            final_prefix_historical_serialized_bytes = sum(prefix.compressed_historical_bytes for prefix in final_prefixes)
            final_prefix_compression_ratio = (
                float(final_prefix_original_bytes / final_prefix_serialized_bytes)
                if final_prefix_serialized_bytes > 0
                else 1.0
            )
            final_prefix_values_per_selected_kv_head = trace.visible_lengths[-1]
            final_prefix_total_values_selected_kv_heads = final_prefix_values_per_selected_kv_head * len(trace.selected_kv_heads)

            for tolerance in tolerances:
                case_results: list[QueryCaseResult] = []
                for record_index in range(trace.query_records):
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
                nonzero_ratios = [case.bound_to_observed_ratio for case in case_results if case.bound_to_observed_ratio is not None]
                config_results.append(
                    ExperimentConfigSummary(
                        config=ExperimentConfig(recent_window=recent_window, block_size=block_size, tolerance=float(tolerance)),
                        evaluated_query_cases=len(case_results),
                        eligible_query_cases=sum(1 for case in case_results if case.candidate_blocks > 0),
                        unique_prefix_slices=unique_prefix_slices,
                        unique_prefix_slices_with_history=unique_prefix_slices_with_history,
                        aggregate_prefix_original_bytes=aggregate_prefix_original_bytes,
                        aggregate_prefix_serialized_bytes=aggregate_prefix_serialized_bytes,
                        aggregate_prefix_historical_serialized_bytes=aggregate_prefix_historical_serialized_bytes,
                        aggregate_prefix_compression_ratio=(
                            float(aggregate_prefix_original_bytes / aggregate_prefix_serialized_bytes)
                            if aggregate_prefix_serialized_bytes > 0
                            else 1.0
                        ),
                        min_prefix_compression_ratio=float(np.min(prefix_compression_ratios)) if prefix_compression_ratios.size else 1.0,
                        mean_prefix_compression_ratio=float(np.mean(prefix_compression_ratios)) if prefix_compression_ratios.size else 1.0,
                        median_prefix_compression_ratio=float(np.median(prefix_compression_ratios)) if prefix_compression_ratios.size else 1.0,
                        max_prefix_compression_ratio=float(np.max(prefix_compression_ratios)) if prefix_compression_ratios.size else 1.0,
                        final_prefix_original_bytes=final_prefix_original_bytes,
                        final_prefix_serialized_bytes=final_prefix_serialized_bytes,
                        final_prefix_historical_serialized_bytes=final_prefix_historical_serialized_bytes,
                        final_prefix_compression_ratio=final_prefix_compression_ratio,
                        final_prefix_values_per_selected_kv_head=final_prefix_values_per_selected_kv_head,
                        final_prefix_total_values_selected_kv_heads=final_prefix_total_values_selected_kv_heads,
                        bytes_per_historical_token=(
                            float(aggregate_prefix_historical_serialized_bytes / total_historical_tokens)
                            if total_historical_tokens > 0
                            else 0.0
                        ),
                        max_key_abs_error=max((prefix.max_key_abs_error for prefix in prefix_values_for_aggregation), default=0.0),
                        key_rmse=_rmse(combined_key_diff),
                        max_value_abs_error=max((prefix.max_value_abs_error for prefix in prefix_values_for_aggregation), default=0.0),
                        value_rmse=_rmse(combined_value_diff),
                        number_of_blocks=sum(len(prefix.blocks) for prefix in prefix_values_for_aggregation),
                        independently_decoded_blocks_tested=sum(prefix.independent_blocks_tested for prefix in prefix_values_for_aggregation),
                        independent_versus_full_decode_max_difference=max(
                            (prefix.independent_decode_max_diff for prefix in prefix_values_for_aggregation),
                            default=0.0,
                        ),
                        unaffected_block_decoding_tests=sum(1 for prefix in prefix_values_for_aggregation if prefix.unaffected_block_decode_tested),
                        unaffected_block_decoding_passed=all(prefix.unaffected_block_decode_passed for prefix in prefix_values_for_aggregation),
                        serializer_roundtrip_checks_passed=all(
                            prefix.authoritative_serialized_roundtrip_used for prefix in prefix_values_for_aggregation
                        ),
                        total_candidate_blocks=candidate_blocks,
                        certified_skipped_blocks=certified_skipped_blocks,
                        decoded_blocks=decoded_blocks,
                        skipped_block_fraction=float(certified_skipped_blocks / candidate_blocks) if candidate_blocks > 0 else 0.0,
                        decoded_block_fraction=float(decoded_blocks / candidate_blocks) if candidate_blocks > 0 else 0.0,
                        max_certified_bound=max((case.certificate_bound_upper_float for case in case_results), default=0.0),
                        max_observed_reconstructed_skipping_error=max((case.observed_skip_error for case in case_results), default=0.0),
                        max_rigorous_skip_error_upper=max((case.rigorous_skip_error_upper_float for case in case_results), default=0.0),
                        min_bound_to_observed_ratio_nonzero=min(nonzero_ratios) if nonzero_ratios else None,
                        rigorous_interval_violation_count=sum(1 for case in case_results if case.rigorous_interval_violation),
                        approximate_observed_violation_count=sum(1 for case in case_results if case.approximate_observed_violation),
                        false_safe_violation_count=sum(1 for case in case_results if case.rigorous_interval_violation),
                        fallback_count=sum(1 for case in case_results if case.numerical_fallback_used),
                        max_model_reference_gap=max((case.model_reference_gap for case in case_results), default=0.0),
                        max_true_compression_error=max((case.compression_error for case in case_results), default=0.0),
                        max_reference_total_error=max((case.reference_total_error for case in case_results), default=0.0),
                        max_captured_model_total_gap=max((case.captured_model_total_gap for case in case_results), default=0.0),
                        reference_decomposition_violation_count=sum(
                            1
                            for case in case_results
                            if case.reference_decomposition_lhs > case.reference_decomposition_rhs + 1e-9
                        ),
                        model_relative_decomposition_violation_count=sum(
                            1
                            for case in case_results
                            if case.model_relative_decomposition_lhs > case.model_relative_decomposition_rhs + 1e-9
                        ),
                        query_case_results=tuple(case_results),
                    )
                )

    global_prefix_ratios = np.asarray(all_prefix_ratios, dtype=np.float64)
    return Stage2SmokeResult(
        trace=trace,
        precision=precision,
        random_seed=random_seed,
        global_prefix_ratio_min=float(np.min(global_prefix_ratios)) if global_prefix_ratios.size else 1.0,
        global_prefix_ratio_mean=float(np.mean(global_prefix_ratios)) if global_prefix_ratios.size else 1.0,
        global_prefix_ratio_median=float(np.median(global_prefix_ratios)) if global_prefix_ratios.size else 1.0,
        global_prefix_ratio_max=float(np.max(global_prefix_ratios)) if global_prefix_ratios.size else 1.0,
        configs=tuple(config_results),
    )


def result_to_dict(result: Stage2SmokeResult) -> dict[str, Any]:
    canonical_scale = _canonical_attention_scale(result.trace.head_dim)
    return {
        "scientific_scope": {
            "statement": "This is the first layer-0 integration experiment.",
            "trace_tokens": int(result.trace.sequence_length),
            "smoke_test_only": True,
            "external_baselines_run": False,
            "multi_layer_integration_run": False,
            "generation_integration_run": False,
            "certificate_scope": "The skipping certificate applies relative to reconstructed compressed KV.",
            "compression_error_scope": "Compression error remains empirical.",
            "novelty_scope": "No novelty conclusion follows from this experiment alone.",
        },
        "trace": {
            "trace_path": str(result.trace.trace_path),
            "trace_sha256": result.trace.trace_sha256,
            "trace_sha256_from_report": result.trace.trace_sha256_from_report,
            "checkpoint_repo": result.trace.checkpoint_repo,
            "checkpoint_revision": result.trace.checkpoint_revision,
            "selected_query_heads": list(result.trace.selected_query_heads),
            "selected_kv_heads": list(result.trace.selected_kv_heads),
            "query_to_kv_heads": list(result.trace.query_to_kv_heads),
            "visible_lengths": list(result.trace.visible_lengths),
            "query_positions": list(result.trace.query_positions),
            "sequence_length": result.trace.sequence_length,
            "head_dim": result.trace.head_dim,
            "scaling": result.trace.scaling,
            "canonical_one_over_sqrt_head_dim": canonical_scale,
            "scaling_minus_canonical": float(result.trace.scaling - canonical_scale),
            "source_dtype": result.trace.source_dtype,
            "storage_dtype": result.trace.storage_dtype,
        },
        "precision": result.precision,
        "random_seed": result.random_seed,
        "global_prefix_ratio_summary": {
            "minimum": result.global_prefix_ratio_min,
            "mean": result.global_prefix_ratio_mean,
            "median": result.global_prefix_ratio_median,
            "maximum": result.global_prefix_ratio_max,
        },
        "configurations": [
            {
                **asdict(config_result.config),
                "evaluated_query_cases": config_result.evaluated_query_cases,
                "eligible_query_cases": config_result.eligible_query_cases,
                "unique_prefix_slices": config_result.unique_prefix_slices,
                "unique_prefix_slices_with_history": config_result.unique_prefix_slices_with_history,
                "aggregate_prefix_original_bytes": config_result.aggregate_prefix_original_bytes,
                "aggregate_prefix_serialized_bytes": config_result.aggregate_prefix_serialized_bytes,
                "aggregate_prefix_historical_serialized_bytes": config_result.aggregate_prefix_historical_serialized_bytes,
                "aggregate_prefix_compression_ratio": config_result.aggregate_prefix_compression_ratio,
                "min_prefix_compression_ratio": config_result.min_prefix_compression_ratio,
                "mean_prefix_compression_ratio": config_result.mean_prefix_compression_ratio,
                "median_prefix_compression_ratio": config_result.median_prefix_compression_ratio,
                "max_prefix_compression_ratio": config_result.max_prefix_compression_ratio,
                "final_prefix_original_bytes": config_result.final_prefix_original_bytes,
                "final_prefix_serialized_bytes": config_result.final_prefix_serialized_bytes,
                "final_prefix_historical_serialized_bytes": config_result.final_prefix_historical_serialized_bytes,
                "final_prefix_compression_ratio": config_result.final_prefix_compression_ratio,
                "final_prefix_values_per_selected_kv_head": config_result.final_prefix_values_per_selected_kv_head,
                "final_prefix_total_values_selected_kv_heads": config_result.final_prefix_total_values_selected_kv_heads,
                "bytes_per_historical_token": config_result.bytes_per_historical_token,
                "max_key_abs_error": config_result.max_key_abs_error,
                "key_rmse": config_result.key_rmse,
                "max_value_abs_error": config_result.max_value_abs_error,
                "value_rmse": config_result.value_rmse,
                "number_of_blocks": config_result.number_of_blocks,
                "independently_decoded_blocks_tested": config_result.independently_decoded_blocks_tested,
                "independent_versus_full_decode_max_difference": config_result.independent_versus_full_decode_max_difference,
                "unaffected_block_decoding_tests": config_result.unaffected_block_decoding_tests,
                "unaffected_block_decoding_passed": config_result.unaffected_block_decoding_passed,
                "serializer_roundtrip_checks_passed": config_result.serializer_roundtrip_checks_passed,
                "total_candidate_blocks": config_result.total_candidate_blocks,
                "certified_skipped_blocks": config_result.certified_skipped_blocks,
                "decoded_blocks": config_result.decoded_blocks,
                "skipped_block_fraction": config_result.skipped_block_fraction,
                "decoded_block_fraction": config_result.decoded_block_fraction,
                "max_certified_bound": config_result.max_certified_bound,
                "max_observed_reconstructed_skipping_error": config_result.max_observed_reconstructed_skipping_error,
                "max_rigorous_skip_error_upper": config_result.max_rigorous_skip_error_upper,
                "min_bound_to_observed_ratio_nonzero": config_result.min_bound_to_observed_ratio_nonzero,
                "rigorous_interval_violation_count": config_result.rigorous_interval_violation_count,
                "approximate_observed_violation_count": config_result.approximate_observed_violation_count,
                "false_safe_violation_count": config_result.false_safe_violation_count,
                "fallback_count": config_result.fallback_count,
                "max_model_reference_gap": config_result.max_model_reference_gap,
                "max_true_compression_error": config_result.max_true_compression_error,
                "max_reference_total_error": config_result.max_reference_total_error,
                "max_captured_model_total_gap": config_result.max_captured_model_total_gap,
                "reference_decomposition_violation_count": config_result.reference_decomposition_violation_count,
                "model_relative_decomposition_violation_count": config_result.model_relative_decomposition_violation_count,
                "query_case_results": [
                    {
                        **asdict(case),
                    }
                    for case in config_result.query_case_results
                ],
            }
            for config_result in result.configs
        ],
    }
