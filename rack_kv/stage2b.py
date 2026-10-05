from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .stage2 import (
    PrefixCompressionResult,
    QueryCaseResult,
    _build_prefix_result,
    _build_query_case_base,
    _canonical_attention_scale,
    _run_query_case_tolerance_bundle,
    validate_compact_trace,
)


STAGE2B_SEQUENCE_LENGTH = 256


@dataclass(frozen=True)
class Stage2BConfig:
    recent_window: int
    block_size: int
    tolerance: float


@dataclass(frozen=True)
class Stage2BMemorySummary:
    recent_window: int
    block_size: int
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
    number_of_blocks: int
    independently_decoded_blocks_tested: int
    independent_versus_full_decode_max_difference: float
    unaffected_block_decoding_tests: int
    unaffected_block_decoding_passed: bool
    serializer_roundtrip_checks_passed: bool
    max_key_abs_error: float
    key_rmse: float
    max_value_abs_error: float
    value_rmse: float


@dataclass(frozen=True)
class Stage2BConfigSummary:
    config: Stage2BConfig
    evaluated_query_cases: int
    eligible_query_cases: int
    ineligible_query_cases: int
    total_candidate_blocks: int
    certified_skipped_blocks: int
    decoded_blocks: int
    weighted_skipped_block_fraction: float
    weighted_decoded_block_fraction: float
    mean_case_skipped_fraction: float
    median_case_skipped_fraction: float
    max_case_skipped_fraction: float
    fraction_cases_with_any_skipped_block: float
    max_certified_bound: float
    max_observed_reconstructed_skipping_error: float
    max_rigorous_skip_error_upper: float
    min_bound_to_observed_ratio_nonzero: float | None
    max_bound_to_observed_ratio_nonzero: float | None
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
class Stage2BResult:
    trace_path: Path
    trace_sha256: str
    trace_sequence_length: int
    trace_query_shape: tuple[int, ...]
    trace_final_key_shape: tuple[int, ...]
    trace_final_value_shape: tuple[int, ...]
    trace_model_head_output_shape: tuple[int, ...]
    trace_selected_query_heads: tuple[int, ...]
    trace_selected_kv_heads: tuple[int, ...]
    trace_query_to_kv_heads: tuple[int, ...]
    trace_visible_lengths: tuple[int, ...]
    trace_query_positions: tuple[int, ...]
    trace_scaling: float
    trace_canonical_scaling: float
    evaluated_query_positions: tuple[int, ...]
    precision: int
    random_seed: int
    global_prefix_ratio_min: float
    global_prefix_ratio_mean: float
    global_prefix_ratio_median: float
    global_prefix_ratio_max: float
    memory_summaries: tuple[Stage2BMemorySummary, ...]
    config_summaries: tuple[Stage2BConfigSummary, ...]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rmse(diff: np.ndarray) -> float:
    if diff.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(diff, dtype=np.float64), dtype=np.float64)))


def select_stage2b_query_positions(sequence_length: int) -> tuple[int, ...]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    positions = set(range(31, sequence_length, 16))
    positions.update(range(248, sequence_length))
    return tuple(sorted(position for position in positions if 0 <= position < sequence_length))


def _per_case_skipped_fraction(case: QueryCaseResult) -> float:
    if case.candidate_blocks <= 0:
        return 0.0
    return float(case.certified_skipped_blocks / case.candidate_blocks)


def _build_memory_summary(
    *,
    trace: Any,
    prefix_values_for_aggregation: tuple[PrefixCompressionResult, ...],
    recent_window: int,
    block_size: int,
) -> Stage2BMemorySummary:
    unique_prefix_slices = len(prefix_values_for_aggregation)
    unique_prefix_slices_with_history = sum(1 for prefix in prefix_values_for_aggregation if prefix.historical_tokens > 0)
    aggregate_prefix_original_bytes = sum(prefix.original_kv_bytes for prefix in prefix_values_for_aggregation)
    aggregate_prefix_serialized_bytes = sum(prefix.total_compressed_bytes for prefix in prefix_values_for_aggregation)
    aggregate_prefix_historical_serialized_bytes = sum(prefix.compressed_historical_bytes for prefix in prefix_values_for_aggregation)
    total_historical_tokens = sum(prefix.historical_tokens for prefix in prefix_values_for_aggregation)
    prefix_compression_ratios = np.asarray(
        [prefix.compression_ratio for prefix in prefix_values_for_aggregation],
        dtype=np.float64,
    )
    key_diff_arrays = [prefix.reconstructed_keys - prefix.original_keys for prefix in prefix_values_for_aggregation]
    value_diff_arrays = [prefix.reconstructed_values - prefix.original_values for prefix in prefix_values_for_aggregation]
    combined_key_diff = np.concatenate([diff.reshape(-1) for diff in key_diff_arrays]) if key_diff_arrays else np.zeros((0,), dtype=np.float64)
    combined_value_diff = np.concatenate([diff.reshape(-1) for diff in value_diff_arrays]) if value_diff_arrays else np.zeros((0,), dtype=np.float64)
    final_prefixes = tuple(
        prefix
        for prefix in prefix_values_for_aggregation
        if prefix.record_index == trace.query_records - 1
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
    return Stage2BMemorySummary(
        recent_window=recent_window,
        block_size=block_size,
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
        number_of_blocks=sum(len(prefix.blocks) for prefix in prefix_values_for_aggregation),
        independently_decoded_blocks_tested=sum(prefix.independent_blocks_tested for prefix in prefix_values_for_aggregation),
        independent_versus_full_decode_max_difference=max(
            (prefix.independent_decode_max_diff for prefix in prefix_values_for_aggregation),
            default=0.0,
        ),
        unaffected_block_decoding_tests=sum(1 for prefix in prefix_values_for_aggregation if prefix.unaffected_block_decode_tested),
        unaffected_block_decoding_passed=all(prefix.unaffected_block_decode_passed for prefix in prefix_values_for_aggregation),
        serializer_roundtrip_checks_passed=all(prefix.authoritative_serialized_roundtrip_used for prefix in prefix_values_for_aggregation),
        max_key_abs_error=max((prefix.max_key_abs_error for prefix in prefix_values_for_aggregation), default=0.0),
        key_rmse=_rmse(combined_key_diff),
        max_value_abs_error=max((prefix.max_value_abs_error for prefix in prefix_values_for_aggregation), default=0.0),
        value_rmse=_rmse(combined_value_diff),
    )


def _evaluate_stage2b_wb_pair(
    *,
    trace: Any,
    selected_positions: Sequence[int],
    recent_window: int,
    block_size: int,
    tolerances: Sequence[float],
    precision: int,
) -> tuple[Stage2BMemorySummary, tuple[Stage2BConfigSummary, ...], tuple[float, ...]]:
    prefix_cache: dict[tuple[int, int], PrefixCompressionResult] = {}
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
    prefix_values_for_aggregation = tuple(prefix_cache.values())
    if not prefix_values_for_aggregation:
        raise ValueError("Stage 2B prefix cache is empty.")
    memory_summary = _build_memory_summary(
        trace=trace,
        prefix_values_for_aggregation=prefix_values_for_aggregation,
        recent_window=recent_window,
        block_size=block_size,
    )
    prefix_ratios = tuple(prefix.compression_ratio for prefix in prefix_values_for_aggregation)
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
    case_results_by_tolerance_index: list[list[QueryCaseResult]] = [[] for _ in tolerances]

    for record_index in selected_positions:
        for query_local_index in range(trace.query_head_count):
            kv_head_global = trace.query_to_kv_heads[query_local_index]
            prefix_result = prefix_cache[(record_index, kv_head_global)]
            bundled_results = _run_query_case_tolerance_bundle(
                trace=trace,
                prefix_result=prefix_result,
                query_local_index=query_local_index,
                tolerances=tolerances,
                precision=precision,
                base_case=case_bases[(record_index, query_local_index)],
            )
            if len(bundled_results) != len(tolerances):
                raise AssertionError("Stage 2B bundled query-case evaluation returned an unexpected tolerance count.")
            for tolerance_index, case_result in enumerate(bundled_results):
                case_results_by_tolerance_index[tolerance_index].append(case_result)

    config_summaries: list[Stage2BConfigSummary] = []
    for tolerance, case_results in zip(tolerances, case_results_by_tolerance_index):
        candidate_blocks = sum(case.candidate_blocks for case in case_results)
        certified_skipped_blocks = sum(case.certified_skipped_blocks for case in case_results)
        decoded_blocks = sum(case.decoded_blocks for case in case_results)
        eligible_cases = [case for case in case_results if case.candidate_blocks > 0]
        skipped_case_fractions = np.asarray(
            [_per_case_skipped_fraction(case) for case in eligible_cases],
            dtype=np.float64,
        )
        nonzero_ratios = [case.bound_to_observed_ratio for case in case_results if case.bound_to_observed_ratio is not None]
        config_summaries.append(
            Stage2BConfigSummary(
                config=Stage2BConfig(
                    recent_window=recent_window,
                    block_size=block_size,
                    tolerance=float(tolerance),
                ),
                evaluated_query_cases=len(case_results),
                eligible_query_cases=len(eligible_cases),
                ineligible_query_cases=len(case_results) - len(eligible_cases),
                total_candidate_blocks=candidate_blocks,
                certified_skipped_blocks=certified_skipped_blocks,
                decoded_blocks=decoded_blocks,
                weighted_skipped_block_fraction=float(certified_skipped_blocks / candidate_blocks) if candidate_blocks > 0 else 0.0,
                weighted_decoded_block_fraction=float(decoded_blocks / candidate_blocks) if candidate_blocks > 0 else 0.0,
                mean_case_skipped_fraction=float(np.mean(skipped_case_fractions)) if skipped_case_fractions.size else 0.0,
                median_case_skipped_fraction=float(np.median(skipped_case_fractions)) if skipped_case_fractions.size else 0.0,
                max_case_skipped_fraction=float(np.max(skipped_case_fractions)) if skipped_case_fractions.size else 0.0,
                fraction_cases_with_any_skipped_block=(
                    float(sum(1 for case in eligible_cases if case.certified_skipped_blocks > 0) / len(eligible_cases))
                    if eligible_cases
                    else 0.0
                ),
                max_certified_bound=max((case.certificate_bound_upper_float for case in case_results), default=0.0),
                max_observed_reconstructed_skipping_error=max((case.observed_skip_error for case in case_results), default=0.0),
                max_rigorous_skip_error_upper=max((case.rigorous_skip_error_upper_float for case in case_results), default=0.0),
                min_bound_to_observed_ratio_nonzero=min(nonzero_ratios) if nonzero_ratios else None,
                max_bound_to_observed_ratio_nonzero=max(nonzero_ratios) if nonzero_ratios else None,
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
    return memory_summary, tuple(config_summaries), prefix_ratios


def run_stage2b_pilot(
    *,
    trace_path: str | Path,
    recent_windows: Sequence[int] = (16, 32),
    block_sizes: Sequence[int] = (8, 16, 32),
    tolerances: Sequence[float] = (0.0, 0.01, 0.05, 0.1),
    precision: int = 256,
    random_seed: int = 0,
    evaluated_query_positions: Sequence[int] | None = None,
) -> Stage2BResult:
    if precision <= 0:
        raise ValueError("precision must be positive.")
    trace = validate_compact_trace(trace_path)
    if trace.sequence_length != STAGE2B_SEQUENCE_LENGTH:
        raise ValueError(
            f"Stage 2B requires an exact {STAGE2B_SEQUENCE_LENGTH}-token compact trace; got {trace.sequence_length}."
        )
    if trace.query_records != STAGE2B_SEQUENCE_LENGTH:
        raise ValueError(
            f"Stage 2B requires {STAGE2B_SEQUENCE_LENGTH} query records; got {trace.query_records}."
        )

    np.random.seed(random_seed)
    if evaluated_query_positions is None:
        selected_positions = select_stage2b_query_positions(trace.sequence_length)
    else:
        selected_positions = tuple(
            sorted({int(position) for position in evaluated_query_positions})
        )
        if any(position < 0 or position >= trace.sequence_length for position in selected_positions):
            raise ValueError("evaluated_query_positions must stay within the trace sequence length.")
    if not selected_positions:
        raise ValueError("Stage 2B produced no evaluated query positions.")

    all_prefix_ratios: list[float] = []
    memory_summaries: list[Stage2BMemorySummary] = []
    config_summaries: list[Stage2BConfigSummary] = []

    for recent_window in recent_windows:
        for block_size in block_sizes:
            memory_summary, wb_config_summaries, prefix_ratios = _evaluate_stage2b_wb_pair(
                trace=trace,
                selected_positions=selected_positions,
                recent_window=recent_window,
                block_size=block_size,
                tolerances=tolerances,
                precision=precision,
            )
            memory_summaries.append(memory_summary)
            all_prefix_ratios.extend(prefix_ratios)
            config_summaries.extend(wb_config_summaries)

    global_prefix_ratios = np.asarray(all_prefix_ratios, dtype=np.float64)
    canonical_scaling = _canonical_attention_scale(trace.head_dim)
    return Stage2BResult(
        trace_path=trace.trace_path,
        trace_sha256=trace.trace_sha256,
        trace_sequence_length=trace.sequence_length,
        trace_query_shape=tuple(int(dim) for dim in trace.queries.shape),
        trace_final_key_shape=tuple(int(dim) for dim in trace.final_keys.shape),
        trace_final_value_shape=tuple(int(dim) for dim in trace.final_values.shape),
        trace_model_head_output_shape=tuple(int(dim) for dim in trace.model_head_outputs.shape),
        trace_selected_query_heads=trace.selected_query_heads,
        trace_selected_kv_heads=trace.selected_kv_heads,
        trace_query_to_kv_heads=trace.query_to_kv_heads,
        trace_visible_lengths=trace.visible_lengths,
        trace_query_positions=trace.query_positions,
        trace_scaling=trace.scaling,
        trace_canonical_scaling=canonical_scaling,
        evaluated_query_positions=selected_positions,
        precision=precision,
        random_seed=random_seed,
        global_prefix_ratio_min=float(np.min(global_prefix_ratios)) if global_prefix_ratios.size else 1.0,
        global_prefix_ratio_mean=float(np.mean(global_prefix_ratios)) if global_prefix_ratios.size else 1.0,
        global_prefix_ratio_median=float(np.median(global_prefix_ratios)) if global_prefix_ratios.size else 1.0,
        global_prefix_ratio_max=float(np.max(global_prefix_ratios)) if global_prefix_ratios.size else 1.0,
        memory_summaries=tuple(memory_summaries),
        config_summaries=tuple(config_summaries),
    )


def result_to_dict(result: Stage2BResult) -> dict[str, Any]:
    return {
        "scientific_scope": {
            "statement": "This is a controlled 256-token layer-0 pilot.",
            "trace_tokens": int(result.trace_sequence_length),
            "long_context_benchmark": False,
            "selected_heads_only": True,
            "single_deterministic_prompt": True,
            "external_baselines_run": False,
            "multi_layer_integration_run": False,
            "generation_integration_run": False,
            "certificate_scope": "Only skipping relative to reconstructed compressed KV is certified.",
            "compression_error_scope": "Compression error remains empirical.",
            "novelty_scope": "No novelty conclusion follows from this experiment alone.",
        },
        "trace": {
            "trace_path": str(result.trace_path),
            "trace_sha256": result.trace_sha256,
            "sequence_length": result.trace_sequence_length,
            "queries_shape": list(result.trace_query_shape),
            "final_keys_shape": list(result.trace_final_key_shape),
            "final_values_shape": list(result.trace_final_value_shape),
            "model_head_outputs_shape": list(result.trace_model_head_output_shape),
            "selected_query_heads": list(result.trace_selected_query_heads),
            "selected_kv_heads": list(result.trace_selected_kv_heads),
            "query_to_kv_heads": list(result.trace_query_to_kv_heads),
            "visible_lengths": list(result.trace_visible_lengths),
            "query_positions": list(result.trace_query_positions),
            "evaluated_query_positions": list(result.evaluated_query_positions),
            "scaling": result.trace_scaling,
            "canonical_one_over_sqrt_head_dim": result.trace_canonical_scaling,
            "scaling_minus_canonical": result.trace_scaling - result.trace_canonical_scaling,
        },
        "precision": result.precision,
        "random_seed": result.random_seed,
        "global_prefix_ratio_summary": {
            "minimum": result.global_prefix_ratio_min,
            "mean": result.global_prefix_ratio_mean,
            "median": result.global_prefix_ratio_median,
            "maximum": result.global_prefix_ratio_max,
        },
        "memory_by_wb": [asdict(summary) for summary in result.memory_summaries],
        "configurations": [
            {
                **asdict(summary.config),
                "evaluated_query_cases": summary.evaluated_query_cases,
                "eligible_query_cases": summary.eligible_query_cases,
                "ineligible_query_cases": summary.ineligible_query_cases,
                "total_candidate_blocks": summary.total_candidate_blocks,
                "certified_skipped_blocks": summary.certified_skipped_blocks,
                "decoded_blocks": summary.decoded_blocks,
                "weighted_skipped_block_fraction": summary.weighted_skipped_block_fraction,
                "weighted_decoded_block_fraction": summary.weighted_decoded_block_fraction,
                "mean_case_skipped_fraction": summary.mean_case_skipped_fraction,
                "median_case_skipped_fraction": summary.median_case_skipped_fraction,
                "max_case_skipped_fraction": summary.max_case_skipped_fraction,
                "fraction_cases_with_any_skipped_block": summary.fraction_cases_with_any_skipped_block,
                "max_certified_bound": summary.max_certified_bound,
                "max_observed_reconstructed_skipping_error": summary.max_observed_reconstructed_skipping_error,
                "max_rigorous_skip_error_upper": summary.max_rigorous_skip_error_upper,
                "min_bound_to_observed_ratio_nonzero": summary.min_bound_to_observed_ratio_nonzero,
                "max_bound_to_observed_ratio_nonzero": summary.max_bound_to_observed_ratio_nonzero,
                "rigorous_interval_violation_count": summary.rigorous_interval_violation_count,
                "approximate_observed_violation_count": summary.approximate_observed_violation_count,
                "false_safe_violation_count": summary.false_safe_violation_count,
                "fallback_count": summary.fallback_count,
                "max_model_reference_gap": summary.max_model_reference_gap,
                "max_true_compression_error": summary.max_true_compression_error,
                "max_reference_total_error": summary.max_reference_total_error,
                "max_captured_model_total_gap": summary.max_captured_model_total_gap,
                "reference_decomposition_violation_count": summary.reference_decomposition_violation_count,
                "model_relative_decomposition_violation_count": summary.model_relative_decomposition_violation_count,
                "query_case_results": [asdict(case) for case in summary.query_case_results],
            }
            for summary in result.config_summaries
        ],
    }
