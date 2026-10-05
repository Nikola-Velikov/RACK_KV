from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .stage2 import (
    QueryCaseResult,
    _build_prefix_result,
    _build_query_case_base,
    _run_query_case_tolerance_bundle,
    validate_compact_trace,
)


@dataclass(frozen=True)
class Stage3ReducedLayerCaseResult:
    trace_path: Path
    trace_sha256: str
    layer_index: int
    record_index: int
    query_local_index: int
    query_head_global: int
    kv_head_global: int
    recent_window: int
    block_size: int
    precision: int
    tolerances: tuple[float, ...]
    results: tuple[QueryCaseResult, ...]


def run_stage3_reduced_layer_case(
    *,
    trace_path: str | Path,
    recent_window: int,
    block_size: int,
    tolerances: Sequence[float],
    precision: int,
    record_index: int | None = None,
    query_local_index: int = 0,
) -> Stage3ReducedLayerCaseResult:
    if precision <= 0:
        raise ValueError("precision must be positive.")
    if recent_window <= 0:
        raise ValueError("recent_window must be positive.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    tolerance_tuple = tuple(float(value) for value in tolerances)
    if not tolerance_tuple:
        raise ValueError("tolerances must be non-empty.")

    trace = validate_compact_trace(trace_path, allow_nonzero_layer=True)
    if record_index is None:
        record_index = trace.query_records - 1
    record_index = int(record_index)
    query_local_index = int(query_local_index)
    if record_index < 0 or record_index >= trace.query_records:
        raise IndexError("record_index is outside the trace query-record range.")
    if query_local_index < 0 or query_local_index >= trace.query_head_count:
        raise IndexError("query_local_index is outside the selected query-head range.")

    prefix_keys, prefix_values = trace.query_case_prefix(
        record_index=record_index,
        query_local_index=query_local_index,
    )
    kv_head_global = trace.query_to_kv_heads[query_local_index]
    prefix_result = _build_prefix_result(
        record_index=record_index,
        kv_head_global=kv_head_global,
        prefix_keys=prefix_keys,
        prefix_values=prefix_values,
        recent_window=recent_window,
        block_size=block_size,
        precision=precision,
    )
    base_case = _build_query_case_base(
        trace=trace,
        prefix_result=prefix_result,
        query_local_index=query_local_index,
        precision=precision,
    )
    results = _run_query_case_tolerance_bundle(
        trace=trace,
        prefix_result=prefix_result,
        query_local_index=query_local_index,
        tolerances=tolerance_tuple,
        precision=precision,
        base_case=base_case,
    )
    return Stage3ReducedLayerCaseResult(
        trace_path=Path(trace_path),
        trace_sha256=trace.trace_sha256,
        layer_index=trace.layer_index,
        record_index=record_index,
        query_local_index=query_local_index,
        query_head_global=trace.selected_query_heads[query_local_index],
        kv_head_global=kv_head_global,
        recent_window=recent_window,
        block_size=block_size,
        precision=precision,
        tolerances=tolerance_tuple,
        results=tuple(results),
    )


def result_to_dict(result: Stage3ReducedLayerCaseResult) -> dict[str, Any]:
    return {
        "trace_path": str(result.trace_path),
        "trace_sha256": result.trace_sha256,
        "layer_index": result.layer_index,
        "record_index": result.record_index,
        "query_local_index": result.query_local_index,
        "query_head_global": result.query_head_global,
        "kv_head_global": result.kv_head_global,
        "recent_window": result.recent_window,
        "block_size": result.block_size,
        "precision": result.precision,
        "tolerances": list(result.tolerances),
        "results": [asdict(case) for case in result.results],
    }
