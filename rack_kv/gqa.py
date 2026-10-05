"""Rigorous GQA-group physical-eligibility certification for RACK-KV V2.

This module does not skip I/O or alter attention execution.  It identifies the
regions that a later physical implementation may omit because every query head
sharing a KV head has accepted the same region under its own cumulative MPFR
budget.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Mapping, Sequence

import gmpy2
import numpy as np

from .hierarchy import HierarchyIndex, _global_bound, validate_hierarchy
from .llama_trace import query_head_to_kv_head
from .rigorous import DEFAULT_PRECISION, exact_mpfr, exact_vector, rounded_add, rounded_mul, rounded_sqrt


def gqa_groups(*, num_attention_heads: int, num_key_value_heads: int) -> tuple[tuple[int, ...], ...]:
    """Return query-head groups using the configured model's GQA mapping."""
    if num_attention_heads <= 0 or num_key_value_heads <= 0:
        raise ValueError("head counts must be positive.")
    groups = [[] for _ in range(num_key_value_heads)]
    for head in range(num_attention_heads):
        groups[query_head_to_kv_head(head, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads)].append(head)
    return tuple(tuple(group) for group in groups)


@dataclass(frozen=True)
class GQAHeadDecision:
    query_head: int
    bound_before: gmpy2.mpfr
    candidate_bound_after: gmpy2.mpfr
    certificate_name: str
    passed: bool
    rigorous: bool = True


@dataclass(frozen=True)
class GQARegionDecision:
    kv_head: int
    region_id: int
    tree_level: int
    token_start: int
    token_end: int
    mapped_query_heads: tuple[int, ...]
    head_decisions: tuple[GQAHeadDecision, ...]
    all_heads_passed: bool
    gqa_skip_vote_count: int
    b_concat: gmpy2.mpfr | None
    output_projection_norm_upper: gmpy2.mpfr | None
    b_group: gmpy2.mpfr | None


@dataclass(frozen=True)
class GQATraversalResult:
    kv_head: int
    mapped_query_heads: tuple[int, ...]
    physical_eligible_regions: tuple[int, ...]
    physical_eligible_leaf_blocks: tuple[int, ...]
    required_leaf_blocks: tuple[int, ...]
    region_decisions: tuple[GQARegionDecision, ...]
    logical_head_skips: Mapping[int, tuple[int, ...]]
    nodes_considered: int
    nodes_certified: int
    nodes_descended: int
    potential_payload_bytes_avoided: int
    numerical_fallback_used: bool


def output_projection_group_columns(
    output_projection: np.ndarray,
    *,
    query_heads: Sequence[int],
    head_dim: int,
) -> np.ndarray:
    """Extract W_O columns that multiply the concatenated selected heads."""
    matrix = np.asarray(output_projection, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("output projection must be rank-2 [output, input].")
    columns = np.concatenate([np.arange(head * head_dim, (head + 1) * head_dim) for head in query_heads])
    if columns.size == 0 or columns[-1] >= matrix.shape[1]:
        raise ValueError("output projection does not contain the requested head columns.")
    return matrix[:, columns]


def output_projection_norm_upper_bound(matrix: np.ndarray, *, precision: int = DEFAULT_PRECISION) -> gmpy2.mpfr:
    """Outward-safe Frobenius upper bound, replaceable by a later tighter API."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("projection matrix must be finite and rank-2.")
    squared = exact_mpfr(0, precision=precision)
    for value in values.ravel(order="C"):
        exact = exact_mpfr(value, precision=precision)
        squared = rounded_add(squared, rounded_mul(exact, exact, precision=precision, round_mode=gmpy2.RoundUp), precision=precision, round_mode=gmpy2.RoundUp)
    return rounded_sqrt(squared, precision=precision, round_mode=gmpy2.RoundUp)


def concatenation_bound(head_bounds: Sequence[gmpy2.mpfr], *, precision: int = DEFAULT_PRECISION) -> gmpy2.mpfr:
    if not head_bounds:
        raise ValueError("at least one per-head bound is required.")
    squared = exact_mpfr(0, precision=precision)
    for bound in head_bounds:
        if bound < 0:
            raise ValueError("per-head bounds must be nonnegative.")
        squared = rounded_add(squared, rounded_mul(bound, bound, precision=precision, round_mode=gmpy2.RoundUp), precision=precision, round_mode=gmpy2.RoundUp)
    return rounded_sqrt(squared, precision=precision, round_mode=gmpy2.RoundUp)


def projected_group_bound(head_bounds: Sequence[gmpy2.mpfr], projection_group: np.ndarray, *, precision: int = DEFAULT_PRECISION) -> tuple[gmpy2.mpfr, gmpy2.mpfr]:
    """Return (B_concat, ||W_O,G||_F B_concat), both outward-safe."""
    concat = concatenation_bound(head_bounds, precision=precision)
    norm_upper = output_projection_norm_upper_bound(projection_group, precision=precision)
    return concat, rounded_mul(norm_upper, concat, precision=precision, round_mode=gmpy2.RoundUp)


def certify_gqa_hierarchical_group(
    *,
    kv_head: int,
    query_by_head: Mapping[int, np.ndarray],
    recent_keys_by_head: Mapping[int, np.ndarray],
    recent_values_by_head: Mapping[int, np.ndarray],
    index: HierarchyIndex,
    leaf_keys: Sequence[np.ndarray],
    leaf_values: Sequence[np.ndarray],
    epsilon_head: float = 0.05,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
    output_projection: np.ndarray | None = None,
    num_attention_heads: int = 32,
    num_key_value_heads: int = 8,
    head_dim: int | None = None,
    leaf_payload_bytes: Mapping[int, int] | None = None,
) -> GQATraversalResult:
    """Certify shared regions only under unanimous cumulative MPFR approval."""
    validate_hierarchy(index)
    groups = gqa_groups(num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads)
    if kv_head < 0 or kv_head >= len(groups):
        raise IndexError("kv_head is outside the configured GQA groups.")
    heads = groups[kv_head]
    if set(query_by_head) != set(heads) or set(recent_keys_by_head) != set(heads) or set(recent_values_by_head) != set(heads):
        raise ValueError("GQA certification requires a query and recent cache for every mapped query head.")
    query_exact = {head: exact_vector(np.asarray(query_by_head[head], dtype=np.float64), precision=precision) for head in heads}
    tolerance = exact_mpfr(epsilon_head, precision=precision)
    queue = [(-index.nodes[index.root_id].token_count, index.nodes[index.root_id].token_start, index.root_id)]
    committed: list[int] = []
    decisions: list[GQARegionDecision] = []
    logical_skips = {head: [] for head in heads}
    considered = certified = descended = 0
    fallback = False
    projection_group = None
    if output_projection is not None:
        if head_dim is None:
            raise ValueError("head_dim is required with output_projection.")
        projection_group = output_projection_group_columns(output_projection, query_heads=heads, head_dim=head_dim)
    while queue:
        _negative_size, _start, node_id = heapq.heappop(queue)
        node = index.nodes[node_id]
        considered += 1
        head_decisions = []
        for head in heads:
            try:
                before, _before_name = _global_bound(query_exact=query_exact[head], recent_keys=recent_keys_by_head[head], recent_values=recent_values_by_head[head], index=index, skipped_node_ids=committed, all_leaf_keys=leaf_keys, all_leaf_values=leaf_values, precision=precision, attention_scale=attention_scale)
                after, name = _global_bound(query_exact=query_exact[head], recent_keys=recent_keys_by_head[head], recent_values=recent_values_by_head[head], index=index, skipped_node_ids=tuple(committed + [node_id]), all_leaf_keys=leaf_keys, all_leaf_values=leaf_values, precision=precision, attention_scale=attention_scale)
                passed = after <= tolerance
            except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
                fallback = True
                before, after, name, passed = exact_mpfr(math.inf, precision=precision), exact_mpfr(math.inf, precision=precision), "fallback", False
            head_decisions.append(GQAHeadDecision(head, before, after, name, passed, rigorous=not fallback))
        votes = sum(decision.passed for decision in head_decisions)
        all_passed = votes == len(heads) and all(decision.rigorous for decision in head_decisions)
        concat = norm_upper_value = group = None
        if not fallback:
            bounds = [decision.candidate_bound_after for decision in head_decisions]
            concat = concatenation_bound(bounds, precision=precision)
            if projection_group is not None:
                norm_upper_value = output_projection_norm_upper_bound(projection_group, precision=precision)
                group = rounded_mul(norm_upper_value, concat, precision=precision, round_mode=gmpy2.RoundUp)
        decisions.append(GQARegionDecision(kv_head, node_id, node.level, node.token_start, node.token_end, heads, tuple(head_decisions), all_passed, votes, concat, norm_upper_value, group))
        for decision in head_decisions:
            if decision.passed:
                logical_skips[decision.query_head].append(node_id)
        if all_passed:
            committed.append(node_id)
            certified += 1
        elif node.child_ids:
            descended += 1
            for child_id in node.child_ids:
                child = index.nodes[child_id]
                heapq.heappush(queue, (-child.token_count, child.token_start, child_id))
    eligible_leaves = tuple(sorted({leaf for node_id in committed for leaf in index.descendants(node_id)}))
    required = tuple(node_id for node_id in index.leaf_node_ids if node_id not in eligible_leaves)
    potential = sum((leaf_payload_bytes or {}).get(index.nodes[node_id].token_start, 0) for node_id in eligible_leaves)
    return GQATraversalResult(kv_head, heads, tuple(committed), eligible_leaves, required, tuple(decisions), {head: tuple(sorted(set(values))) for head, values in logical_skips.items()}, considered, certified, descended, potential, fallback)
