"""Hierarchical, metadata-only anisotropic certification for RACK-KV V2.

Leaf payloads remain the independently decodable V1 compressed blocks.  The
index below stores only regional certification metadata and never serializes a
second copy of keys or values.  Every authorization recomputes the existing
theorem on the *cumulative* skipped set, so separate nodes do not receive
independent error budgets.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Sequence

import gmpy2
import numpy as np

from .anisotropic import (
    AnisotropicBlockSummary,
    AnisotropicExecutionResult,
    anisotropic_logit_upper_bound_from_exact,
    build_anisotropic_summary_from_keys,
    execute_anisotropic_flat,
    reconstructed_attention_output,
)
from .certificate import _certificates_from_bounds, _point_logit_interval_from_exact
from .codec import CompressedBlock
from .rigorous import (
    DEFAULT_PRECISION, exact_mpfr, exact_vector, max_exact, norm_upper,
    rounded_add, rounded_div, rounded_exp, rounded_mul, rounded_sub,
)


@dataclass(frozen=True)
class HierarchyNode:
    node_id: int
    level: int
    token_start: int
    token_end: int
    child_ids: tuple[int, ...]
    leaf_index: int | None
    summary: AnisotropicBlockSummary
    value_norm_upper: gmpy2.mpfr

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True)
class HierarchyIndex:
    nodes: dict[int, HierarchyNode]
    root_id: int
    leaf_node_ids: tuple[int, ...]
    fanout: int
    rank: int

    def descendants(self, node_id: int) -> tuple[int, ...]:
        node = self.nodes[node_id]
        if node.leaf_index is not None:
            return (node.node_id,)
        leaves: list[int] = []
        for child in node.child_ids:
            leaves.extend(self.descendants(child))
        return tuple(leaves)


@dataclass(frozen=True)
class HierarchicalTraversal:
    skipped_node_ids: tuple[int, ...]
    kept_leaf_node_ids: tuple[int, ...]
    skipped_leaf_node_ids: tuple[int, ...]
    certificate_bound: gmpy2.mpfr
    certificate_name: str
    nodes_considered: int
    nodes_certified: int
    nodes_descended: int
    leaf_certificates_evaluated: int
    largest_certified_region: int
    highest_certified_tree_level: int
    descendant_leaf_evaluations_avoided: int
    ordering: tuple[int, ...]
    numerical_fallback_used: bool


@dataclass(frozen=True)
class HierarchicalExecutionResult:
    mode: str
    output: np.ndarray
    compression_only_output: np.ndarray
    kept_block_starts: tuple[int, ...]
    skipped_block_starts: tuple[int, ...]
    traversal: HierarchicalTraversal
    attention_token_count: int


def _value_norm_bound(values: np.ndarray, *, precision: int) -> gmpy2.mpfr:
    return max_exact([norm_upper(exact_vector(row, precision=precision), precision=precision) for row in values], precision=precision)


def build_hierarchy(
    historical_blocks: Sequence[CompressedBlock],
    *,
    fanout: int = 4,
    rank: int = 8,
    precision: int = DEFAULT_PRECISION,
) -> HierarchyIndex:
    """Build a consecutive-range metadata index; no node retains KV payload."""
    if fanout < 2:
        raise ValueError("fanout must be at least two.")
    if not historical_blocks:
        raise ValueError("a hierarchy needs at least one leaf block.")
    ordered = tuple(sorted(historical_blocks, key=lambda block: block.header.block_start))
    nodes: dict[int, HierarchyNode] = {}
    key_regions: dict[int, np.ndarray] = {}
    value_regions: dict[int, np.ndarray] = {}
    current: list[int] = []
    next_id = 0
    previous_end: int | None = None
    for leaf_index, block in enumerate(ordered):
        keys, values = block.decode_block()
        start, end = block.header.block_start, block.header.block_start + block.block_len
        if previous_end is not None and start != previous_end:
            raise ValueError("historical leaves must be consecutive for hierarchical certification.")
        previous_end = end
        summary = build_anisotropic_summary_from_keys(
            keys, block_start=start, rank=rank, precision=precision,
        )
        node_id = next_id
        next_id += 1
        nodes[node_id] = HierarchyNode(node_id, 0, start, end, (), leaf_index, summary, _value_norm_bound(values, precision=precision))
        key_regions[node_id], value_regions[node_id] = keys, values
        current.append(node_id)
    leaf_node_ids = tuple(current)
    level = 0
    while len(current) > 1:
        level += 1
        parents: list[int] = []
        for begin in range(0, len(current), fanout):
            children = tuple(current[begin:begin + fanout])
            child_nodes = [nodes[child] for child in children]
            start, end = child_nodes[0].token_start, child_nodes[-1].token_end
            if any(left.token_end != right.token_start for left, right in zip(child_nodes, child_nodes[1:])):
                raise ValueError("parent children must cover one consecutive range.")
            keys = np.vstack([key_regions[child] for child in children])
            values = np.vstack([value_regions[child] for child in children])
            node_id = next_id
            next_id += 1
            summary = build_anisotropic_summary_from_keys(keys, block_start=start, rank=rank, precision=precision)
            nodes[node_id] = HierarchyNode(node_id, level, start, end, children, None, summary, _value_norm_bound(values, precision=precision))
            key_regions[node_id], value_regions[node_id] = keys, values
            parents.append(node_id)
        current = parents
    return HierarchyIndex(nodes=nodes, root_id=current[0], leaf_node_ids=leaf_node_ids, fanout=fanout, rank=rank)


def validate_hierarchy(index: HierarchyIndex) -> None:
    """Validate exact coverage and parent/child range invariants."""
    leaves = [index.nodes[node_id] for node_id in index.leaf_node_ids]
    if len({node.token_start for node in leaves}) != len(leaves):
        raise ValueError("leaf ranges overlap.")
    for left, right in zip(leaves, leaves[1:]):
        if left.token_end != right.token_start:
            raise ValueError("leaves do not cover a consecutive history exactly once.")
    for node in index.nodes.values():
        if node.child_ids:
            children = [index.nodes[child] for child in node.child_ids]
            if node.token_start != children[0].token_start or node.token_end != children[-1].token_end:
                raise ValueError("parent range is not its children's union.")
            if any(left.token_end != right.token_start for left, right in zip(children, children[1:])):
                raise ValueError("parent children overlap or leave a gap.")


def _global_bound(
    *,
    query_exact: Sequence[gmpy2.mpfr],
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
    index: HierarchyIndex,
    skipped_node_ids: Sequence[int],
    all_leaf_keys: Sequence[np.ndarray],
    all_leaf_values: Sequence[np.ndarray],
    precision: int,
    attention_scale: float | None,
) -> tuple[gmpy2.mpfr, str]:
    """Evaluate the unchanged theorem for one disjoint cumulative skipped set."""
    skipped_leaves = {leaf for node_id in skipped_node_ids for leaf in index.descendants(node_id)}
    kept_keys = [np.asarray(recent_keys, dtype=np.float64)]
    kept_values = [np.asarray(recent_values, dtype=np.float64)]
    for leaf_node_id in index.leaf_node_ids:
        leaf = index.nodes[leaf_node_id]
        if leaf_node_id not in skipped_leaves:
            assert leaf.leaf_index is not None
            kept_keys.append(all_leaf_keys[leaf.leaf_index])
            kept_values.append(all_leaf_values[leaf.leaf_index])
    key_array, value_array = np.vstack(kept_keys), np.vstack(kept_values)
    token_entries = []
    for key, value in zip(key_array, value_array):
        token_entries.append((_point_logit_interval_from_exact(query_exact, exact_vector(key, precision=precision), precision=precision, attention_scale=attention_scale), norm_upper(exact_vector(value, precision=precision), precision=precision)))
    caps = [anisotropic_logit_upper_bound_from_exact(query_exact, index.nodes[node_id].summary, precision=precision, attention_scale=attention_scale).upper for node_id in skipped_node_ids]
    shift = max_exact([entry[0].upper for entry in token_entries] + caps, precision=precision)
    z_lower = exact_mpfr(0, precision=precision)
    numerator_upper = exact_mpfr(0, precision=precision)
    for logit, value_norm in token_entries:
        lower = rounded_exp(rounded_sub(logit.lower, shift, precision=precision, round_mode=gmpy2.RoundDown), precision=precision, round_mode=gmpy2.RoundDown)
        upper = rounded_exp(rounded_sub(logit.upper, shift, precision=precision, round_mode=gmpy2.RoundUp), precision=precision, round_mode=gmpy2.RoundUp)
        z_lower = rounded_add(z_lower, lower, precision=precision, round_mode=gmpy2.RoundDown)
        numerator_upper = rounded_add(numerator_upper, rounded_mul(upper, value_norm, precision=precision, round_mode=gmpy2.RoundUp), precision=precision, round_mode=gmpy2.RoundUp)
    output_norm_upper = rounded_div(numerator_upper, z_lower, precision=precision, round_mode=gmpy2.RoundUp)
    masses = []
    nus = []
    for node_id, cap in zip(skipped_node_ids, caps):
        node = index.nodes[node_id]
        masses.append(rounded_mul(exact_mpfr(node.token_count, precision=precision), rounded_exp(rounded_sub(cap, shift, precision=precision, round_mode=gmpy2.RoundUp), precision=precision, round_mode=gmpy2.RoundUp), precision=precision, round_mode=gmpy2.RoundUp))
        nus.append(node.value_norm_upper)
    candidates, _summary = _certificates_from_bounds(z_k_lower=z_lower, output_norm_upper=output_norm_upper, upper_masses=masses, upper_value_norms=nus, precision=precision)
    chosen = min(candidates, key=lambda candidate: candidate.value)
    return chosen.value, chosen.name


def select_kept_leaves(
    *,
    query: np.ndarray,
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
    index: HierarchyIndex,
    leaf_keys: Sequence[np.ndarray],
    leaf_values: Sequence[np.ndarray],
    tolerance: float,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
) -> HierarchicalTraversal:
    """Deterministic coarse-first traversal with one global MPFR budget."""
    validate_hierarchy(index)
    query_exact = exact_vector(np.asarray(query, dtype=np.float64), precision=precision)
    tolerance_exact = exact_mpfr(tolerance, precision=precision)
    queue: list[tuple[int, int, int]] = [(-index.nodes[index.root_id].token_count, index.nodes[index.root_id].token_start, index.root_id)]
    skipped_nodes: list[int] = []
    ordering: list[int] = []
    considered = certified = descended = leaf_evaluations = avoided = 0
    largest = highest = 0
    last_bound = exact_mpfr(0, precision=precision)
    last_name = "zero"
    fallback = False
    while queue:
        _negative_size, _start, node_id = heapq.heappop(queue)
        node = index.nodes[node_id]
        ordering.append(node_id)
        considered += 1
        if node.leaf_index is not None:
            leaf_evaluations += 1
        try:
            bound, name = _global_bound(query_exact=query_exact, recent_keys=recent_keys, recent_values=recent_values, index=index, skipped_node_ids=tuple(skipped_nodes + [node_id]), all_leaf_keys=leaf_keys, all_leaf_values=leaf_values, precision=precision, attention_scale=attention_scale)
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
            fallback = True
            bound, name = exact_mpfr(math.inf, precision=precision), "fallback"
        if bound <= tolerance_exact:
            skipped_nodes.append(node_id)
            certified += 1
            last_bound, last_name = bound, name
            largest = max(largest, node.token_count)
            highest = max(highest, node.level)
            if node.level > 0:
                avoided += len(index.descendants(node_id))
            continue
        if node.child_ids:
            descended += 1
            for child_id in node.child_ids:
                child = index.nodes[child_id]
                heapq.heappush(queue, (-child.token_count, child.token_start, child_id))
    skipped_leaves = tuple(sorted({leaf for node_id in skipped_nodes for leaf in index.descendants(node_id)}))
    kept_leaves = tuple(node_id for node_id in index.leaf_node_ids if node_id not in skipped_leaves)
    return HierarchicalTraversal(tuple(skipped_nodes), kept_leaves, skipped_leaves, last_bound, last_name, considered, certified, descended, leaf_evaluations, largest, highest, avoided, tuple(ordering), fallback)


def execute_anisotropic_hierarchical(
    *,
    query: np.ndarray,
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
    historical_blocks: Sequence[CompressedBlock],
    rank: int = 8,
    fanout: int = 4,
    tolerance: float = 0.05,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
    hierarchy_enabled: bool = True,
) -> HierarchicalExecutionResult | AnisotropicExecutionResult:
    """Run actual hierarchical logical skipping; disabled mode delegates flat V2."""
    if not hierarchy_enabled:
        return execute_anisotropic_flat(query=query, recent_keys=recent_keys, recent_values=recent_values, historical_blocks=historical_blocks, rank=rank, tolerance=tolerance, precision=precision, attention_scale=attention_scale)
    ordered = tuple(sorted(historical_blocks, key=lambda block: block.header.block_start))
    index = build_hierarchy(ordered, fanout=fanout, rank=rank, precision=precision)
    leaf_payloads = [block.decode_block() for block in ordered]
    leaf_keys, leaf_values = [payload[0] for payload in leaf_payloads], [payload[1] for payload in leaf_payloads]
    traversal = select_kept_leaves(query=query, recent_keys=recent_keys, recent_values=recent_values, index=index, leaf_keys=leaf_keys, leaf_values=leaf_values, tolerance=tolerance, precision=precision, attention_scale=attention_scale)
    all_output = reconstructed_attention_output(query, np.vstack([recent_keys, *leaf_keys]), np.vstack([recent_values, *leaf_values]), attention_scale=attention_scale)
    kept_keys = [np.asarray(recent_keys, dtype=np.float64)] + [leaf_keys[index.nodes[node_id].leaf_index] for node_id in traversal.kept_leaf_node_ids]
    kept_values = [np.asarray(recent_values, dtype=np.float64)] + [leaf_values[index.nodes[node_id].leaf_index] for node_id in traversal.kept_leaf_node_ids]
    output = reconstructed_attention_output(query, np.vstack(kept_keys), np.vstack(kept_values), attention_scale=attention_scale)
    return HierarchicalExecutionResult("rack_v2_aniso_hier_r%d" % rank, output, all_output, tuple(index.nodes[node_id].token_start for node_id in traversal.kept_leaf_node_ids), tuple(index.nodes[node_id].token_start for node_id in traversal.skipped_leaf_node_ids), traversal, int(sum(part.shape[0] for part in kept_keys)))
