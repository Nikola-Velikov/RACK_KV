"""Shadow-only low-rank anisotropic certificate for RACK-KV 2.0.

The module does not participate in the V1 attention path.  A float64 SVD
selects finite basis vectors; MPFR interval arithmetic then treats those
vectors and the finite coefficients as exact metadata and proves an explicit
residual radius.  Only the block logit cap differs from the frozen V1
certificate.  The output-error theorem and progressive decode schedule are
reused unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable, Mapping, Sequence

import gmpy2
import numpy as np

from .certificate import (
    _certificates_from_bounds,
    _point_logit_interval_from_exact,
    _prepare_block_exact,
)
from .codec import CompressedBlock
from .rigorous import (
    DEFAULT_PRECISION,
    Interval,
    add_intervals,
    divide_interval_by_positive_interval,
    dot_interval,
    exact_mpfr,
    exact_vector,
    max_exact,
    min_exact,
    mpfr_context,
    multiply_interval_by_nonnegative_scalar,
    norm_lower,
    norm_upper,
    rounded_add,
    rounded_div,
    rounded_exp,
    rounded_mul,
    rounded_sqrt,
    rounded_sub,
    sqrt_dimension_interval,
)
from .types import DecodeSchedule


CenterMode = str


@dataclass(frozen=True)
class AnisotropicBlockSummary:
    """Finite directional metadata plus an MPFR-proven residual radius."""

    block_start: int
    block_len: int
    head_dim: int
    rank: int
    center_mode: CenterMode
    center: np.ndarray
    basis: np.ndarray
    coefficient_lower: np.ndarray
    coefficient_upper: np.ndarray
    rho_res_upper: gmpy2.mpfr
    rho_res_upper_text: str
    rho_sphere: float
    singular_values: np.ndarray
    effective_rank: float
    build_seconds: float
    metadata_bytes: int

    def __post_init__(self) -> None:
        if self.rank < 0 or self.rank > self.head_dim:
            raise ValueError("rank must be within [0, head_dim].")
        if self.center_mode not in {"first_token", "centroid"}:
            raise ValueError("center_mode must be first_token or centroid.")
        if self.center.shape != (self.head_dim,):
            raise ValueError("center has the wrong shape.")
        if self.basis.shape != (self.head_dim, self.rank):
            raise ValueError("basis has the wrong shape.")
        if self.coefficient_lower.shape != (self.rank,) or self.coefficient_upper.shape != (self.rank,):
            raise ValueError("coefficient interval arrays have the wrong shape.")
        if np.any(self.coefficient_lower > self.coefficient_upper):
            raise ValueError("coefficient lower bounds exceed upper bounds.")
        for array in (self.center, self.basis, self.coefficient_lower, self.coefficient_upper):
            if not np.all(np.isfinite(array)):
                raise ValueError("anisotropic metadata must be finite.")
        if self.rho_res_upper < 0:
            raise ValueError("residual radius must be nonnegative.")


@dataclass(frozen=True)
class ApproximateAnisotropicBound:
    logit_upper: float
    log_mass_upper: float
    rigorous_authorization: bool = False
    would_skip: None = None


@dataclass(frozen=True)
class ShadowBlockDecision:
    block_start: int
    block_len: int
    certificate_bound: gmpy2.mpfr
    certificate_name: str
    would_skip: bool
    iteration: int


@dataclass(frozen=True)
class ShadowCertificationResult:
    rank: int
    center_mode: CenterMode
    would_certify: bool
    certificate_bound: gmpy2.mpfr
    certificate_name: str
    decoded_block_starts: tuple[int, ...]
    skipped_block_starts: tuple[int, ...]
    decisions: tuple[ShadowBlockDecision, ...]
    iterations: int
    numerical_fallback_used: bool
    evaluation_seconds: float
    rigorous: bool = True


@dataclass(frozen=True)
class AnisotropicExecutionResult:
    """Actual V2 logical-attention result authorized exclusively by MPFR."""

    mode: str
    output: np.ndarray
    compression_only_output: np.ndarray
    kept_block_starts: tuple[int, ...]
    skipped_block_starts: tuple[int, ...]
    certificate: ShadowCertificationResult
    attention_token_count: int


def reconstructed_attention_output(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    attention_scale: float | None = None,
) -> np.ndarray:
    """Stable float64 attention over a specified reconstructed token set."""
    query = np.asarray(query, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0] or keys.shape[0] == 0:
        raise ValueError("attention requires matching nonempty rank-2 key/value arrays.")
    scale = attention_scale if attention_scale is not None else 1.0 / math.sqrt(keys.shape[1])
    logits = (keys @ query) * scale
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= np.sum(weights)
    return weights @ values


def execute_anisotropic_flat(
    *,
    query: np.ndarray,
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
    historical_blocks: Sequence[CompressedBlock],
    rank: int,
    tolerance: float,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
    schedule: DecodeSchedule = DecodeSchedule.LARGEST_U_TIMES_NU,
) -> AnisotropicExecutionResult:
    """Run real logical skipping using the V2 anisotropic MPFR certificate.

    The function intentionally leaves compression and serialized payloads
    untouched.  It decodes the same reconstructed blocks as V1, but removes
    only rigorously authorized blocks from the attention arrays.
    """
    summaries = tuple(build_anisotropic_summary(block, rank=rank, precision=precision) for block in historical_blocks)
    certificate = progressive_anisotropic_shadow(
        query=query,
        recent_keys=recent_keys,
        recent_values=recent_values,
        historical_blocks=historical_blocks,
        summaries=summaries,
        tolerance=tolerance,
        schedule=schedule,
        precision=precision,
        attention_scale=attention_scale,
    )
    if not certificate.rigorous:
        raise RuntimeError("float64 diagnostics cannot authorize V2 logical skipping.")
    decoded = {block.header.block_start: block.decode_block() for block in historical_blocks}
    all_keys = [np.asarray(recent_keys, dtype=np.float64)] + [decoded[block.header.block_start][0] for block in historical_blocks]
    all_values = [np.asarray(recent_values, dtype=np.float64)] + [decoded[block.header.block_start][1] for block in historical_blocks]
    compression_only = reconstructed_attention_output(query, np.vstack(all_keys), np.vstack(all_values), attention_scale=attention_scale)
    skipped = set(certificate.skipped_block_starts)
    kept_blocks = tuple(block.header.block_start for block in historical_blocks if block.header.block_start not in skipped)
    kept_keys = [np.asarray(recent_keys, dtype=np.float64)] + [decoded[start][0] for start in kept_blocks]
    kept_values = [np.asarray(recent_values, dtype=np.float64)] + [decoded[start][1] for start in kept_blocks]
    output = reconstructed_attention_output(query, np.vstack(kept_keys), np.vstack(kept_values), attention_scale=attention_scale)
    return AnisotropicExecutionResult(
        mode=f"rack_v2_aniso_flat_r{rank}",
        output=output,
        compression_only_output=compression_only,
        kept_block_starts=kept_blocks,
        skipped_block_starts=tuple(sorted(skipped)),
        certificate=certificate,
        attention_token_count=int(sum(part.shape[0] for part in kept_keys)),
    )


@dataclass(frozen=True)
class PreparedShadowToken:
    logit_interval: Interval
    value_norm_upper: gmpy2.mpfr


@dataclass(frozen=True)
class PreparedShadowBlock:
    block: CompressedBlock
    block_len_exact: gmpy2.mpfr
    nu_exact: gmpy2.mpfr
    token_entries: tuple[PreparedShadowToken, ...]


@dataclass(frozen=True)
class PreparedAnisotropicCase:
    query_exact: tuple[gmpy2.mpfr, ...]
    query_norm_bounds: Interval
    kept_entries: tuple[PreparedShadowToken, ...]
    historical_blocks: tuple[PreparedShadowBlock, ...]
    precision: int
    attention_scale: float | None


def _interval_product(left: Interval, right: Interval, *, precision: int) -> Interval:
    lower = [
        rounded_mul(a, b, precision=precision, round_mode=gmpy2.RoundDown)
        for a in (left.lower, left.upper)
        for b in (right.lower, right.upper)
    ]
    upper = [
        rounded_mul(a, b, precision=precision, round_mode=gmpy2.RoundUp)
        for a in (left.lower, left.upper)
        for b in (right.lower, right.upper)
    ]
    return Interval(min_exact(lower, precision=precision), max_exact(upper, precision=precision))


def _subtract_intervals(left: Interval, right: Interval, *, precision: int) -> Interval:
    return Interval(
        rounded_sub(left.lower, right.upper, precision=precision, round_mode=gmpy2.RoundDown),
        rounded_sub(left.upper, right.lower, precision=precision, round_mode=gmpy2.RoundUp),
    )


def _point_interval(value: float, *, precision: int) -> Interval:
    exact = exact_mpfr(np.float64(value), precision=precision)
    return Interval(exact, exact)


def _fast_point_dot_interval(
    left: Sequence[gmpy2.mpfr],
    right: Sequence[gmpy2.mpfr],
    *,
    precision: int,
) -> Interval:
    """Directed MPFR dot product with one context setup per endpoint."""
    if len(left) != len(right):
        raise ValueError("dot product dimension mismatch.")
    with mpfr_context(precision=precision, round_mode=gmpy2.RoundDown):
        lower = gmpy2.mpfr(0)
        for lhs, rhs in zip(left, right):
            lower = gmpy2.fma(lhs, rhs, lower)
    with mpfr_context(precision=precision, round_mode=gmpy2.RoundUp):
        upper = gmpy2.mpfr(0)
        for lhs, rhs in zip(left, right):
            upper = gmpy2.fma(lhs, rhs, upper)
    return Interval(lower, upper)


def _interval_vector_norm_upper(intervals: Sequence[Interval], *, precision: int) -> gmpy2.mpfr:
    with mpfr_context(precision=precision, round_mode=gmpy2.RoundUp):
        squared_sum = gmpy2.mpfr(0)
        for interval in intervals:
            magnitude = max(-interval.lower, interval.upper)
            squared_sum = gmpy2.fma(magnitude, magnitude, squared_sum)
        return gmpy2.sqrt(squared_sum)


def _interval_vector_subtract(left: np.ndarray, right: np.ndarray, *, precision: int) -> tuple[Interval, ...]:
    return tuple(
        Interval(
            rounded_sub(exact_mpfr(np.float64(a), precision=precision), exact_mpfr(np.float64(b), precision=precision),
                        precision=precision, round_mode=gmpy2.RoundDown),
            rounded_sub(exact_mpfr(np.float64(a), precision=precision), exact_mpfr(np.float64(b), precision=precision),
                        precision=precision, round_mode=gmpy2.RoundUp),
        )
        for a, b in zip(left, right)
    )


def _rigorous_residual_radius(
    keys: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    coefficients: np.ndarray,
    *,
    precision: int,
) -> gmpy2.mpfr:
    radii: list[gmpy2.mpfr] = []
    for key, coefficient_row in zip(keys, coefficients):
        x_intervals = _interval_vector_subtract(key, center, precision=precision)
        residual_intervals: list[Interval] = []
        coefficient_exact = exact_vector(coefficient_row, precision=precision)
        for component, x_interval in enumerate(x_intervals):
            projection = _fast_point_dot_interval(
                exact_vector(basis[component, :], precision=precision),
                coefficient_exact,
                precision=precision,
            )
            residual_intervals.append(_subtract_intervals(x_interval, projection, precision=precision))
        radii.append(_interval_vector_norm_upper(residual_intervals, precision=precision))
    return max_exact(radii, precision=precision)


def _effective_rank(singular_values: np.ndarray) -> float:
    energy = np.square(singular_values, dtype=np.float64)
    total = float(energy.sum())
    if total == 0.0:
        return 0.0
    probabilities = energy / total
    positive = probabilities[probabilities > 0]
    return float(np.exp(-np.sum(positive * np.log(positive))))


def build_anisotropic_summary(
    block: CompressedBlock,
    *,
    rank: int,
    center_mode: CenterMode = "first_token",
    precision: int = DEFAULT_PRECISION,
    fixed_basis: np.ndarray | None = None,
) -> AnisotropicBlockSummary:
    """Build shadow metadata without changing the compressed block."""
    keys = np.asarray(block.decode_key_block(), dtype=np.float64)
    return build_anisotropic_summary_from_keys(
        keys,
        block_start=block.header.block_start,
        rank=rank,
        center_mode=center_mode,
        precision=precision,
        fixed_basis=fixed_basis,
        v1_rho_upper=float(block.header.rho_upper),
    )


def build_anisotropic_summary_from_keys(
    keys: np.ndarray,
    *,
    block_start: int,
    rank: int,
    center_mode: CenterMode = "first_token",
    precision: int = DEFAULT_PRECISION,
    fixed_basis: np.ndarray | None = None,
    v1_rho_upper: float | None = None,
) -> AnisotropicBlockSummary:
    """Build rigorous directional metadata for an already reconstructed region.

    ``keys`` are never quantized or serialized here.  This permits hierarchy
    nodes to use the exact reconstructed leaf representation while storing
    metadata only.  Supplying ``v1_rho_upper`` preserves the V1 rank-zero
    operation sequence for a compressed leaf.
    """
    started = time.perf_counter()
    keys = np.asarray(keys, dtype=np.float64)
    if keys.ndim != 2 or keys.shape[0] == 0:
        raise ValueError("keys must be a nonempty rank-2 array.")
    if block_start < 0:
        raise ValueError("block_start must be nonnegative.")
    if rank < 0 or rank > keys.shape[1]:
        raise ValueError("rank must be within [0, key dimension].")
    if center_mode == "first_token":
        center = np.asarray(keys[0], dtype=np.float64).copy()
    elif center_mode == "centroid":
        center = np.asarray(keys.mean(axis=0), dtype=np.float64)
    else:
        raise ValueError("center_mode must be first_token or centroid.")
    residual_matrix = np.asarray(keys - center, dtype=np.float64)
    if fixed_basis is None:
        _left, singular_values, right = np.linalg.svd(residual_matrix, full_matrices=True)
        basis = np.asarray(right[:rank].T, dtype=np.float64).copy()
    else:
        basis = np.asarray(fixed_basis, dtype=np.float64).copy()
        if basis.shape != (keys.shape[1], rank):
            raise ValueError("fixed_basis has the wrong shape.")
        singular_values = np.linalg.svd(residual_matrix, compute_uv=False)
    coefficients = np.asarray(residual_matrix @ basis, dtype=np.float64)
    coefficient_lower = coefficients.min(axis=0) if rank else np.zeros((0,), dtype=np.float64)
    coefficient_upper = coefficients.max(axis=0) if rank else np.zeros((0,), dtype=np.float64)

    # Rank zero with the V1 center deliberately imports the serialized V1
    # radius, making the rigorous query cap identical to the frozen sphere.
    if rank == 0 and center_mode == "first_token" and v1_rho_upper is not None:
        rho_res = exact_mpfr(v1_rho_upper, precision=precision)
    else:
        rho_res = _rigorous_residual_radius(
            keys,
            center,
            basis,
            coefficients,
            precision=precision,
        )
    rho_sphere = float(v1_rho_upper) if center_mode == "first_token" and v1_rho_upper is not None else float(
        np.linalg.norm(keys - center, axis=1).max()
    )
    # Float64 values are counted because their exact bit patterns define the
    # shadow proof metadata.  The existing center is free in first-token mode.
    center_bytes = 0 if center_mode == "first_token" else keys.shape[1] * 8
    metadata_bytes = center_bytes + rank * keys.shape[1] * 8 + 2 * rank * 8 + (4 if rank == 0 and center_mode == "first_token" else 8)
    return AnisotropicBlockSummary(
        block_start=block_start,
        block_len=int(keys.shape[0]),
        head_dim=keys.shape[1],
        rank=rank,
        center_mode=center_mode,
        center=center,
        basis=basis,
        coefficient_lower=np.asarray(coefficient_lower, dtype=np.float64),
        coefficient_upper=np.asarray(coefficient_upper, dtype=np.float64),
        rho_res_upper=rho_res,
        rho_res_upper_text=str(rho_res),
        rho_sphere=rho_sphere,
        singular_values=np.asarray(singular_values, dtype=np.float64),
        effective_rank=_effective_rank(np.asarray(singular_values, dtype=np.float64)),
        build_seconds=time.perf_counter() - started,
        metadata_bytes=metadata_bytes,
    )


def approximate_anisotropic_bound(
    query: np.ndarray,
    summary: AnisotropicBlockSummary,
    *,
    attention_scale: float | None = None,
) -> ApproximateAnisotropicBound:
    query = np.asarray(query, dtype=np.float64)
    projected = query @ summary.basis
    directional = np.where(projected >= 0.0, projected * summary.coefficient_upper, projected * summary.coefficient_lower)
    numerator = float(query @ summary.center + directional.sum() + np.linalg.norm(query) * float(summary.rho_res_upper))
    scale = attention_scale if attention_scale is not None else 1.0 / math.sqrt(summary.head_dim)
    upper = numerator * scale
    return ApproximateAnisotropicBound(logit_upper=upper, log_mass_upper=math.log(summary.block_len) + upper)


def anisotropic_logit_upper_bound_from_exact(
    query_exact: Sequence[gmpy2.mpfr],
    summary: AnisotropicBlockSummary,
    *,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
    query_norm_bounds: Interval | None = None,
) -> Interval:
    if len(query_exact) != summary.head_dim:
        raise ValueError("query and summary dimensions differ.")
    query_norm_bounds = query_norm_bounds or Interval(
        norm_lower(query_exact, precision=precision),
        norm_upper(query_exact, precision=precision),
    )
    if summary.rank == 0 and summary.center_mode == "first_token":
        numerator = dot_interval(query_exact, exact_vector(summary.center, precision=precision), precision=precision)
    else:
        numerator = _fast_point_dot_interval(
            query_exact,
            exact_vector(summary.center, precision=precision),
            precision=precision,
        )
    for column in range(summary.rank):
        projection = _fast_point_dot_interval(
            query_exact,
            exact_vector(summary.basis[:, column], precision=precision),
            precision=precision,
        )
        coefficient_interval = Interval(
            exact_mpfr(summary.coefficient_lower[column], precision=precision),
            exact_mpfr(summary.coefficient_upper[column], precision=precision),
        )
        numerator = add_intervals(
            numerator,
            _interval_product(projection, coefficient_interval, precision=precision),
            precision=precision,
        )
    radial_upper = rounded_mul(
        query_norm_bounds.upper,
        summary.rho_res_upper,
        precision=precision,
        round_mode=gmpy2.RoundUp,
    )
    # For rank zero, reproduce the exact V1 operation sequence (including the
    # unused positive lower radial endpoint) for an interval-level invariant.
    if summary.rank == 0 and summary.center_mode == "first_token":
        radial = multiply_interval_by_nonnegative_scalar(query_norm_bounds, summary.rho_res_upper, precision=precision)
    else:
        radial = Interval(-radial_upper, radial_upper)
    numerator = add_intervals(numerator, radial, precision=precision)
    if attention_scale is None:
        return divide_interval_by_positive_interval(
            numerator,
            sqrt_dimension_interval(summary.head_dim, precision=precision),
            precision=precision,
        )
    return multiply_interval_by_nonnegative_scalar(
        numerator,
        exact_mpfr(attention_scale, precision=precision),
        precision=precision,
    )


def anisotropic_logit_upper_bound(
    query: np.ndarray,
    summary: AnisotropicBlockSummary,
    *,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
) -> Interval:
    return anisotropic_logit_upper_bound_from_exact(
        exact_vector(np.asarray(query, dtype=np.float64), precision=precision),
        summary,
        precision=precision,
        attention_scale=attention_scale,
    )


def anisotropic_log_mass_upper(
    query: np.ndarray,
    summary: AnisotropicBlockSummary,
    *,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
) -> gmpy2.mpfr:
    cap = anisotropic_logit_upper_bound(
        query,
        summary,
        precision=precision,
        attention_scale=attention_scale,
    ).upper
    return rounded_add(
        gmpy2.log(exact_mpfr(summary.block_len, precision=precision)),
        cap,
        precision=precision,
        round_mode=gmpy2.RoundUp,
    )


def prepare_anisotropic_case(
    *,
    query: np.ndarray,
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
    historical_blocks: Sequence[CompressedBlock],
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
) -> PreparedAnisotropicCase:
    """Prepare all rank-independent MPFR quantities once for a query case."""
    query_array = np.asarray(query, dtype=np.float64)
    recent_keys_array = np.asarray(recent_keys, dtype=np.float64)
    recent_values_array = np.asarray(recent_values, dtype=np.float64)
    if recent_keys_array.ndim != 2 or recent_values_array.ndim != 2:
        raise ValueError("recent keys and values must be rank-2.")
    query_exact = tuple(exact_vector(query_array, precision=precision))
    query_norm_bounds = Interval(norm_lower(query_exact, precision=precision), norm_upper(query_exact, precision=precision))
    kept_entries: list[PreparedShadowToken] = []
    for key, value in zip(recent_keys_array, recent_values_array):
        key_exact = tuple(exact_vector(key, precision=precision))
        value_exact = tuple(exact_vector(value, precision=precision))
        kept_entries.append(PreparedShadowToken(
            logit_interval=_point_logit_interval_from_exact(
                query_exact, key_exact, precision=precision, attention_scale=attention_scale
            ),
            value_norm_upper=norm_upper(value_exact, precision=precision),
        ))
    prepared_blocks: list[PreparedShadowBlock] = []
    for block in historical_blocks:
        prepared = _prepare_block_exact(block, precision=precision)
        token_entries = tuple(
            PreparedShadowToken(
                logit_interval=_point_logit_interval_from_exact(
                    query_exact, key_exact, precision=precision, attention_scale=attention_scale
                ),
                value_norm_upper=norm_upper(value_exact, precision=precision),
            )
            for key_exact, value_exact in zip(prepared.decoded_keys_exact, prepared.decoded_values_exact)
        )
        prepared_blocks.append(PreparedShadowBlock(
            block=block,
            block_len_exact=exact_mpfr(block.block_len, precision=precision),
            nu_exact=exact_mpfr(block.header.nu_upper, precision=precision),
            token_entries=token_entries,
        ))
    return PreparedAnisotropicCase(
        query_exact=query_exact,
        query_norm_bounds=query_norm_bounds,
        kept_entries=tuple(kept_entries),
        historical_blocks=tuple(prepared_blocks),
        precision=precision,
        attention_scale=attention_scale,
    )


def _progressive_fixed_shift_shadow(
    *,
    kept_entries: list[dict[str, object]],
    block_cache: list[dict[str, object]],
    summaries: Sequence[AnisotropicBlockSummary],
    tolerance_exact: gmpy2.mpfr,
    schedule: DecodeSchedule,
    precision: int,
    started: float,
) -> ShadowCertificationResult:
    """Equivalent theorem evaluation with one globally valid softmax shift."""
    shift = max_exact(
        [entry["logit_interval"].upper for entry in kept_entries]
        + [entry["cap"] for entry in block_cache]
        + [token["logit_interval"].upper for block in block_cache for token in block["token_entries"]],
        precision=precision,
    )

    def token_sums(entries):
        z_lower = exact_mpfr(0, precision=precision)
        numerator_upper = exact_mpfr(0, precision=precision)
        for entry in entries:
            logit = entry["logit_interval"]
            shifted_lower = rounded_sub(logit.lower, shift, precision=precision, round_mode=gmpy2.RoundDown)
            shifted_upper = rounded_sub(logit.upper, shift, precision=precision, round_mode=gmpy2.RoundUp)
            weight_lower = rounded_exp(shifted_lower, precision=precision, round_mode=gmpy2.RoundDown)
            weight_upper = rounded_exp(shifted_upper, precision=precision, round_mode=gmpy2.RoundUp)
            z_lower = rounded_add(z_lower, weight_lower, precision=precision, round_mode=gmpy2.RoundDown)
            numerator_upper = rounded_add(
                numerator_upper,
                rounded_mul(weight_upper, entry["value_norm_upper"], precision=precision, round_mode=gmpy2.RoundUp),
                precision=precision,
                round_mode=gmpy2.RoundUp,
            )
        return z_lower, numerator_upper

    z_k_lower, numerator_norm_upper = token_sums(kept_entries)
    for entry in block_cache:
        entry["kept_z"], entry["kept_num"] = token_sums(entry["token_entries"])
        shifted = rounded_sub(entry["cap"], shift, precision=precision, round_mode=gmpy2.RoundUp)
        entry["upper_mass"] = rounded_mul(
            entry["block_len"],
            rounded_exp(shifted, precision=precision, round_mode=gmpy2.RoundUp),
            precision=precision,
            round_mode=gmpy2.RoundUp,
        )

    decoded: list[int] = []
    decisions: list[ShadowBlockDecision] = []
    iterations = 0
    while block_cache:
        iterations += 1
        if z_k_lower <= 0:
            raise ValueError("kept mass lower bound is nonpositive.")
        output_norm_upper = rounded_div(
            numerator_norm_upper, z_k_lower, precision=precision, round_mode=gmpy2.RoundUp
        )
        upper_masses = [entry["upper_mass"] for entry in block_cache]
        upper_value_norms = [entry["nu"] for entry in block_cache]
        candidates, _bound_summary = _certificates_from_bounds(
            z_k_lower=z_k_lower,
            output_norm_upper=output_norm_upper,
            upper_masses=upper_masses,
            upper_value_norms=upper_value_norms,
            precision=precision,
        )
        chosen = min(candidates, key=lambda candidate: candidate.value)
        if chosen.value <= tolerance_exact:
            skipped = tuple(entry["block"].header.block_start for entry in block_cache)
            decisions.extend(
                ShadowBlockDecision(
                    block_start=entry["block"].header.block_start,
                    block_len=entry["block"].block_len,
                    certificate_bound=chosen.value,
                    certificate_name=chosen.name,
                    would_skip=True,
                    iteration=iterations,
                )
                for entry in block_cache
            )
            return ShadowCertificationResult(
                rank=summaries[0].rank,
                center_mode=summaries[0].center_mode,
                would_certify=True,
                certificate_bound=chosen.value,
                certificate_name=chosen.name,
                decoded_block_starts=tuple(decoded),
                skipped_block_starts=skipped,
                decisions=tuple(decisions),
                iterations=iterations,
                numerical_fallback_used=False,
                evaluation_seconds=time.perf_counter() - started,
            )
        if schedule == DecodeSchedule.LARGEST_U:
            index = max(range(len(block_cache)), key=lambda i: upper_masses[i])
        elif schedule == DecodeSchedule.LARGEST_U_TIMES_NU:
            scores = [
                rounded_mul(mass, nu, precision=precision, round_mode=gmpy2.RoundUp)
                for mass, nu in zip(upper_masses, upper_value_norms)
            ]
            index = max(range(len(block_cache)), key=lambda i: scores[i])
        else:
            raise ValueError(f"unsupported decode schedule: {schedule}")
        selected = block_cache.pop(index)
        block = selected["block"]
        decisions.append(ShadowBlockDecision(
            block_start=block.header.block_start,
            block_len=block.block_len,
            certificate_bound=chosen.value,
            certificate_name=chosen.name,
            would_skip=False,
            iteration=iterations,
        ))
        decoded.append(block.header.block_start)
        z_k_lower = rounded_add(z_k_lower, selected["kept_z"], precision=precision, round_mode=gmpy2.RoundDown)
        numerator_norm_upper = rounded_add(
            numerator_norm_upper, selected["kept_num"], precision=precision, round_mode=gmpy2.RoundUp
        )
    return ShadowCertificationResult(
        rank=summaries[0].rank,
        center_mode=summaries[0].center_mode,
        would_certify=False,
        certificate_bound=exact_mpfr(0, precision=precision),
        certificate_name="zero",
        decoded_block_starts=tuple(decoded),
        skipped_block_starts=tuple(),
        decisions=tuple(decisions),
        iterations=iterations,
        numerical_fallback_used=False,
        evaluation_seconds=time.perf_counter() - started,
    )


def progressive_anisotropic_shadow(
    *,
    query: np.ndarray,
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
    historical_blocks: Sequence[CompressedBlock],
    summaries: Sequence[AnisotropicBlockSummary],
    tolerance: float,
    schedule: DecodeSchedule = DecodeSchedule.LARGEST_U_TIMES_NU,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
    prepared_case: PreparedAnisotropicCase | None = None,
) -> ShadowCertificationResult:
    """Run the V1 progressive theorem with anisotropic caps, without outputs."""
    started = time.perf_counter()
    if len(historical_blocks) != len(summaries):
        raise ValueError("historical_blocks and summaries must have matching lengths.")
    if any(block.header.block_start != summary.block_start for block, summary in zip(historical_blocks, summaries)):
        raise ValueError("summary order does not match historical blocks.")
    prepared_case = prepared_case or prepare_anisotropic_case(
        query=query,
        recent_keys=recent_keys,
        recent_values=recent_values,
        historical_blocks=historical_blocks,
        precision=precision,
        attention_scale=attention_scale,
    )
    if prepared_case.precision != precision or prepared_case.attention_scale != attention_scale:
        raise ValueError("prepared case numerical configuration mismatch.")
    if len(prepared_case.historical_blocks) != len(historical_blocks):
        raise ValueError("prepared case block count mismatch.")
    query_exact = prepared_case.query_exact
    query_norm_bounds = prepared_case.query_norm_bounds
    kept_entries: list[dict[str, object]] = [
        {"logit_interval": entry.logit_interval, "value_norm_upper": entry.value_norm_upper}
        for entry in prepared_case.kept_entries
    ]
    block_cache: list[dict[str, object]] = []
    for prepared, summary in zip(prepared_case.historical_blocks, summaries):
        block = prepared.block
        token_entries = [
            {"logit_interval": entry.logit_interval, "value_norm_upper": entry.value_norm_upper}
            for entry in prepared.token_entries
        ]
        block_cache.append({
            "block": block,
            "summary": summary,
            "cap": anisotropic_logit_upper_bound_from_exact(
                query_exact,
                summary,
                precision=precision,
                attention_scale=attention_scale,
                query_norm_bounds=query_norm_bounds,
            ).upper,
            "block_len": prepared.block_len_exact,
            "nu": prepared.nu_exact,
            "token_entries": token_entries,
        })
    if not kept_entries and block_cache:
        first = block_cache.pop(0)
        kept_entries.extend(first["token_entries"])
    if not kept_entries:
        raise ValueError("certification requires at least one kept token.")

    decoded: list[int] = []
    decisions: list[ShadowBlockDecision] = []
    tolerance_exact = exact_mpfr(tolerance, precision=precision)
    if summaries:
        try:
            return _progressive_fixed_shift_shadow(
                kept_entries=kept_entries,
                block_cache=block_cache,
                summaries=summaries,
                tolerance_exact=tolerance_exact,
                schedule=schedule,
                precision=precision,
                started=started,
            )
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
            decisions = tuple(
                ShadowBlockDecision(
                    block_start=entry["block"].header.block_start,
                    block_len=entry["block"].block_len,
                    certificate_bound=exact_mpfr(0, precision=precision),
                    certificate_name="fallback",
                    would_skip=False,
                    iteration=0,
                )
                for entry in block_cache
            )
            return ShadowCertificationResult(
                rank=summaries[0].rank,
                center_mode=summaries[0].center_mode,
                would_certify=False,
                certificate_bound=exact_mpfr(0, precision=precision),
                certificate_name="fallback",
                decoded_block_starts=tuple(entry["block"].header.block_start for entry in block_cache),
                skipped_block_starts=tuple(),
                decisions=decisions,
                iterations=0,
                numerical_fallback_used=True,
                evaluation_seconds=time.perf_counter() - started,
            )
    iterations = 0
    last_bound = exact_mpfr(0, precision=precision)
    last_name = "zero"
    try:
        while block_cache:
            iterations += 1
            shift = max_exact(
                [entry["logit_interval"].upper for entry in kept_entries]
                + [entry["cap"] for entry in block_cache],
                precision=precision,
            )
            z_k_lower = exact_mpfr(0, precision=precision)
            numerator_norm_upper = exact_mpfr(0, precision=precision)
            for entry in kept_entries:
                logit = entry["logit_interval"]
                lower = rounded_sub(logit.lower, shift, precision=precision, round_mode=gmpy2.RoundDown)
                upper = rounded_sub(logit.upper, shift, precision=precision, round_mode=gmpy2.RoundUp)
                weight_lower = rounded_exp(lower, precision=precision, round_mode=gmpy2.RoundDown)
                weight_upper = rounded_exp(upper, precision=precision, round_mode=gmpy2.RoundUp)
                z_k_lower = rounded_add(z_k_lower, weight_lower, precision=precision, round_mode=gmpy2.RoundDown)
                numerator_norm_upper = rounded_add(
                    numerator_norm_upper,
                    rounded_mul(weight_upper, entry["value_norm_upper"], precision=precision, round_mode=gmpy2.RoundUp),
                    precision=precision,
                    round_mode=gmpy2.RoundUp,
                )
            if z_k_lower <= 0:
                raise ValueError("kept mass lower bound is nonpositive.")
            output_norm_upper = rounded_div(
                numerator_norm_upper, z_k_lower, precision=precision, round_mode=gmpy2.RoundUp
            )
            upper_masses: list[gmpy2.mpfr] = []
            upper_value_norms: list[gmpy2.mpfr] = []
            for entry in block_cache:
                shifted = rounded_sub(entry["cap"], shift, precision=precision, round_mode=gmpy2.RoundUp)
                upper_masses.append(
                    rounded_mul(
                        entry["block_len"],
                        rounded_exp(shifted, precision=precision, round_mode=gmpy2.RoundUp),
                        precision=precision,
                        round_mode=gmpy2.RoundUp,
                    )
                )
                upper_value_norms.append(entry["nu"])
            candidates, _summary = _certificates_from_bounds(
                z_k_lower=z_k_lower,
                output_norm_upper=output_norm_upper,
                upper_masses=upper_masses,
                upper_value_norms=upper_value_norms,
                precision=precision,
            )
            chosen = min(candidates, key=lambda candidate: candidate.value)
            last_bound, last_name = chosen.value, chosen.name
            if chosen.value <= tolerance_exact:
                skipped = tuple(entry["block"].header.block_start for entry in block_cache)
                decisions.extend(
                    ShadowBlockDecision(
                        block_start=entry["block"].header.block_start,
                        block_len=entry["block"].block_len,
                        certificate_bound=chosen.value,
                        certificate_name=chosen.name,
                        would_skip=True,
                        iteration=iterations,
                    )
                    for entry in block_cache
                )
                return ShadowCertificationResult(
                    rank=summaries[0].rank if summaries else 0,
                    center_mode=summaries[0].center_mode if summaries else "first_token",
                    would_certify=True,
                    certificate_bound=chosen.value,
                    certificate_name=chosen.name,
                    decoded_block_starts=tuple(decoded),
                    skipped_block_starts=skipped,
                    decisions=tuple(decisions),
                    iterations=iterations,
                    numerical_fallback_used=False,
                    evaluation_seconds=time.perf_counter() - started,
                )
            if schedule == DecodeSchedule.LARGEST_U:
                index = max(range(len(block_cache)), key=lambda i: upper_masses[i])
            elif schedule == DecodeSchedule.LARGEST_U_TIMES_NU:
                scores = [
                    rounded_mul(mass, nu, precision=precision, round_mode=gmpy2.RoundUp)
                    for mass, nu in zip(upper_masses, upper_value_norms)
                ]
                index = max(range(len(block_cache)), key=lambda i: scores[i])
            else:
                raise ValueError(f"unsupported decode schedule: {schedule}")
            selected = block_cache.pop(index)
            block = selected["block"]
            decisions.append(ShadowBlockDecision(
                block_start=block.header.block_start,
                block_len=block.block_len,
                certificate_bound=chosen.value,
                certificate_name=chosen.name,
                would_skip=False,
                iteration=iterations,
            ))
            decoded.append(block.header.block_start)
            kept_entries.extend(selected["token_entries"])
    except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
        for entry in block_cache:
            block = entry["block"]
            decisions.append(ShadowBlockDecision(
                block_start=block.header.block_start,
                block_len=block.block_len,
                certificate_bound=last_bound,
                certificate_name=last_name,
                would_skip=False,
                iteration=iterations,
            ))
            decoded.append(block.header.block_start)
        return ShadowCertificationResult(
            rank=summaries[0].rank if summaries else 0,
            center_mode=summaries[0].center_mode if summaries else "first_token",
            would_certify=False,
            certificate_bound=exact_mpfr(0, precision=precision),
            certificate_name="fallback",
            decoded_block_starts=tuple(decoded),
            skipped_block_starts=tuple(),
            decisions=tuple(decisions),
            iterations=iterations,
            numerical_fallback_used=True,
            evaluation_seconds=time.perf_counter() - started,
        )
    return ShadowCertificationResult(
        rank=summaries[0].rank if summaries else 0,
        center_mode=summaries[0].center_mode if summaries else "first_token",
        would_certify=False,
        certificate_bound=exact_mpfr(0, precision=precision),
        certificate_name="zero",
        decoded_block_starts=tuple(decoded),
        skipped_block_starts=tuple(),
        decisions=tuple(decisions),
        iterations=iterations,
        numerical_fallback_used=False,
        evaluation_seconds=time.perf_counter() - started,
    )


def shared_basis_from_calibration(
    calibration_blocks: Iterable[CompressedBlock],
    *,
    rank: int,
) -> np.ndarray:
    """Fit a shared basis only from an explicitly supplied calibration set."""
    residuals = []
    for block in calibration_blocks:
        keys = np.asarray(block.decode_key_block(), dtype=np.float64)
        residuals.append(keys - keys[0])
    if not residuals:
        raise ValueError("an explicit nonempty calibration set is required.")
    matrix = np.vstack(residuals)
    _left, _singular, right = np.linalg.svd(matrix, full_matrices=True)
    if rank < 0 or rank > matrix.shape[1]:
        raise ValueError("rank must be within [0, key dimension].")
    return np.asarray(right[:rank].T, dtype=np.float64)
