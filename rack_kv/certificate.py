from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import gmpy2
import numpy as np

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
    multiply_interval_by_nonnegative_scalar,
    norm_lower,
    norm_upper,
    rounded_add,
    rounded_div,
    rounded_dot,
    rounded_exp,
    rounded_mul,
    rounded_sub,
    rounded_sum,
    rounded_sqrt,
    subtract_point_from_interval,
    sqrt_dimension_interval,
)
from .types import CertificateMode, DecodeSchedule


@dataclass(frozen=True)
class CertificateCandidate:
    value: gmpy2.mpfr
    name: str


@dataclass(frozen=True)
class CertificateBoundSummary:
    z_k_lower: gmpy2.mpfr
    output_norm_upper: gmpy2.mpfr
    u_s_upper: gmpy2.mpfr
    nu_s_upper: gmpy2.mpfr
    w_s_upper: gmpy2.mpfr


@dataclass(frozen=True)
class CertificationResult:
    certified: bool
    mode: CertificateMode
    chosen_certificate: str
    certificate_value_mpfr: gmpy2.mpfr | None
    certificate_value_text: str | None
    certificate_value_upper_float: float | None
    decoded_block_starts: tuple[int, ...]
    skipped_block_starts: tuple[int, ...]
    iterations: int
    z_k_lower_text: str | None
    z_k_lower_upper_float: float | None
    u_s_upper_text: str | None
    u_s_upper_float: float | None
    nu_s_upper_text: str | None
    nu_s_upper_float: float | None
    kept_output_norm_upper_text: str | None
    kept_output_norm_upper_float: float | None
    output_vector_approx: np.ndarray
    output_mpfr_approx_text: tuple[str, ...]
    output_is_formally_certified: bool
    numerical_fallback_used: bool
    message: str


@dataclass(frozen=True)
class ProgressiveCertificationStep:
    chosen_certificate: str
    certificate_value_mpfr: gmpy2.mpfr
    certificate_value_text: str
    certificate_value_upper_float: float
    decoded_block_starts: tuple[int, ...]
    skipped_block_starts: tuple[int, ...]
    iterations: int
    z_k_lower_text: str | None
    z_k_lower_upper_float: float | None
    u_s_upper_text: str | None
    u_s_upper_float: float | None
    nu_s_upper_text: str | None
    nu_s_upper_float: float | None
    kept_output_norm_upper_text: str | None
    kept_output_norm_upper_float: float | None
    numerical_fallback_used: bool
    message: str


@dataclass(frozen=True)
class PreparedBlockExact:
    block: CompressedBlock
    anchor_exact: tuple[gmpy2.mpfr, ...]
    rho_exact: gmpy2.mpfr
    nu_exact: gmpy2.mpfr
    decoded_keys_exact: tuple[tuple[gmpy2.mpfr, ...], ...]
    decoded_values_exact: tuple[tuple[gmpy2.mpfr, ...], ...]


@dataclass(frozen=True)
class PreparedCertificationInputs:
    recent_keys_exact: tuple[tuple[gmpy2.mpfr, ...], ...]
    recent_values_exact: tuple[tuple[gmpy2.mpfr, ...], ...]
    historical_blocks_exact: tuple[PreparedBlockExact, ...]
    value_dim: int


def _mpfr_to_upper_float(value: gmpy2.mpfr | None) -> float | None:
    if value is None:
        return None
    if value == 0:
        return 0.0
    approximate = float(value)
    if math.isinf(approximate):
        return math.inf
    return float(np.nextafter(approximate, np.inf))


def _mpfr_vector_to_text(values: Sequence[gmpy2.mpfr]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _to_exact_vectors(matrix: np.ndarray, *, precision: int) -> list[list[gmpy2.mpfr]]:
    array = np.asarray(matrix)
    return [exact_vector(row, precision=precision) for row in array]


def _to_exact_vectors_tuple(matrix: np.ndarray, *, precision: int) -> tuple[tuple[gmpy2.mpfr, ...], ...]:
    return tuple(tuple(row) for row in _to_exact_vectors(matrix, precision=precision))


def _stack_rows(rows: Sequence[np.ndarray], width: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, width), dtype=np.float64)
    return np.vstack(rows).astype(np.float64)


def _prepare_block_exact(block: CompressedBlock, *, precision: int) -> PreparedBlockExact:
    decoded_keys, decoded_values = block.decode_block()
    return PreparedBlockExact(
        block=block,
        anchor_exact=tuple(exact_vector(block.header.anchor_key, precision=precision)),
        rho_exact=exact_mpfr(block.header.rho_upper, precision=precision),
        nu_exact=exact_mpfr(block.header.nu_upper, precision=precision),
        decoded_keys_exact=_to_exact_vectors_tuple(decoded_keys, precision=precision),
        decoded_values_exact=_to_exact_vectors_tuple(decoded_values, precision=precision),
    )


def prepare_block_exact(block: CompressedBlock, *, precision: int = DEFAULT_PRECISION) -> PreparedBlockExact:
    return _prepare_block_exact(block, precision=precision)


def prepare_certification_inputs(
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
    historical_blocks: Sequence[CompressedBlock],
    *,
    precision: int = DEFAULT_PRECISION,
) -> PreparedCertificationInputs:
    recent_keys_array = np.asarray(recent_keys, dtype=np.float64)
    recent_values_array = np.asarray(recent_values, dtype=np.float64)
    if recent_keys_array.ndim != 2 or recent_values_array.ndim != 2:
        raise ValueError("recent_keys and recent_values must be rank-2 arrays.")
    if recent_keys_array.shape[0] != recent_values_array.shape[0]:
        raise ValueError("recent_keys and recent_values must contain the same number of tokens.")
    if recent_keys_array.shape[0] > 0 and recent_keys_array.shape[1] <= 0:
        raise ValueError("recent_keys must have positive width when non-empty.")
    if recent_values_array.shape[0] > 0 and recent_values_array.shape[1] <= 0:
        raise ValueError("recent_values must have positive width when non-empty.")
    if not np.all(np.isfinite(recent_keys_array)):
        raise ValueError("recent_keys must be finite.")
    if not np.all(np.isfinite(recent_values_array)):
        raise ValueError("recent_values must be finite.")

    value_dim = int(recent_values_array.shape[1]) if recent_values_array.ndim == 2 else 0
    if value_dim == 0 and historical_blocks:
        value_dim = int(historical_blocks[0].header.anchor_value.shape[0])
    if value_dim <= 0:
        raise ValueError("A positive value dimension is required.")
    return PreparedCertificationInputs(
        recent_keys_exact=_to_exact_vectors_tuple(recent_keys_array, precision=precision),
        recent_values_exact=_to_exact_vectors_tuple(recent_values_array, precision=precision),
        historical_blocks_exact=tuple(_prepare_block_exact(block, precision=precision) for block in historical_blocks),
        value_dim=value_dim,
    )


def prepared_full_exact_rows(
    prepared: PreparedCertificationInputs,
) -> tuple[tuple[tuple[gmpy2.mpfr, ...], ...], tuple[tuple[gmpy2.mpfr, ...], ...]]:
    key_rows: list[tuple[gmpy2.mpfr, ...]] = []
    value_rows: list[tuple[gmpy2.mpfr, ...]] = []
    for block in prepared.historical_blocks_exact:
        key_rows.extend(block.decoded_keys_exact)
        value_rows.extend(block.decoded_values_exact)
    key_rows.extend(prepared.recent_keys_exact)
    value_rows.extend(prepared.recent_values_exact)
    return tuple(key_rows), tuple(value_rows)


def prepared_kept_exact_rows(
    prepared: PreparedCertificationInputs,
    *,
    decoded_block_starts: Sequence[int],
) -> tuple[tuple[tuple[gmpy2.mpfr, ...], ...], tuple[tuple[gmpy2.mpfr, ...], ...]]:
    decoded = set(int(value) for value in decoded_block_starts)
    key_rows: list[tuple[gmpy2.mpfr, ...]] = []
    value_rows: list[tuple[gmpy2.mpfr, ...]] = []
    for block in prepared.historical_blocks_exact:
        if int(block.block.header.block_start) in decoded:
            key_rows.extend(block.decoded_keys_exact)
            value_rows.extend(block.decoded_values_exact)
    key_rows.extend(prepared.recent_keys_exact)
    value_rows.extend(prepared.recent_values_exact)
    return tuple(key_rows), tuple(value_rows)


def _validate_inputs(
    *,
    query: np.ndarray,
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
    historical_blocks: Sequence[CompressedBlock],
    tolerance: float,
    precision: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_array = np.asarray(query, dtype=np.float64)
    recent_keys_array = np.asarray(recent_keys, dtype=np.float64)
    recent_values_array = np.asarray(recent_values, dtype=np.float64)

    if precision <= 0:
        raise ValueError("precision must be positive.")
    if query_array.ndim != 1 or query_array.size == 0:
        raise ValueError("query must be a non-empty rank-1 array.")
    if not np.all(np.isfinite(query_array)):
        raise ValueError("query must be finite.")
    if recent_keys_array.ndim != 2 or recent_values_array.ndim != 2:
        raise ValueError("recent_keys and recent_values must be rank-2 arrays.")
    if recent_keys_array.shape[0] != recent_values_array.shape[0]:
        raise ValueError("recent_keys and recent_values must contain the same number of tokens.")
    if recent_keys_array.shape[1] != query_array.shape[0]:
        raise ValueError("recent_keys width must match the query dimension.")
    if recent_values_array.shape[0] > 0 and not np.all(np.isfinite(recent_values_array)):
        raise ValueError("recent_values must be finite.")
    if recent_keys_array.shape[0] > 0 and not np.all(np.isfinite(recent_keys_array)):
        raise ValueError("recent_keys must be finite.")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")

    value_dim = recent_values_array.shape[1] if recent_values_array.shape[0] > 0 else None
    for block in historical_blocks:
        if int(block.header.anchor_key.shape[0]) != query_array.shape[0]:
            raise ValueError("All block key dimensions must match the query dimension.")
        if value_dim is None:
            value_dim = int(block.header.anchor_value.shape[0])
        elif int(block.header.anchor_value.shape[0]) != value_dim:
            raise ValueError("All block value dimensions must match.")
        if not np.isfinite(block.header.key_scale) or float(block.header.key_scale) == 0.0:
            raise ValueError("Block key scales must be finite and nonzero.")
        if not np.isfinite(block.header.value_scale) or float(block.header.value_scale) == 0.0:
            raise ValueError("Block value scales must be finite and nonzero.")
        if not np.isfinite(block.header.rho_upper) or float(block.header.rho_upper) < 0.0:
            raise ValueError("Block rho metadata must be finite and nonnegative.")
        if not np.isfinite(block.header.nu_upper) or float(block.header.nu_upper) < 0.0:
            raise ValueError("Block nu metadata must be finite and nonnegative.")

    if recent_values_array.shape[0] == 0 and value_dim is None and not historical_blocks:
        raise ValueError("At least one kept or historical token is required.")
    return query_array, recent_keys_array, recent_values_array


def _point_logit_interval(
    query: Sequence[gmpy2.mpfr],
    key: Sequence[gmpy2.mpfr],
    *,
    precision: int,
    attention_scale: float | None = None,
) -> Interval:
    numerator = dot_interval(query, key, precision=precision)
    if attention_scale is None:
        denominator = sqrt_dimension_interval(len(query), precision=precision)
        return divide_interval_by_positive_interval(numerator, denominator, precision=precision)
    scale = exact_mpfr(attention_scale, precision=precision)
    return multiply_interval_by_nonnegative_scalar(numerator, scale, precision=precision)


def _point_logit_interval_from_exact(
    query: Sequence[gmpy2.mpfr],
    key: Sequence[gmpy2.mpfr],
    *,
    precision: int,
    attention_scale: float | None = None,
) -> Interval:
    return _point_logit_interval(query, key, precision=precision, attention_scale=attention_scale)


def _block_logit_cap_interval(
    query: Sequence[gmpy2.mpfr],
    block: CompressedBlock,
    *,
    precision: int,
    attention_scale: float | None = None,
) -> Interval:
    anchor = exact_vector(block.header.anchor_key, precision=precision)
    dot_bounds = dot_interval(query, anchor, precision=precision)
    q_norm_bounds = Interval(
        lower=norm_lower(query, precision=precision),
        upper=norm_upper(query, precision=precision),
    )
    rho = exact_mpfr(block.header.rho_upper, precision=precision)
    radial_term = multiply_interval_by_nonnegative_scalar(q_norm_bounds, rho, precision=precision)
    numerator = add_intervals(dot_bounds, radial_term, precision=precision)
    if attention_scale is None:
        denominator = sqrt_dimension_interval(len(query), precision=precision)
        return divide_interval_by_positive_interval(numerator, denominator, precision=precision)
    scale = exact_mpfr(attention_scale, precision=precision)
    return multiply_interval_by_nonnegative_scalar(numerator, scale, precision=precision)


def _block_logit_cap_interval_prepared(
    query: Sequence[gmpy2.mpfr],
    query_norm_bounds: Interval,
    prepared_block: PreparedBlockExact,
    *,
    precision: int,
    attention_scale: float | None = None,
) -> Interval:
    dot_bounds = dot_interval(query, prepared_block.anchor_exact, precision=precision)
    radial_term = multiply_interval_by_nonnegative_scalar(query_norm_bounds, prepared_block.rho_exact, precision=precision)
    numerator = add_intervals(dot_bounds, radial_term, precision=precision)
    if attention_scale is None:
        denominator = sqrt_dimension_interval(len(query), precision=precision)
        return divide_interval_by_positive_interval(numerator, denominator, precision=precision)
    scale = exact_mpfr(attention_scale, precision=precision)
    return multiply_interval_by_nonnegative_scalar(numerator, scale, precision=precision)


def _shift_upper(
    query: Sequence[gmpy2.mpfr],
    kept_keys: Sequence[Sequence[gmpy2.mpfr]],
    remaining_blocks: Sequence[CompressedBlock],
    *,
    precision: int,
    attention_scale: float | None = None,
) -> gmpy2.mpfr:
    candidates: list[gmpy2.mpfr] = []
    for key in kept_keys:
        candidates.append(_point_logit_interval(query, key, precision=precision, attention_scale=attention_scale).upper)
    for block in remaining_blocks:
        candidates.append(
            _block_logit_cap_interval(query, block, precision=precision, attention_scale=attention_scale).upper
        )
    if not candidates:
        raise ValueError("Shift requires at least one kept token or remaining block.")
    return max_exact(candidates, precision=precision)


def _shift_upper_prepared(
    query: Sequence[gmpy2.mpfr],
    kept_keys: Sequence[Sequence[gmpy2.mpfr]],
    remaining_blocks: Sequence[PreparedBlockExact],
    query_norm_bounds: Interval,
    *,
    precision: int,
    attention_scale: float | None = None,
) -> gmpy2.mpfr:
    candidates: list[gmpy2.mpfr] = []
    for key in kept_keys:
        candidates.append(
            _point_logit_interval_from_exact(query, key, precision=precision, attention_scale=attention_scale).upper
        )
    for prepared_block in remaining_blocks:
        candidates.append(
            _block_logit_cap_interval_prepared(
                query,
                query_norm_bounds,
                prepared_block,
                precision=precision,
                attention_scale=attention_scale,
            ).upper
        )
    if not candidates:
        raise ValueError("Shift requires at least one kept token or remaining block.")
    return max_exact(candidates, precision=precision)


def _kept_mass_and_output_norm_upper(
    query: Sequence[gmpy2.mpfr],
    kept_keys: Sequence[Sequence[gmpy2.mpfr]],
    kept_values: Sequence[Sequence[gmpy2.mpfr]],
    shift: gmpy2.mpfr,
    *,
    precision: int,
    attention_scale: float | None = None,
) -> tuple[gmpy2.mpfr, gmpy2.mpfr]:
    if not kept_keys:
        raise ValueError("A positive kept set is required for certification.")

    weights_lower: list[gmpy2.mpfr] = []
    numerator_norm_terms: list[gmpy2.mpfr] = []
    for key, value in zip(kept_keys, kept_values):
        logit = _point_logit_interval(query, key, precision=precision, attention_scale=attention_scale)
        shifted_lower = rounded_sub(logit.lower, shift, precision=precision, round_mode=gmpy2.RoundDown)
        shifted_upper = rounded_sub(logit.upper, shift, precision=precision, round_mode=gmpy2.RoundUp)
        weight_lower = rounded_exp(shifted_lower, precision=precision, round_mode=gmpy2.RoundDown)
        weight_upper = rounded_exp(shifted_upper, precision=precision, round_mode=gmpy2.RoundUp)
        weights_lower.append(weight_lower)
        value_norm = norm_upper(value, precision=precision)
        numerator_norm_terms.append(rounded_mul(weight_upper, value_norm, precision=precision, round_mode=gmpy2.RoundUp))

    z_k_lower = rounded_sum(weights_lower, precision=precision, round_mode=gmpy2.RoundDown)
    if z_k_lower <= 0:
        raise ValueError("Kept-set mass lower bound must stay positive.")
    numerator_norm_upper = rounded_sum(numerator_norm_terms, precision=precision, round_mode=gmpy2.RoundUp)
    output_norm_upper = rounded_div(numerator_norm_upper, z_k_lower, precision=precision, round_mode=gmpy2.RoundUp)
    return z_k_lower, output_norm_upper


def _remaining_upper_bounds(
    query: Sequence[gmpy2.mpfr],
    remaining_blocks: Sequence[CompressedBlock],
    shift: gmpy2.mpfr,
    *,
    precision: int,
    attention_scale: float | None = None,
) -> tuple[list[gmpy2.mpfr], list[gmpy2.mpfr]]:
    upper_masses: list[gmpy2.mpfr] = []
    upper_value_norms: list[gmpy2.mpfr] = []
    for block in remaining_blocks:
        beta_upper = _block_logit_cap_interval(
            query,
            block,
            precision=precision,
            attention_scale=attention_scale,
        ).upper
        shifted = rounded_sub(beta_upper, shift, precision=precision, round_mode=gmpy2.RoundUp)
        mass = rounded_exp(shifted, precision=precision, round_mode=gmpy2.RoundUp)
        block_len = exact_mpfr(block.block_len, precision=precision)
        upper_masses.append(rounded_mul(block_len, mass, precision=precision, round_mode=gmpy2.RoundUp))
        upper_value_norms.append(exact_mpfr(block.header.nu_upper, precision=precision))
    return upper_masses, upper_value_norms


def _remaining_upper_bounds_prepared(
    query: Sequence[gmpy2.mpfr],
    remaining_blocks: Sequence[PreparedBlockExact],
    shift: gmpy2.mpfr,
    query_norm_bounds: Interval,
    *,
    precision: int,
    attention_scale: float | None = None,
) -> tuple[list[gmpy2.mpfr], list[gmpy2.mpfr]]:
    upper_masses: list[gmpy2.mpfr] = []
    upper_value_norms: list[gmpy2.mpfr] = []
    for prepared_block in remaining_blocks:
        beta_upper = _block_logit_cap_interval_prepared(
            query,
            query_norm_bounds,
            prepared_block,
            precision=precision,
            attention_scale=attention_scale,
        ).upper
        shifted = rounded_sub(beta_upper, shift, precision=precision, round_mode=gmpy2.RoundUp)
        mass = rounded_exp(shifted, precision=precision, round_mode=gmpy2.RoundUp)
        block_len = exact_mpfr(prepared_block.block.block_len, precision=precision)
        upper_masses.append(rounded_mul(block_len, mass, precision=precision, round_mode=gmpy2.RoundUp))
        upper_value_norms.append(prepared_block.nu_exact)
    return upper_masses, upper_value_norms


def _certificates_from_bounds(
    *,
    z_k_lower: gmpy2.mpfr,
    output_norm_upper: gmpy2.mpfr,
    upper_masses: Sequence[gmpy2.mpfr],
    upper_value_norms: Sequence[gmpy2.mpfr],
    precision: int,
) -> tuple[list[CertificateCandidate], CertificateBoundSummary]:
    if not upper_masses:
        zero = exact_mpfr(0, precision=precision)
        return [CertificateCandidate(value=zero, name="zero")], CertificateBoundSummary(
            z_k_lower=z_k_lower,
            output_norm_upper=output_norm_upper,
            u_s_upper=zero,
            nu_s_upper=zero,
            w_s_upper=zero,
        )

    u_s_upper = rounded_sum(upper_masses, precision=precision, round_mode=gmpy2.RoundUp)
    nu_s_upper = max_exact(list(upper_value_norms), precision=precision)
    w_s_terms = [
        rounded_mul(mass, nu, precision=precision, round_mode=gmpy2.RoundUp)
        for mass, nu in zip(upper_masses, upper_value_norms)
    ]
    w_s_upper = rounded_sum(w_s_terms, precision=precision, round_mode=gmpy2.RoundUp)

    denom1_down = rounded_add(z_k_lower, u_s_upper, precision=precision, round_mode=gmpy2.RoundDown)
    ratio1_upper = rounded_div(u_s_upper, denom1_down, precision=precision, round_mode=gmpy2.RoundUp)
    sum1_upper = rounded_add(nu_s_upper, output_norm_upper, precision=precision, round_mode=gmpy2.RoundUp)
    cert1 = rounded_mul(ratio1_upper, sum1_upper, precision=precision, round_mode=gmpy2.RoundUp)

    product_upper = rounded_mul(u_s_upper, output_norm_upper, precision=precision, round_mode=gmpy2.RoundUp)
    numerator2_upper = rounded_add(w_s_upper, product_upper, precision=precision, round_mode=gmpy2.RoundUp)
    cert2 = rounded_div(numerator2_upper, z_k_lower, precision=precision, round_mode=gmpy2.RoundUp)

    return [
        CertificateCandidate(value=cert1, name="cert1"),
        CertificateCandidate(value=cert2, name="cert2"),
    ], CertificateBoundSummary(
        z_k_lower=z_k_lower,
        output_norm_upper=output_norm_upper,
        u_s_upper=u_s_upper,
        nu_s_upper=nu_s_upper,
        w_s_upper=w_s_upper,
    )


def _choose_block_to_decode(
    upper_masses: Sequence[gmpy2.mpfr],
    upper_value_norms: Sequence[gmpy2.mpfr],
    *,
    schedule: DecodeSchedule,
    precision: int,
) -> int:
    if schedule == DecodeSchedule.LARGEST_U:
        return max(range(len(upper_masses)), key=lambda idx: upper_masses[idx])
    if schedule == DecodeSchedule.LARGEST_U_TIMES_NU:
        scores = [
            rounded_mul(mass, nu, precision=precision, round_mode=gmpy2.RoundUp)
            for mass, nu in zip(upper_masses, upper_value_norms)
        ]
        return max(range(len(scores)), key=lambda idx: scores[idx])
    raise ValueError(f"Unsupported decode schedule: {schedule}")


def exact_reference_output_mpfr(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
) -> list[gmpy2.mpfr]:
    query_array = np.asarray(query, dtype=np.float64)
    keys_array = np.asarray(keys, dtype=np.float64)
    values_array = np.asarray(values, dtype=np.float64)
    if keys_array.ndim != 2 or values_array.ndim != 2:
        raise ValueError("keys and values must be rank-2 arrays.")
    if keys_array.shape[0] != values_array.shape[0]:
        raise ValueError("keys and values must contain the same number of tokens.")
    if keys_array.shape[0] == 0:
        raise ValueError("At least one token is required to compute an attention output.")
    if keys_array.shape[1] != query_array.shape[0]:
        raise ValueError("Key width must match query dimension.")
    if attention_scale is not None and (not np.isfinite(attention_scale) or attention_scale <= 0.0):
        raise ValueError("attention_scale must be finite and positive when provided.")

    query_mpfr = exact_vector(query_array, precision=precision)
    key_rows = _to_exact_vectors(keys_array, precision=precision)
    value_rows = _to_exact_vectors(values_array, precision=precision)
    return _exact_reference_output_from_exact_vectors(
        query_mpfr,
        key_rows,
        value_rows,
        value_dim=values_array.shape[1],
        precision=precision,
        attention_scale=attention_scale,
    )


def _exact_reference_output_from_exact_vectors(
    query_mpfr: Sequence[gmpy2.mpfr],
    key_rows: Sequence[Sequence[gmpy2.mpfr]],
    value_rows: Sequence[Sequence[gmpy2.mpfr]],
    *,
    value_dim: int,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
) -> list[gmpy2.mpfr]:
    scale_exact = exact_mpfr(attention_scale, precision=precision) if attention_scale is not None else None
    sqrt_d = None
    if scale_exact is None:
        sqrt_d = rounded_sqrt(
            exact_mpfr(len(query_mpfr), precision=precision),
            precision=precision,
            round_mode=gmpy2.RoundToNearest,
        )

    logits = []
    for key in key_rows:
        dot = rounded_dot(query_mpfr, key, precision=precision, round_mode=gmpy2.RoundToNearest)
        if scale_exact is None:
            logits.append(rounded_div(dot, sqrt_d, precision=precision, round_mode=gmpy2.RoundToNearest))
        else:
            logits.append(rounded_mul(dot, scale_exact, precision=precision, round_mode=gmpy2.RoundToNearest))
    max_logit = max_exact(logits, precision=precision)
    shifted_weights = [
        rounded_exp(
            rounded_sub(logit, max_logit, precision=precision, round_mode=gmpy2.RoundToNearest),
            precision=precision,
            round_mode=gmpy2.RoundToNearest,
        )
        for logit in logits
    ]
    denominator = rounded_sum(shifted_weights, precision=precision, round_mode=gmpy2.RoundToNearest)
    if denominator == 0:
        raise ValueError("Reference denominator evaluated to zero.")

    output: list[gmpy2.mpfr] = []
    for component in range(value_dim):
        terms = [
            rounded_mul(weight, value[component], precision=precision, round_mode=gmpy2.RoundToNearest)
            for weight, value in zip(shifted_weights, value_rows)
        ]
        numerator = rounded_sum(terms, precision=precision, round_mode=gmpy2.RoundToNearest)
        output.append(rounded_div(numerator, denominator, precision=precision, round_mode=gmpy2.RoundToNearest))
    return output


def exact_reference_output(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
) -> np.ndarray:
    return np.asarray(
        [
            float(component)
            for component in exact_reference_output_mpfr(
                query,
                keys,
                values,
                precision=precision,
                attention_scale=attention_scale,
            )
        ]
    )


def exact_reference_output_from_exact_rows(
    query_exact: Sequence[gmpy2.mpfr],
    key_rows: Sequence[Sequence[gmpy2.mpfr]],
    value_rows: Sequence[Sequence[gmpy2.mpfr]],
    *,
    value_dim: int,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
) -> list[gmpy2.mpfr]:
    return _exact_reference_output_from_exact_vectors(
        query_exact,
        key_rows,
        value_rows,
        value_dim=value_dim,
        precision=precision,
        attention_scale=attention_scale,
    )


def rigorous_attention_output_interval(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
) -> tuple[tuple[Interval, ...], gmpy2.mpfr]:
    query_array = np.asarray(query, dtype=np.float64)
    keys_array = np.asarray(keys, dtype=np.float64)
    values_array = np.asarray(values, dtype=np.float64)
    if keys_array.ndim != 2 or values_array.ndim != 2:
        raise ValueError("keys and values must be rank-2 arrays.")
    if keys_array.shape[0] != values_array.shape[0]:
        raise ValueError("keys and values must contain the same number of tokens.")
    if keys_array.shape[0] == 0:
        raise ValueError("At least one token is required to compute an attention output interval.")
    if keys_array.shape[1] != query_array.shape[0]:
        raise ValueError("Key width must match query dimension.")
    if attention_scale is not None and (not np.isfinite(attention_scale) or attention_scale <= 0.0):
        raise ValueError("attention_scale must be finite and positive when provided.")

    query_exact = exact_vector(query_array, precision=precision)
    key_rows = _to_exact_vectors(keys_array, precision=precision)
    value_rows = _to_exact_vectors(values_array, precision=precision)
    return _rigorous_attention_output_interval_from_exact_vectors(
        query_exact,
        key_rows,
        value_rows,
        value_dim=values_array.shape[1],
        precision=precision,
        attention_scale=attention_scale,
    )


def rigorous_attention_output_interval_from_exact_rows(
    query_exact: Sequence[gmpy2.mpfr],
    key_rows: Sequence[Sequence[gmpy2.mpfr]],
    value_rows: Sequence[Sequence[gmpy2.mpfr]],
    *,
    value_dim: int,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
) -> tuple[tuple[Interval, ...], gmpy2.mpfr]:
    return _rigorous_attention_output_interval_from_exact_vectors(
        query_exact,
        key_rows,
        value_rows,
        value_dim=value_dim,
        precision=precision,
        attention_scale=attention_scale,
    )


def _rigorous_attention_output_interval_from_exact_vectors(
    query_exact: Sequence[gmpy2.mpfr],
    key_rows: Sequence[Sequence[gmpy2.mpfr]],
    value_rows: Sequence[Sequence[gmpy2.mpfr]],
    *,
    value_dim: int,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
) -> tuple[tuple[Interval, ...], gmpy2.mpfr]:
    logit_intervals = [
        _point_logit_interval(query_exact, key, precision=precision, attention_scale=attention_scale) for key in key_rows
    ]
    shift = max_exact([interval.upper for interval in logit_intervals], precision=precision)

    weight_intervals: list[Interval] = []
    for interval in logit_intervals:
        shifted_interval = subtract_point_from_interval(interval, shift, precision=precision)
        weight_intervals.append(
            Interval(
                lower=rounded_exp(shifted_interval.lower, precision=precision, round_mode=gmpy2.RoundDown),
                upper=rounded_exp(shifted_interval.upper, precision=precision, round_mode=gmpy2.RoundUp),
            )
        )

    denominator_interval = Interval(
        lower=rounded_sum((interval.lower for interval in weight_intervals), precision=precision, round_mode=gmpy2.RoundDown),
        upper=rounded_sum((interval.upper for interval in weight_intervals), precision=precision, round_mode=gmpy2.RoundUp),
    )
    if denominator_interval.lower <= 0:
        raise ValueError("Attention denominator interval must stay positive.")

    output_intervals: list[Interval] = []
    for component in range(value_dim):
        lower_terms: list[gmpy2.mpfr] = []
        upper_terms: list[gmpy2.mpfr] = []
        for weight_interval, value_row in zip(weight_intervals, value_rows):
            component_value = value_row[component]
            if component_value >= 0:
                lower_terms.append(
                    rounded_mul(weight_interval.lower, component_value, precision=precision, round_mode=gmpy2.RoundDown)
                )
                upper_terms.append(
                    rounded_mul(weight_interval.upper, component_value, precision=precision, round_mode=gmpy2.RoundUp)
                )
            else:
                lower_terms.append(
                    rounded_mul(weight_interval.upper, component_value, precision=precision, round_mode=gmpy2.RoundDown)
                )
                upper_terms.append(
                    rounded_mul(weight_interval.lower, component_value, precision=precision, round_mode=gmpy2.RoundUp)
                )
        numerator_interval = Interval(
            lower=rounded_sum(lower_terms, precision=precision, round_mode=gmpy2.RoundDown),
            upper=rounded_sum(upper_terms, precision=precision, round_mode=gmpy2.RoundUp),
        )
        output_intervals.append(
            divide_interval_by_positive_interval(numerator_interval, denominator_interval, precision=precision)
        )
    return tuple(output_intervals), shift


def rigorous_output_error_upper_from_intervals(
    full_output: Sequence[Interval],
    kept_output: Sequence[Interval],
    *,
    precision: int = DEFAULT_PRECISION,
) -> gmpy2.mpfr:
    if len(full_output) != len(kept_output):
        raise ValueError("Output interval vectors must have matching dimensions.")
    squared_terms: list[gmpy2.mpfr] = []
    for full_component, kept_component in zip(full_output, kept_output):
        diff_lower = rounded_sub(full_component.lower, kept_component.upper, precision=precision, round_mode=gmpy2.RoundDown)
        diff_upper = rounded_sub(full_component.upper, kept_component.lower, precision=precision, round_mode=gmpy2.RoundUp)
        abs_lower = -diff_lower if diff_lower < 0 else diff_lower
        abs_upper = -diff_upper if diff_upper < 0 else diff_upper
        abs_diff_upper = max_exact([abs_lower, abs_upper], precision=precision)
        squared_terms.append(rounded_mul(abs_diff_upper, abs_diff_upper, precision=precision, round_mode=gmpy2.RoundUp))
    squared_sum = rounded_sum(squared_terms, precision=precision, round_mode=gmpy2.RoundUp)
    return rounded_sqrt(squared_sum, precision=precision, round_mode=gmpy2.RoundUp)


def rigorous_output_error_norm(
    full_output: Sequence[gmpy2.mpfr],
    kept_output: Sequence[gmpy2.mpfr],
    *,
    precision: int = DEFAULT_PRECISION,
) -> gmpy2.mpfr:
    if len(full_output) != len(kept_output):
        raise ValueError("Output vectors must have matching dimensions.")
    differences = [
        rounded_sub(full_component, kept_component, precision=precision, round_mode=gmpy2.RoundToNearest)
        for full_component, kept_component in zip(full_output, kept_output)
    ]
    return norm_upper(differences, precision=precision)


def _zero_certificate_result(
    *,
    mode: CertificateMode,
    decoded_block_starts: Sequence[int],
    output_vector_approx: np.ndarray,
    output_mpfr_approx: Sequence[gmpy2.mpfr],
    iterations: int,
    message: str,
    numerical_fallback_used: bool,
    precision: int,
) -> CertificationResult:
    zero = exact_mpfr(0, precision=precision)
    return CertificationResult(
        certified=True,
        mode=mode,
        chosen_certificate="zero",
        certificate_value_mpfr=zero,
        certificate_value_text=str(zero),
        certificate_value_upper_float=0.0,
        decoded_block_starts=tuple(decoded_block_starts),
        skipped_block_starts=tuple(),
        iterations=iterations,
        z_k_lower_text=None,
        z_k_lower_upper_float=None,
        u_s_upper_text=None,
        u_s_upper_float=None,
        nu_s_upper_text=None,
        nu_s_upper_float=None,
        kept_output_norm_upper_text=None,
        kept_output_norm_upper_float=None,
        output_vector_approx=output_vector_approx,
        output_mpfr_approx_text=_mpfr_vector_to_text(output_mpfr_approx),
        output_is_formally_certified=False,
        numerical_fallback_used=numerical_fallback_used,
        message=message,
    )


def _failure_result(*, mode: CertificateMode, message: str) -> CertificationResult:
    return CertificationResult(
        certified=False,
        mode=mode,
        chosen_certificate="failure",
        certificate_value_mpfr=None,
        certificate_value_text=None,
        certificate_value_upper_float=None,
        decoded_block_starts=tuple(),
        skipped_block_starts=tuple(),
        iterations=0,
        z_k_lower_text=None,
        z_k_lower_upper_float=None,
        u_s_upper_text=None,
        u_s_upper_float=None,
        nu_s_upper_text=None,
        nu_s_upper_float=None,
        kept_output_norm_upper_text=None,
        kept_output_norm_upper_float=None,
        output_vector_approx=np.zeros((0,), dtype=np.float64),
        output_mpfr_approx_text=tuple(),
        output_is_formally_certified=False,
        numerical_fallback_used=False,
        message=message,
    )


def _zero_progress_step(
    *,
    decoded_block_starts: Sequence[int],
    iterations: int,
    message: str,
    numerical_fallback_used: bool,
    precision: int,
) -> ProgressiveCertificationStep:
    zero = exact_mpfr(0, precision=precision)
    return ProgressiveCertificationStep(
        chosen_certificate="zero",
        certificate_value_mpfr=zero,
        certificate_value_text=str(zero),
        certificate_value_upper_float=0.0,
        decoded_block_starts=tuple(decoded_block_starts),
        skipped_block_starts=tuple(),
        iterations=iterations,
        z_k_lower_text=None,
        z_k_lower_upper_float=None,
        u_s_upper_text=None,
        u_s_upper_float=None,
        nu_s_upper_text=None,
        nu_s_upper_float=None,
        kept_output_norm_upper_text=None,
        kept_output_norm_upper_float=None,
        numerical_fallback_used=numerical_fallback_used,
        message=message,
    )


def progressive_certification_steps(
    *,
    query: np.ndarray,
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
    historical_blocks: Sequence[CompressedBlock],
    schedule: DecodeSchedule = DecodeSchedule.LARGEST_U_TIMES_NU,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
) -> tuple[ProgressiveCertificationStep, ...]:
    query_array, recent_keys_array, recent_values_array = _validate_inputs(
        query=query,
        recent_keys=recent_keys,
        recent_values=recent_values,
        historical_blocks=historical_blocks,
        tolerance=0.0,
        precision=precision,
    )
    if attention_scale is not None and (not np.isfinite(attention_scale) or attention_scale <= 0.0):
        raise ValueError("attention_scale must be finite and positive when provided.")

    kept_key_rows_exact: list[tuple[gmpy2.mpfr, ...]] = list(_to_exact_vectors_tuple(recent_keys_array, precision=precision))
    kept_value_rows_exact: list[tuple[gmpy2.mpfr, ...]] = list(_to_exact_vectors_tuple(recent_values_array, precision=precision))
    remaining = [_prepare_block_exact(block, precision=precision) for block in historical_blocks]
    decoded_block_starts: list[int] = []
    query_exact = exact_vector(query_array, precision=precision)
    query_norm_bounds = Interval(
        lower=norm_lower(query_exact, precision=precision),
        upper=norm_upper(query_exact, precision=precision),
    )

    if not kept_key_rows_exact and remaining:
        decoded = remaining.pop(0)
        kept_key_rows_exact.extend(decoded.decoded_keys_exact)
        kept_value_rows_exact.extend(decoded.decoded_values_exact)
        decoded_block_starts.append(decoded.block.header.block_start)

    if not kept_key_rows_exact:
        raise ValueError("Certification requires at least one kept token or decodable block.")

    def _token_bounds(
        key_exact: Sequence[gmpy2.mpfr],
        value_exact: Sequence[gmpy2.mpfr],
        *,
        shift: gmpy2.mpfr,
    ) -> tuple[gmpy2.mpfr, gmpy2.mpfr]:
        logit = _point_logit_interval_from_exact(
            query_exact,
            key_exact,
            precision=precision,
            attention_scale=attention_scale,
        )
        shifted_lower = rounded_sub(logit.lower, shift, precision=precision, round_mode=gmpy2.RoundDown)
        shifted_upper = rounded_sub(logit.upper, shift, precision=precision, round_mode=gmpy2.RoundUp)
        weight_lower = rounded_exp(shifted_lower, precision=precision, round_mode=gmpy2.RoundDown)
        weight_upper = rounded_exp(shifted_upper, precision=precision, round_mode=gmpy2.RoundUp)
        value_norm_upper = norm_upper(value_exact, precision=precision)
        numerator_upper = rounded_mul(weight_upper, value_norm_upper, precision=precision, round_mode=gmpy2.RoundUp)
        return weight_lower, numerator_upper

    steps: list[ProgressiveCertificationStep] = []
    iterations = 0
    try:
        kept_entries: list[dict[str, object]] = []
        for key_exact, value_exact in zip(kept_key_rows_exact, kept_value_rows_exact):
            kept_entries.append(
                {
                    "logit_interval": _point_logit_interval_from_exact(
                        query_exact,
                        key_exact,
                        precision=precision,
                        attention_scale=attention_scale,
                    ),
                    "value_norm_upper": norm_upper(value_exact, precision=precision),
                }
            )

        block_cache: list[dict[str, object]] = []
        for prepared_block in remaining:
            block_cap_upper = _block_logit_cap_interval_prepared(
                query_exact,
                query_norm_bounds,
                prepared_block,
                precision=precision,
                attention_scale=attention_scale,
            ).upper
            token_entries: list[dict[str, object]] = []
            for key_exact, value_exact in zip(prepared_block.decoded_keys_exact, prepared_block.decoded_values_exact):
                token_entries.append(
                    {
                        "logit_interval": _point_logit_interval_from_exact(
                            query_exact,
                            key_exact,
                            precision=precision,
                            attention_scale=attention_scale,
                        ),
                        "value_norm_upper": norm_upper(value_exact, precision=precision),
                    }
                )
            block_cache.append(
                {
                    "prepared_block": prepared_block,
                    "block_cap_upper": block_cap_upper,
                    "block_len_exact": exact_mpfr(prepared_block.block.block_len, precision=precision),
                    "nu_exact": prepared_block.nu_exact,
                    "token_entries": token_entries,
                }
            )
    except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError) as error:
        for prepared_block in remaining:
            decoded_block_starts.append(prepared_block.block.header.block_start)
        steps.append(
            _zero_progress_step(
                decoded_block_starts=decoded_block_starts,
                iterations=iterations,
                message=f"Decoded all remaining blocks after numerical-certification fallback: {error}",
                numerical_fallback_used=True,
                precision=precision,
            )
        )
        return tuple(steps)

    while block_cache:
        iterations += 1
        try:
            shift_candidates = [entry["logit_interval"].upper for entry in kept_entries]
            shift_candidates.extend(entry["block_cap_upper"] for entry in block_cache)
            shift = max_exact(shift_candidates, precision=precision)

            z_k_lower = exact_mpfr(0, precision=precision)
            numerator_norm_upper = exact_mpfr(0, precision=precision)
            for entry in kept_entries:
                logit_interval = entry["logit_interval"]
                shifted_lower = rounded_sub(logit_interval.lower, shift, precision=precision, round_mode=gmpy2.RoundDown)
                shifted_upper = rounded_sub(logit_interval.upper, shift, precision=precision, round_mode=gmpy2.RoundUp)
                weight_lower = rounded_exp(shifted_lower, precision=precision, round_mode=gmpy2.RoundDown)
                weight_upper = rounded_exp(shifted_upper, precision=precision, round_mode=gmpy2.RoundUp)
                z_k_lower = rounded_add(z_k_lower, weight_lower, precision=precision, round_mode=gmpy2.RoundDown)
                numerator_norm_upper = rounded_add(
                    numerator_norm_upper,
                    rounded_mul(weight_upper, entry["value_norm_upper"], precision=precision, round_mode=gmpy2.RoundUp),
                    precision=precision,
                    round_mode=gmpy2.RoundUp,
                )
            if z_k_lower <= 0:
                raise ValueError("Kept-set mass lower bound must stay positive.")

            output_norm_upper = rounded_div(
                numerator_norm_upper,
                z_k_lower,
                precision=precision,
                round_mode=gmpy2.RoundUp,
            )
            upper_masses = []
            for entry in block_cache:
                shifted = rounded_sub(entry["block_cap_upper"], shift, precision=precision, round_mode=gmpy2.RoundUp)
                mass = rounded_exp(shifted, precision=precision, round_mode=gmpy2.RoundUp)
                upper_masses.append(
                    rounded_mul(entry["block_len_exact"], mass, precision=precision, round_mode=gmpy2.RoundUp)
                )
            upper_value_norms = [entry["nu_exact"] for entry in block_cache]
            candidates, bound_summary = _certificates_from_bounds(
                z_k_lower=z_k_lower,
                output_norm_upper=output_norm_upper,
                upper_masses=upper_masses,
                upper_value_norms=upper_value_norms,
                precision=precision,
            )
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError) as error:
            for entry in block_cache:
                prepared_block = entry["prepared_block"]
                decoded_block_starts.append(prepared_block.block.header.block_start)
            steps.append(
                _zero_progress_step(
                    decoded_block_starts=decoded_block_starts,
                    iterations=iterations,
                    message=f"Decoded all remaining blocks after numerical-certification fallback: {error}",
                    numerical_fallback_used=True,
                    precision=precision,
                )
            )
            return tuple(steps)

        chosen = min(candidates, key=lambda candidate: candidate.value)
        steps.append(
            ProgressiveCertificationStep(
                chosen_certificate=chosen.name,
                certificate_value_mpfr=chosen.value,
                certificate_value_text=str(chosen.value),
                certificate_value_upper_float=_mpfr_to_upper_float(chosen.value) or 0.0,
                decoded_block_starts=tuple(decoded_block_starts),
                skipped_block_starts=tuple(
                    entry["prepared_block"].block.header.block_start for entry in block_cache
                ),
                iterations=iterations,
                z_k_lower_text=str(bound_summary.z_k_lower),
                z_k_lower_upper_float=_mpfr_to_upper_float(bound_summary.z_k_lower),
                u_s_upper_text=str(bound_summary.u_s_upper),
                u_s_upper_float=_mpfr_to_upper_float(bound_summary.u_s_upper),
                nu_s_upper_text=str(bound_summary.nu_s_upper),
                nu_s_upper_float=_mpfr_to_upper_float(bound_summary.nu_s_upper),
                kept_output_norm_upper_text=str(bound_summary.output_norm_upper),
                kept_output_norm_upper_float=_mpfr_to_upper_float(bound_summary.output_norm_upper),
                numerical_fallback_used=False,
                message="Certified skip against the fully reconstructed compressed representation.",
            )
        )

        if schedule == DecodeSchedule.LARGEST_U:
            decode_index = max(range(len(block_cache)), key=lambda idx: upper_masses[idx])
        elif schedule == DecodeSchedule.LARGEST_U_TIMES_NU:
            decode_scores = [
                rounded_mul(upper_masses[idx], upper_value_norms[idx], precision=precision, round_mode=gmpy2.RoundUp)
                for idx in range(len(block_cache))
            ]
            decode_index = max(range(len(block_cache)), key=lambda idx: decode_scores[idx])
        else:
            raise ValueError(f"Unsupported decode schedule: {schedule}")
        decoded_entry = block_cache.pop(decode_index)
        prepared_block = decoded_entry["prepared_block"]
        decoded_block_starts.append(prepared_block.block.header.block_start)
        kept_entries.extend(decoded_entry["token_entries"])

    steps.append(
        _zero_progress_step(
            decoded_block_starts=decoded_block_starts,
            iterations=iterations,
            message="All historical blocks decoded; certified skip error is zero.",
            numerical_fallback_used=False,
            precision=precision,
        )
    )
    return tuple(steps)


def certify_progressive_skipping(
    *,
    query: np.ndarray,
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
    historical_blocks: Sequence[CompressedBlock],
    tolerance: float,
    mode: CertificateMode = CertificateMode.RIGOROUS_REFERENCE,
    schedule: DecodeSchedule = DecodeSchedule.LARGEST_U_TIMES_NU,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
) -> CertificationResult:
    if mode != CertificateMode.RIGOROUS_REFERENCE:
        return _failure_result(mode=mode, message="Only rigorous_reference may issue formal certificates.")

    query_array, recent_keys_array, recent_values_array = _validate_inputs(
        query=query,
        recent_keys=recent_keys,
        recent_values=recent_values,
        historical_blocks=historical_blocks,
        tolerance=tolerance,
        precision=precision,
    )
    if attention_scale is not None and (not np.isfinite(attention_scale) or attention_scale <= 0.0):
        raise ValueError("attention_scale must be finite and positive when provided.")

    kept_key_rows_exact: list[tuple[gmpy2.mpfr, ...]] = list(_to_exact_vectors_tuple(recent_keys_array, precision=precision))
    kept_value_rows_exact: list[tuple[gmpy2.mpfr, ...]] = list(_to_exact_vectors_tuple(recent_values_array, precision=precision))
    remaining = [_prepare_block_exact(block, precision=precision) for block in historical_blocks]
    decoded_block_starts: list[int] = []
    tolerance_exact = exact_mpfr(tolerance, precision=precision)
    query_exact = exact_vector(query_array, precision=precision)
    query_norm_bounds = Interval(
        lower=norm_lower(query_exact, precision=precision),
        upper=norm_upper(query_exact, precision=precision),
    )

    if not kept_key_rows_exact and remaining:
        decoded = remaining.pop(0)
        kept_key_rows_exact.extend(decoded.decoded_keys_exact)
        kept_value_rows_exact.extend(decoded.decoded_values_exact)
        decoded_block_starts.append(decoded.block.header.block_start)

    if not kept_key_rows_exact:
        return _failure_result(mode=mode, message="Certification requires at least one kept token or decodable block.")

    iterations = 0
    while remaining:
        iterations += 1
        try:
            shift = _shift_upper_prepared(
                query_exact,
                kept_key_rows_exact,
                remaining,
                query_norm_bounds,
                precision=precision,
                attention_scale=attention_scale,
            )
            z_k_lower, output_norm_upper = _kept_mass_and_output_norm_upper(
                query_exact,
                kept_key_rows_exact,
                kept_value_rows_exact,
                shift,
                precision=precision,
                attention_scale=attention_scale,
            )
            upper_masses, upper_value_norms = _remaining_upper_bounds(
                query_exact,
                [prepared_block.block for prepared_block in remaining],
                shift,
                precision=precision,
                attention_scale=attention_scale,
            )
            candidates, bound_summary = _certificates_from_bounds(
                z_k_lower=z_k_lower,
                output_norm_upper=output_norm_upper,
                upper_masses=upper_masses,
                upper_value_norms=upper_value_norms,
                precision=precision,
            )
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError) as error:
            for prepared_block in remaining:
                kept_key_rows_exact.extend(prepared_block.decoded_keys_exact)
                kept_value_rows_exact.extend(prepared_block.decoded_values_exact)
                decoded_block_starts.append(prepared_block.block.header.block_start)
            value_dim = len(kept_value_rows_exact[0])
            output_mpfr = _exact_reference_output_from_exact_vectors(
                query_exact,
                kept_key_rows_exact,
                kept_value_rows_exact,
                value_dim=value_dim,
                precision=precision,
                attention_scale=attention_scale,
            )
            output = np.asarray([float(component) for component in output_mpfr], dtype=np.float64)
            return _zero_certificate_result(
                mode=mode,
                decoded_block_starts=decoded_block_starts,
                output_vector_approx=output,
                output_mpfr_approx=output_mpfr,
                iterations=iterations,
                message=f"Decoded all remaining blocks after numerical-certification fallback: {error}",
                numerical_fallback_used=True,
                precision=precision,
            )

        chosen = min(candidates, key=lambda candidate: candidate.value)
        if chosen.value <= tolerance_exact:
            value_dim = len(kept_value_rows_exact[0])
            output_mpfr = _exact_reference_output_from_exact_vectors(
                query_exact,
                kept_key_rows_exact,
                kept_value_rows_exact,
                value_dim=value_dim,
                precision=precision,
                attention_scale=attention_scale,
            )
            output = np.asarray([float(component) for component in output_mpfr], dtype=np.float64)
            return CertificationResult(
                certified=True,
                mode=mode,
                chosen_certificate=chosen.name,
                certificate_value_mpfr=chosen.value,
                certificate_value_text=str(chosen.value),
                certificate_value_upper_float=_mpfr_to_upper_float(chosen.value),
                decoded_block_starts=tuple(decoded_block_starts),
                skipped_block_starts=tuple(prepared_block.block.header.block_start for prepared_block in remaining),
                iterations=iterations,
                z_k_lower_text=str(bound_summary.z_k_lower),
                z_k_lower_upper_float=_mpfr_to_upper_float(bound_summary.z_k_lower),
                u_s_upper_text=str(bound_summary.u_s_upper),
                u_s_upper_float=_mpfr_to_upper_float(bound_summary.u_s_upper),
                nu_s_upper_text=str(bound_summary.nu_s_upper),
                nu_s_upper_float=_mpfr_to_upper_float(bound_summary.nu_s_upper),
                kept_output_norm_upper_text=str(bound_summary.output_norm_upper),
                kept_output_norm_upper_float=_mpfr_to_upper_float(bound_summary.output_norm_upper),
                output_vector_approx=output,
                output_mpfr_approx_text=_mpfr_vector_to_text(output_mpfr),
                output_is_formally_certified=False,
                numerical_fallback_used=False,
                message="Certified skip against the fully reconstructed compressed representation.",
            )

        decode_index = _choose_block_to_decode(
            upper_masses,
            upper_value_norms,
            schedule=schedule,
            precision=precision,
        )
        decoded = remaining.pop(decode_index)
        kept_key_rows_exact.extend(decoded.decoded_keys_exact)
        kept_value_rows_exact.extend(decoded.decoded_values_exact)
        decoded_block_starts.append(decoded.block.header.block_start)

    value_dim = len(kept_value_rows_exact[0])
    output_mpfr = _exact_reference_output_from_exact_vectors(
        query_exact,
        kept_key_rows_exact,
        kept_value_rows_exact,
        value_dim=value_dim,
        precision=precision,
        attention_scale=attention_scale,
    )
    output = np.asarray([float(component) for component in output_mpfr], dtype=np.float64)
    return _zero_certificate_result(
        mode=mode,
        decoded_block_starts=decoded_block_starts,
        output_vector_approx=output,
        output_mpfr_approx=output_mpfr,
        iterations=iterations,
        message="All historical blocks decoded; certified skip error is zero.",
        numerical_fallback_used=False,
        precision=precision,
    )


def certify_progressive_skipping_prepared(
    *,
    query: np.ndarray,
    prepared_inputs: PreparedCertificationInputs,
    tolerance: float,
    mode: CertificateMode = CertificateMode.RIGOROUS_REFERENCE,
    schedule: DecodeSchedule = DecodeSchedule.LARGEST_U_TIMES_NU,
    precision: int = DEFAULT_PRECISION,
    attention_scale: float | None = None,
) -> CertificationResult:
    if mode != CertificateMode.RIGOROUS_REFERENCE:
        return _failure_result(mode=mode, message="Only rigorous_reference may issue formal certificates.")

    query_array = np.asarray(query, dtype=np.float64)
    if precision <= 0:
        raise ValueError("precision must be positive.")
    if query_array.ndim != 1 or query_array.size == 0:
        raise ValueError("query must be a non-empty rank-1 array.")
    if not np.all(np.isfinite(query_array)):
        raise ValueError("query must be finite.")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    if attention_scale is not None and (not np.isfinite(attention_scale) or attention_scale <= 0.0):
        raise ValueError("attention_scale must be finite and positive when provided.")

    kept_key_rows_exact: list[tuple[gmpy2.mpfr, ...]] = list(prepared_inputs.recent_keys_exact)
    kept_value_rows_exact: list[tuple[gmpy2.mpfr, ...]] = list(prepared_inputs.recent_values_exact)
    remaining = list(prepared_inputs.historical_blocks_exact)
    decoded_block_starts: list[int] = []
    tolerance_exact = exact_mpfr(tolerance, precision=precision)
    query_exact = exact_vector(query_array, precision=precision)
    query_norm_bounds = Interval(
        lower=norm_lower(query_exact, precision=precision),
        upper=norm_upper(query_exact, precision=precision),
    )

    if not kept_key_rows_exact and remaining:
        decoded = remaining.pop(0)
        kept_key_rows_exact.extend(decoded.decoded_keys_exact)
        kept_value_rows_exact.extend(decoded.decoded_values_exact)
        decoded_block_starts.append(decoded.block.header.block_start)

    if not kept_key_rows_exact:
        return _failure_result(mode=mode, message="Certification requires at least one kept token or decodable block.")

    iterations = 0
    while remaining:
        iterations += 1
        try:
            shift = _shift_upper_prepared(
                query_exact,
                kept_key_rows_exact,
                remaining,
                query_norm_bounds,
                precision=precision,
                attention_scale=attention_scale,
            )
            z_k_lower, output_norm_upper = _kept_mass_and_output_norm_upper(
                query_exact,
                kept_key_rows_exact,
                kept_value_rows_exact,
                shift,
                precision=precision,
                attention_scale=attention_scale,
            )
            upper_masses, upper_value_norms = _remaining_upper_bounds_prepared(
                query_exact,
                remaining,
                shift,
                query_norm_bounds,
                precision=precision,
                attention_scale=attention_scale,
            )
            candidates, bound_summary = _certificates_from_bounds(
                z_k_lower=z_k_lower,
                output_norm_upper=output_norm_upper,
                upper_masses=upper_masses,
                upper_value_norms=upper_value_norms,
                precision=precision,
            )
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError) as error:
            for prepared_block in remaining:
                kept_key_rows_exact.extend(prepared_block.decoded_keys_exact)
                kept_value_rows_exact.extend(prepared_block.decoded_values_exact)
                decoded_block_starts.append(prepared_block.block.header.block_start)
            output_mpfr = _exact_reference_output_from_exact_vectors(
                query_exact,
                kept_key_rows_exact,
                kept_value_rows_exact,
                value_dim=prepared_inputs.value_dim,
                precision=precision,
                attention_scale=attention_scale,
            )
            output = np.asarray([float(component) for component in output_mpfr], dtype=np.float64)
            return _zero_certificate_result(
                mode=mode,
                decoded_block_starts=decoded_block_starts,
                output_vector_approx=output,
                output_mpfr_approx=output_mpfr,
                iterations=iterations,
                message=f"Decoded all remaining blocks after numerical-certification fallback: {error}",
                numerical_fallback_used=True,
                precision=precision,
            )

        chosen = min(candidates, key=lambda candidate: candidate.value)
        if chosen.value <= tolerance_exact:
            output_mpfr = _exact_reference_output_from_exact_vectors(
                query_exact,
                kept_key_rows_exact,
                kept_value_rows_exact,
                value_dim=prepared_inputs.value_dim,
                precision=precision,
                attention_scale=attention_scale,
            )
            output = np.asarray([float(component) for component in output_mpfr], dtype=np.float64)
            return CertificationResult(
                certified=True,
                mode=mode,
                chosen_certificate=chosen.name,
                certificate_value_mpfr=chosen.value,
                certificate_value_text=str(chosen.value),
                certificate_value_upper_float=_mpfr_to_upper_float(chosen.value),
                decoded_block_starts=tuple(decoded_block_starts),
                skipped_block_starts=tuple(prepared_block.block.header.block_start for prepared_block in remaining),
                iterations=iterations,
                z_k_lower_text=str(bound_summary.z_k_lower),
                z_k_lower_upper_float=_mpfr_to_upper_float(bound_summary.z_k_lower),
                u_s_upper_text=str(bound_summary.u_s_upper),
                u_s_upper_float=_mpfr_to_upper_float(bound_summary.u_s_upper),
                nu_s_upper_text=str(bound_summary.nu_s_upper),
                nu_s_upper_float=_mpfr_to_upper_float(bound_summary.nu_s_upper),
                kept_output_norm_upper_text=str(bound_summary.output_norm_upper),
                kept_output_norm_upper_float=_mpfr_to_upper_float(bound_summary.output_norm_upper),
                output_vector_approx=output,
                output_mpfr_approx_text=_mpfr_vector_to_text(output_mpfr),
                output_is_formally_certified=False,
                numerical_fallback_used=False,
                message="Certified skip against the fully reconstructed compressed representation.",
            )

        decode_index = _choose_block_to_decode(
            upper_masses,
            upper_value_norms,
            schedule=schedule,
            precision=precision,
        )
        decoded = remaining.pop(decode_index)
        kept_key_rows_exact.extend(decoded.decoded_keys_exact)
        kept_value_rows_exact.extend(decoded.decoded_values_exact)
        decoded_block_starts.append(decoded.block.header.block_start)

    output_mpfr = _exact_reference_output_from_exact_vectors(
        query_exact,
        kept_key_rows_exact,
        kept_value_rows_exact,
        value_dim=prepared_inputs.value_dim,
        precision=precision,
        attention_scale=attention_scale,
    )
    output = np.asarray([float(component) for component in output_mpfr], dtype=np.float64)
    return _zero_certificate_result(
        mode=mode,
        decoded_block_starts=decoded_block_starts,
        output_vector_approx=output,
        output_mpfr_approx=output_mpfr,
        iterations=iterations,
        message="All historical blocks decoded; certified skip error is zero.",
        numerical_fallback_used=False,
        precision=precision,
    )
