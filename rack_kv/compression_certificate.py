"""Rigorous local compression-error bounds for reconstructed KV attention."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
from functools import cmp_to_key

import gmpy2

from .rigorous import (
    Interval, add_intervals, exact_mpfr, exact_vector, dot_interval,
    max_exact, min_exact, norm_upper, rounded_add, rounded_div, rounded_exp,
    rounded_mul, rounded_sub, mpfr_context,
)
from .ieee import exact_mpq


@dataclass(frozen=True)
class CompressionErrorMetadata:
    kappa: tuple[gmpy2.mpfr, ...]
    eta: tuple[gmpy2.mpfr, ...]
    block_kappa: gmpy2.mpfr
    block_eta: gmpy2.mpfr
    metadata_bytes: int


@dataclass(frozen=True)
class CompressionCertificate:
    e_value: gmpy2.mpfr
    e_value_weighted: gmpy2.mpfr
    e_probability: gmpy2.mpfr
    e_probability_fallback: gmpy2.mpfr
    e_comp: gmpy2.mpfr
    lower_probabilities: tuple[gmpy2.mpfr, ...]
    upper_probabilities: tuple[gmpy2.mpfr, ...]
    probability_distances: tuple[gmpy2.mpfr, ...]
    delta: tuple[gmpy2.mpfr, ...]


@dataclass(frozen=True)
class TightCompressionCertificate:
    """Step-11 certificate and its auditable ablation components."""

    e_value_step10: gmpy2.mpfr
    e_value_simplex: gmpy2.mpfr
    e_value_tight: gmpy2.mpfr
    e_probability_step10: gmpy2.mpfr
    e_probability_coordinate: gmpy2.mpfr
    e_probability_mass_zero: gmpy2.mpfr
    e_probability_zero_center: gmpy2.mpfr
    e_probability_mean_center: gmpy2.mpfr
    e_probability_ohat_center: gmpy2.mpfr
    e_probability_tight: gmpy2.mpfr
    e_comp_step10: gmpy2.mpfr
    e_comp_coordinate: gmpy2.mpfr
    e_comp_mass: gmpy2.mpfr
    e_comp_centered: gmpy2.mpfr
    e_comp_tight: gmpy2.mpfr
    selected_center: str
    lower_probabilities: tuple[gmpy2.mpfr, ...]
    upper_probabilities: tuple[gmpy2.mpfr, ...]
    step10_lower_probabilities: tuple[gmpy2.mpfr, ...]
    step10_upper_probabilities: tuple[gmpy2.mpfr, ...]
    probability_distances: tuple[gmpy2.mpfr, ...]
    tight_distances: tuple[gmpy2.mpfr, ...]
    delta: tuple[gmpy2.mpfr, ...]
    p_hat_lower: tuple[gmpy2.mpfr, ...]
    p_hat_upper: tuple[gmpy2.mpfr, ...]
    mass_t: gmpy2.mpfr
    center_weights: tuple[tuple[gmpy2.mpfr, ...], ...]


def _round_sum(values, *, precision: int, mode):
    ctx = gmpy2.get_context().copy()
    ctx.precision = precision
    ctx.round = mode
    ctx.real_round = mode
    with gmpy2.context(ctx):
        total = gmpy2.mpfr(0)
        for value in values:
            total += value
        return total


def _round_norm(vector, *, precision: int, mode):
    ctx = gmpy2.get_context().copy()
    ctx.precision = precision
    ctx.round = mode
    ctx.real_round = mode
    with gmpy2.context(ctx):
        total = gmpy2.mpfr(0)
        for value in vector:
            total += value * value
        return gmpy2.sqrt(total)


def _tight_probability_extrema(lower_logits, upper_logits, *, precision: int):
    """Coordinate-wise softmax extrema using the exact independent-box formulas."""
    down = gmpy2.RoundDown
    up = gmpy2.RoundUp
    shift = max(upper_logits)
    ctxd = gmpy2.get_context().copy(); ctxd.precision = precision; ctxd.round = down; ctxd.real_round = down
    ctxu = gmpy2.get_context().copy(); ctxu.precision = precision; ctxu.round = up; ctxu.real_round = up
    with gmpy2.context(ctxd):
        low_exp = [gmpy2.exp(x - shift) for x in lower_logits]
    with gmpy2.context(ctxu):
        high_exp = [gmpy2.exp(x - shift) for x in upper_logits]
    sum_low = _round_sum(low_exp, precision=precision, mode=down)
    sum_high = _round_sum(high_exp, precision=precision, mode=up)
    lower = []
    upper = []
    for i in range(len(lower_logits)):
        # Lower: numerator exp(L_i), all other coordinates at U_j.
        den_hi = rounded_add(
            rounded_sub(sum_high, high_exp[i], precision=precision, round_mode=up),
            high_exp[i] if False else low_exp[i],
            precision=precision,
            round_mode=up,
        )
        # Upper: numerator exp(U_i), all other coordinates at L_j.
        den_lo = rounded_add(
            rounded_sub(sum_low, low_exp[i], precision=precision, round_mode=down),
            low_exp[i] if False else high_exp[i],
            precision=precision,
            round_mode=down,
        )
        lower.append(rounded_div(low_exp[i], den_hi, precision=precision, round_mode=down))
        upper.append(rounded_div(high_exp[i], den_lo, precision=precision, round_mode=up))
    return tuple(lower), tuple(upper), tuple(low_exp), tuple(high_exp), shift


def _probability_box_from_logits(lower_logits, upper_logits, *, precision: int):
    """Step-10 globally independent numerator/denominator intervals."""
    down = gmpy2.RoundDown
    up = gmpy2.RoundUp
    shift = max(upper_logits)
    ctxd = gmpy2.get_context().copy(); ctxd.precision = precision; ctxd.round = down; ctxd.real_round = down
    ctxu = gmpy2.get_context().copy(); ctxu.precision = precision; ctxu.round = up; ctxu.real_round = up
    with gmpy2.context(ctxd): low_exp = [gmpy2.exp(x - shift) for x in lower_logits]
    with gmpy2.context(ctxu): high_exp = [gmpy2.exp(x - shift) for x in upper_logits]
    zlo = _round_sum(low_exp, precision=precision, mode=down)
    zhi = _round_sum(high_exp, precision=precision, mode=up)
    lower = tuple(rounded_div(x, zhi, precision=precision, round_mode=down) for x in low_exp)
    upper = tuple(rounded_div(x, zlo, precision=precision, round_mode=up) for x in high_exp)
    return lower, upper, shift


def _safe_upper_difference(left, right, *, precision: int):
    value = rounded_sub(left, right, precision=precision, round_mode=gmpy2.RoundUp)
    return max(gmpy2.mpfr(0), value)


def _fractional_knapsack(capacities, weights, capacity, *, precision: int):
    """Safe upper bound using MPFR weight ordering and outward accumulation."""
    def compare(a, b):
        if a[1] > b[1]: return -1
        if a[1] < b[1]: return 1
        return a[0] - b[0]
    order = sorted(enumerate(weights), key=cmp_to_key(compare))
    remaining = capacity
    result = gmpy2.mpfr(0)
    for index, weight in order:
        if remaining <= 0:
            break
        amount = min(capacities[index], remaining)
        result = rounded_add(result, rounded_mul(amount, weight, precision=precision, round_mode=gmpy2.RoundUp), precision=precision, round_mode=gmpy2.RoundUp)
        remaining = rounded_sub(remaining, amount, precision=precision, round_mode=gmpy2.RoundDown)
    return result


def _simplex_value_bound(lower, upper, eta, *, precision: int):
    down = gmpy2.RoundDown
    up = gmpy2.RoundUp
    remaining = rounded_sub(gmpy2.mpfr(1), _round_sum(lower, precision=precision, mode=down), precision=precision, round_mode=up)
    capacities = [max(gmpy2.mpfr(0), rounded_sub(hi, lo, precision=precision, round_mode=up)) for lo, hi in zip(lower, upper)]
    order = sorted(enumerate(eta), key=cmp_to_key(lambda a, b: -1 if a[1] > b[1] else (1 if a[1] < b[1] else a[0] - b[0])))
    result = gmpy2.mpfr(0)
    for i, weight in order:
        result = rounded_add(result, rounded_mul(lower[i], weight, precision=precision, round_mode=up), precision=precision, round_mode=up)
        if remaining <= 0:
            continue
        amount = min(capacities[i], remaining)
        result = rounded_add(result, rounded_mul(amount, weight, precision=precision, round_mode=up), precision=precision, round_mode=up)
        remaining = rounded_sub(remaining, amount, precision=precision, round_mode=down)
    return result


def certify_compression_tight(
    query: Sequence,
    reconstructed_keys,
    reconstructed_values,
    metadata: CompressionErrorMetadata,
    *,
    precision: int = 256,
    attention_scale: float,
    centers=None,
) -> TightCompressionCertificate:
    """Step-11 tight probability and value certificate.

    All candidate bounds are outward-rounded. The original Step-10
    certificate remains implemented separately for direct ablation.
    """
    n = len(reconstructed_keys)
    if n != len(reconstructed_values) or n != len(metadata.kappa):
        raise ValueError("reconstructed tensors and error metadata must have equal token counts")
    q = [exact_mpfr(x, precision=precision) for x in query]
    keys = [[exact_mpfr(x, precision=precision) for x in key] for key in reconstructed_keys]
    values = [[exact_mpfr(x, precision=precision) for x in value] for value in reconstructed_values]
    q_norm = _round_norm(q, precision=precision, mode=gmpy2.RoundUp)
    scale = exact_mpfr(attention_scale, precision=precision)
    down = gmpy2.RoundDown
    up = gmpy2.RoundUp
    logits_down = []
    logits_up = []
    for key in keys:
        logits_down.append(rounded_mul(_round_sum([rounded_mul(a, b, precision=precision, round_mode=down) for a, b in zip(q, key)], precision=precision, mode=down), scale, precision=precision, round_mode=down))
        logits_up.append(rounded_mul(_round_sum([rounded_mul(a, b, precision=precision, round_mode=up) for a, b in zip(q, key)], precision=precision, mode=up), scale, precision=precision, round_mode=up))
    deltas = tuple(rounded_mul(rounded_mul(q_norm, error, precision=precision, round_mode=up), scale, precision=precision, round_mode=up) for error in metadata.kappa)
    lower_logits = tuple(rounded_sub(x, d, precision=precision, round_mode=down) for x, d in zip(logits_down, deltas))
    upper_logits = tuple(rounded_add(x, d, precision=precision, round_mode=up) for x, d in zip(logits_up, deltas))
    old_lower, old_upper, _ = _probability_box_from_logits(lower_logits, upper_logits, precision=precision)
    tight_lower, tight_upper, _, _, _ = _tight_probability_extrema(lower_logits, upper_logits, precision=precision)

    # Reconstructed softmax enclosure used to make deviations safe even
    # though p_hat itself is evaluated at finite MPFR precision.
    phat_lower, phat_upper, _ = _probability_box_from_logits(logits_down, logits_up, precision=precision)
    old_dist = tuple(max(_safe_upper_difference(phat_upper[i], old_lower[i], precision=precision), _safe_upper_difference(old_upper[i], phat_lower[i], precision=precision)) for i in range(n))
    tight_dist = tuple(max(_safe_upper_difference(phat_upper[i], tight_lower[i], precision=precision), _safe_upper_difference(tight_upper[i], phat_lower[i], precision=precision)) for i in range(n))

    value_norms = tuple(_round_norm(value, precision=precision, mode=up) for value in values)
    e_value_step10 = min(max(metadata.eta), _round_sum([rounded_mul(upper, error, precision=precision, round_mode=up) for upper, error in zip(old_upper, metadata.eta)], precision=precision, mode=up))
    e_value_simplex = _simplex_value_bound(tight_lower, tight_upper, metadata.eta, precision=precision)
    e_value_tight = min(max(metadata.eta), e_value_simplex)
    e_prob_step10 = min(_round_sum([rounded_mul(d, w, precision=precision, round_mode=up) for d, w in zip(old_dist, value_norms)], precision=precision, mode=up), rounded_mul(gmpy2.mpfr(2), max(value_norms), precision=precision, round_mode=up))
    e_prob_coordinate = min(_round_sum([rounded_mul(d, w, precision=precision, round_mode=up) for d, w in zip(tight_dist, value_norms)], precision=precision, mode=up), rounded_mul(gmpy2.mpfr(2), max(value_norms), precision=precision, round_mode=up))

    a = tuple(_safe_upper_difference(tight_upper[i], phat_lower[i], precision=precision) for i in range(n))
    b = tuple(_safe_upper_difference(phat_upper[i], tight_lower[i], precision=precision) for i in range(n))
    mass_t = min(_round_sum(a, precision=precision, mode=up), _round_sum(b, precision=precision, mode=up))
    e_prob_zero_transport = _fractional_knapsack(a, value_norms, mass_t, precision=precision) + _fractional_knapsack(b, value_norms, mass_t, precision=precision)
    e_prob_zero_coordinate = _round_sum([rounded_mul(d, w, precision=precision, round_mode=up) for d, w in zip(tight_dist, value_norms)], precision=precision, mode=up)
    e_prob_zero = min(e_prob_zero_transport, e_prob_zero_coordinate)

    if centers is None:
        centers = {"zero": [0.0] * len(values[0])}
    center_names = list(centers.keys())
    center_weights = []
    center_bounds = []
    for name in center_names:
        center = [exact_mpfr(x, precision=precision) for x in centers[name]]
        weights = tuple(_round_norm([value[j] - center[j] for j in range(len(center))], precision=precision, mode=up) for value in values)
        center_weights.append(weights)
        transport = _fractional_knapsack(a, weights, mass_t, precision=precision) + _fractional_knapsack(b, weights, mass_t, precision=precision)
        coordinate = _round_sum([rounded_mul(d, w, precision=precision, round_mode=up) for d, w in zip(tight_dist, weights)], precision=precision, mode=up)
        center_bounds.append(min(transport, coordinate))
    # Require the three named centers for a stable output schema.
    center_map = dict(zip(center_names, center_bounds))
    for name in ("zero", "mean", "ohat"):
        center_map.setdefault(name, e_prob_zero)
    selected_center = min(("zero", "mean", "ohat"), key=lambda name: center_map[name])
    e_prob_tight = min(e_prob_coordinate, center_map[selected_center])
    e_comp_step10 = e_value_step10 + e_prob_step10
    e_comp_coordinate = e_value_step10 + e_prob_coordinate
    e_comp_mass = e_value_step10 + e_prob_zero
    e_comp_centered = e_value_step10 + min(center_map.values())
    e_comp_tight = e_value_tight + e_prob_tight
    return TightCompressionCertificate(
        e_value_step10, e_value_simplex, e_value_tight,
        e_prob_step10, e_prob_coordinate, e_prob_zero,
        center_map["zero"], center_map["mean"], center_map["ohat"], e_prob_tight,
        e_comp_step10, e_comp_coordinate, e_comp_mass, e_comp_centered, e_comp_tight,
        selected_center, tight_lower, tight_upper, old_lower, old_upper,
        tight_dist, tuple(tight_dist), deltas, phat_lower, phat_upper, mass_t,
        tuple(center_weights),
    )


def _norm_difference(left, right, *, precision: int):
    # Keep the same exact-float import and upward norm semantics as the shared
    # rigorous helpers, but avoid rebuilding an MPFR context per coordinate.
    ctx = gmpy2.get_context().copy()
    ctx.precision = precision
    ctx.round = gmpy2.RoundUp
    ctx.real_round = gmpy2.RoundUp
    with gmpy2.context(ctx):
        total = gmpy2.mpfr(0)
        for a, b in zip(left, right):
            da = gmpy2.mpfr(exact_mpq(a))
            db = gmpy2.mpfr(exact_mpq(b))
            diff = da - db
            total += diff * diff
        return gmpy2.sqrt(total)


def build_error_metadata(original_keys, reconstructed_keys, original_values, reconstructed_values, *, precision: int = 256, block_ranges: Sequence[tuple[int, int]] = ()) -> CompressionErrorMetadata:
    kappa = tuple(_norm_difference(a, b, precision=precision) for a, b in zip(original_keys, reconstructed_keys))
    eta = tuple(_norm_difference(a, b, precision=precision) for a, b in zip(original_values, reconstructed_values))
    ranges = tuple(block_ranges)
    block_kappa = max_exact([max_exact(kappa[start:end], precision=precision) for start, end in ranges], precision=precision) if ranges else max_exact(kappa, precision=precision)
    block_eta = max_exact([max_exact(eta[start:end], precision=precision) for start, end in ranges], precision=precision) if ranges else max_exact(eta, precision=precision)
    return CompressionErrorMetadata(kappa, eta, block_kappa, block_eta, len(ranges) * 8)


def _exp_shifted(interval: Interval, shift: gmpy2.mpfr, *, precision: int):
    return Interval(
        rounded_exp(rounded_sub(interval.lower, shift, precision=precision, round_mode=gmpy2.RoundDown), precision=precision, round_mode=gmpy2.RoundDown),
        rounded_exp(rounded_sub(interval.upper, shift, precision=precision, round_mode=gmpy2.RoundUp), precision=precision, round_mode=gmpy2.RoundUp),
    )


def certify_compression(query: Sequence, reconstructed_keys, reconstructed_values, metadata: CompressionErrorMetadata, *, precision: int = 256, attention_scale: float) -> CompressionCertificate:
    n = len(reconstructed_keys)
    if n != len(reconstructed_values) or n != len(metadata.kappa):
        raise ValueError("reconstructed tensors and error metadata must have equal token counts")
    ctx = gmpy2.get_context().copy(); ctx.precision = precision; ctx.trap_invalid = True; ctx.trap_divzero = True
    with gmpy2.context(ctx):
        q = [gmpy2.mpfr(exact_mpq(x)) for x in query]
        keys = [[gmpy2.mpfr(exact_mpq(x)) for x in key] for key in reconstructed_keys]
        values = [[gmpy2.mpfr(exact_mpq(x)) for x in value] for value in reconstructed_values]
    def norm_up(vector):
        local = gmpy2.get_context().copy(); local.precision = precision; local.round = gmpy2.RoundUp; local.real_round = gmpy2.RoundUp
        with gmpy2.context(local): return gmpy2.sqrt(sum(x * x for x in vector))
    q_norm = norm_up(q)
    scale = exact_mpfr(attention_scale, precision=precision)
    logits = []
    for mode in (gmpy2.RoundDown, gmpy2.RoundUp):
        local = gmpy2.get_context().copy(); local.precision = precision; local.round = mode; local.real_round = mode
        with gmpy2.context(local):
            logits.append([sum(a * b for a, b in zip(q, key)) * scale for key in keys])
    lower_raw, upper_raw = logits
    local = gmpy2.get_context().copy(); local.precision = precision; local.round = gmpy2.RoundUp; local.real_round = gmpy2.RoundUp
    with gmpy2.context(local):
        deltas = [q_norm * error * scale for error in metadata.kappa]
        lower_logits = [x - d for x, d in zip(lower_raw, deltas)]
        upper_logits = [x + d for x, d in zip(upper_raw, deltas)]
        shift = max(upper_logits)
    local_down = gmpy2.get_context().copy(); local_down.precision = precision; local_down.round = gmpy2.RoundDown; local_down.real_round = gmpy2.RoundDown
    local_up = gmpy2.get_context().copy(); local_up.precision = precision; local_up.round = gmpy2.RoundUp; local_up.real_round = gmpy2.RoundUp
    with gmpy2.context(local_down): lower_exp = [gmpy2.exp(x - shift) for x in lower_logits]
    with gmpy2.context(local_up): upper_exp = [gmpy2.exp(x - shift) for x in upper_logits]
    with gmpy2.context(local_down): z_minus = sum(lower_exp)
    with gmpy2.context(local_up): z_plus = sum(upper_exp)
    with gmpy2.context(local_down): lower_p = tuple(x / z_plus for x in lower_exp)
    with gmpy2.context(local_up): upper_p = tuple(x / z_minus for x in upper_exp)
    with gmpy2.context(local_down): e_lo = [gmpy2.exp(x - shift) for x in lower_raw]
    with gmpy2.context(local_up): e_hi = [gmpy2.exp(x - shift) for x in upper_raw]
    with gmpy2.context(local_down): zhat_lo = sum(e_lo)
    with gmpy2.context(local_up): zhat_hi = sum(e_hi)
    with gmpy2.context(local_down): phat_lo = [x / zhat_hi for x in e_lo]
    with gmpy2.context(local_up): phat_hi = [x / zhat_lo for x in e_hi]
    with gmpy2.context(local_up): distances = tuple(max(phat_hi[i] - lower_p[i], upper_p[i] - phat_lo[i]) for i in range(n))
    value_norms = [norm_up(value) for value in values]
    with gmpy2.context(local_up):
        e_value_weighted = sum(upper * error for upper, error in zip(upper_p, metadata.eta))
        e_value = min(max(metadata.eta), e_value_weighted)
        e_probability_sum = sum(distance * norm for distance, norm in zip(distances, value_norms))
        e_probability_fallback = gmpy2.mpfr(2) * max(value_norms)
        e_probability = min(e_probability_sum, e_probability_fallback)
        e_comp = e_value + e_probability
    return CompressionCertificate(e_value, e_value_weighted, e_probability, e_probability_fallback, e_comp, lower_p, upper_p, distances, tuple(deltas))
