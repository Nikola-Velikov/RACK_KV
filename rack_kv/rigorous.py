from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Sequence

import gmpy2

from .ieee import exact_mpq


DEFAULT_PRECISION = 256


@dataclass(frozen=True)
class Interval:
    lower: gmpy2.mpfr
    upper: gmpy2.mpfr

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("Interval lower bound exceeds upper bound.")


@contextmanager
def mpfr_context(
    *,
    precision: int = DEFAULT_PRECISION,
    round_mode: int = gmpy2.RoundToNearest,
):
    ctx = gmpy2.get_context().copy()
    ctx.precision = precision
    ctx.round = round_mode
    ctx.real_round = round_mode
    ctx.imag_round = round_mode
    ctx.trap_invalid = True
    ctx.trap_divzero = True
    with gmpy2.context(ctx):
        yield


def _rounded_from_nearest(value: gmpy2.mpfr, *, precision: int, round_mode: int) -> gmpy2.mpfr:
    if round_mode == gmpy2.RoundToNearest:
        return value
    ctx = gmpy2.get_context().copy()
    ctx.precision = precision
    ctx.round = gmpy2.RoundToNearest
    ctx.real_round = gmpy2.RoundToNearest
    ctx.imag_round = gmpy2.RoundToNearest
    with gmpy2.context(ctx):
        if not gmpy2.is_finite(value):
            return value
        if value == 0:
            return gmpy2.mpfr(0)
        if round_mode == gmpy2.RoundUp:
            return gmpy2.next_above(value)
        if round_mode == gmpy2.RoundDown:
            return gmpy2.next_below(value)
    raise ValueError(f"Unsupported rounding mode for rigorous interval wrapper: {round_mode}")


def exact_mpfr(value, *, precision: int = DEFAULT_PRECISION) -> gmpy2.mpfr:
    if isinstance(value, gmpy2.mpfr):
        return value
    if isinstance(value, (float,)) and precision < 53:
        raise ValueError("Exact float64 import requires precision >= 53.")
    if hasattr(value, "dtype") and str(getattr(value, "dtype", "")) == "float64" and precision < 53:
        raise ValueError("Exact float64 import requires precision >= 53.")
    q = exact_mpq(value)
    with mpfr_context(precision=precision, round_mode=gmpy2.RoundToNearest):
        return gmpy2.mpfr(q)


def exact_vector(values: Sequence, *, precision: int = DEFAULT_PRECISION) -> list[gmpy2.mpfr]:
    return [exact_mpfr(value, precision=precision) for value in values]


def rounded_sum(values: Iterable[gmpy2.mpfr], *, precision: int, round_mode: int) -> gmpy2.mpfr:
    value_list = list(values)
    if not value_list:
        return exact_mpfr(0, precision=precision)
    total = exact_mpfr(0, precision=precision)
    for value in value_list:
        total = rounded_add(total, value, precision=precision, round_mode=round_mode)
    return total


def rounded_dot(
    left: Sequence[gmpy2.mpfr],
    right: Sequence[gmpy2.mpfr],
    *,
    precision: int,
    round_mode: int,
) -> gmpy2.mpfr:
    if len(left) != len(right):
        raise ValueError("Dot product dimension mismatch.")
    total = exact_mpfr(0, precision=precision)
    for lhs, rhs in zip(left, right):
        product = rounded_mul(lhs, rhs, precision=precision, round_mode=round_mode)
        total = rounded_add(total, product, precision=precision, round_mode=round_mode)
    return total


def rounded_sqrt(value: gmpy2.mpfr, *, precision: int, round_mode: int) -> gmpy2.mpfr:
    with mpfr_context(precision=precision, round_mode=gmpy2.RoundToNearest):
        nearest = gmpy2.sqrt(value)
    return _rounded_from_nearest(nearest, precision=precision, round_mode=round_mode)


def rounded_exp(value: gmpy2.mpfr, *, precision: int, round_mode: int) -> gmpy2.mpfr:
    with mpfr_context(precision=precision, round_mode=gmpy2.RoundToNearest):
        nearest = gmpy2.exp(value)
    return _rounded_from_nearest(nearest, precision=precision, round_mode=round_mode)


def rounded_sub(left: gmpy2.mpfr, right: gmpy2.mpfr, *, precision: int, round_mode: int) -> gmpy2.mpfr:
    with mpfr_context(precision=precision, round_mode=gmpy2.RoundToNearest):
        nearest = gmpy2.sub(left, right)
    return _rounded_from_nearest(nearest, precision=precision, round_mode=round_mode)


def rounded_div(numerator: gmpy2.mpfr, denominator: gmpy2.mpfr, *, precision: int, round_mode: int) -> gmpy2.mpfr:
    with mpfr_context(precision=precision, round_mode=gmpy2.RoundToNearest):
        nearest = gmpy2.div(numerator, denominator)
    return _rounded_from_nearest(nearest, precision=precision, round_mode=round_mode)


def rounded_mul(left: gmpy2.mpfr, right: gmpy2.mpfr, *, precision: int, round_mode: int) -> gmpy2.mpfr:
    with mpfr_context(precision=precision, round_mode=gmpy2.RoundToNearest):
        nearest = gmpy2.mul(left, right)
    return _rounded_from_nearest(nearest, precision=precision, round_mode=round_mode)


def rounded_add(left: gmpy2.mpfr, right: gmpy2.mpfr, *, precision: int, round_mode: int) -> gmpy2.mpfr:
    with mpfr_context(precision=precision, round_mode=gmpy2.RoundToNearest):
        nearest = gmpy2.add(left, right)
    return _rounded_from_nearest(nearest, precision=precision, round_mode=round_mode)


def norm_upper(vector: Sequence[gmpy2.mpfr], *, precision: int) -> gmpy2.mpfr:
    squares = [rounded_mul(component, component, precision=precision, round_mode=gmpy2.RoundUp) for component in vector]
    squared_sum = rounded_sum(squares, precision=precision, round_mode=gmpy2.RoundUp)
    return rounded_sqrt(squared_sum, precision=precision, round_mode=gmpy2.RoundUp)


def norm_lower(vector: Sequence[gmpy2.mpfr], *, precision: int) -> gmpy2.mpfr:
    squares = [rounded_mul(component, component, precision=precision, round_mode=gmpy2.RoundDown) for component in vector]
    squared_sum = rounded_sum(squares, precision=precision, round_mode=gmpy2.RoundDown)
    return rounded_sqrt(squared_sum, precision=precision, round_mode=gmpy2.RoundDown)


def dot_interval(left: Sequence[gmpy2.mpfr], right: Sequence[gmpy2.mpfr], *, precision: int) -> Interval:
    return Interval(
        lower=rounded_dot(left, right, precision=precision, round_mode=gmpy2.RoundDown),
        upper=rounded_dot(left, right, precision=precision, round_mode=gmpy2.RoundUp),
    )


def min_exact(values: Sequence[gmpy2.mpfr], *, precision: int) -> gmpy2.mpfr:
    if not values:
        raise ValueError("Cannot take min of an empty sequence.")
    with mpfr_context(precision=precision, round_mode=gmpy2.RoundToNearest):
        current = values[0]
        for value in values[1:]:
            current = gmpy2.minnum(current, value)
        return current


def max_exact(values: Sequence[gmpy2.mpfr], *, precision: int) -> gmpy2.mpfr:
    if not values:
        raise ValueError("Cannot take max of an empty sequence.")
    with mpfr_context(precision=precision, round_mode=gmpy2.RoundToNearest):
        current = values[0]
        for value in values[1:]:
            current = gmpy2.maxnum(current, value)
        return current


def upper_sqrt_dimension(head_dim: int, *, precision: int) -> gmpy2.mpfr:
    if head_dim <= 0:
        raise ValueError("Head dimension must be positive.")
    return rounded_sqrt(exact_mpfr(head_dim, precision=precision), precision=precision, round_mode=gmpy2.RoundUp)


def lower_sqrt_dimension(head_dim: int, *, precision: int) -> gmpy2.mpfr:
    if head_dim <= 0:
        raise ValueError("Head dimension must be positive.")
    return rounded_sqrt(exact_mpfr(head_dim, precision=precision), precision=precision, round_mode=gmpy2.RoundDown)


def sqrt_dimension_interval(head_dim: int, *, precision: int) -> Interval:
    return Interval(
        lower=lower_sqrt_dimension(head_dim, precision=precision),
        upper=upper_sqrt_dimension(head_dim, precision=precision),
    )


def multiply_interval_by_nonnegative_scalar(
    interval: Interval,
    scalar: gmpy2.mpfr,
    *,
    precision: int,
) -> Interval:
    if scalar < 0:
        raise ValueError("Scalar must be nonnegative.")
    return Interval(
        lower=rounded_mul(interval.lower, scalar, precision=precision, round_mode=gmpy2.RoundDown),
        upper=rounded_mul(interval.upper, scalar, precision=precision, round_mode=gmpy2.RoundUp),
    )


def add_intervals(left: Interval, right: Interval, *, precision: int) -> Interval:
    return Interval(
        lower=rounded_add(left.lower, right.lower, precision=precision, round_mode=gmpy2.RoundDown),
        upper=rounded_add(left.upper, right.upper, precision=precision, round_mode=gmpy2.RoundUp),
    )


def subtract_point_from_interval(interval: Interval, point: gmpy2.mpfr, *, precision: int) -> Interval:
    return Interval(
        lower=rounded_sub(interval.lower, point, precision=precision, round_mode=gmpy2.RoundDown),
        upper=rounded_sub(interval.upper, point, precision=precision, round_mode=gmpy2.RoundUp),
    )


def divide_interval_by_positive_interval(
    numerator: Interval,
    denominator: Interval,
    *,
    precision: int,
) -> Interval:
    if denominator.lower <= 0:
        raise ValueError("Positive denominator interval required for rigorous division.")
    lower_candidates = [
        rounded_div(numerator.lower, denominator.lower, precision=precision, round_mode=gmpy2.RoundDown),
        rounded_div(numerator.lower, denominator.upper, precision=precision, round_mode=gmpy2.RoundDown),
        rounded_div(numerator.upper, denominator.lower, precision=precision, round_mode=gmpy2.RoundDown),
        rounded_div(numerator.upper, denominator.upper, precision=precision, round_mode=gmpy2.RoundDown),
    ]
    upper_candidates = [
        rounded_div(numerator.lower, denominator.lower, precision=precision, round_mode=gmpy2.RoundUp),
        rounded_div(numerator.lower, denominator.upper, precision=precision, round_mode=gmpy2.RoundUp),
        rounded_div(numerator.upper, denominator.lower, precision=precision, round_mode=gmpy2.RoundUp),
        rounded_div(numerator.upper, denominator.upper, precision=precision, round_mode=gmpy2.RoundUp),
    ]
    return Interval(
        lower=min_exact(lower_candidates, precision=precision),
        upper=max_exact(upper_candidates, precision=precision),
    )
