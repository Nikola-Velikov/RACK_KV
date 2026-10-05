from __future__ import annotations

import math
import struct
from typing import Any

import gmpy2
import numpy as np


def _bits_from_float16(value: np.float16) -> int:
    return int(np.asarray(value, dtype=np.float16).view(np.uint16).item())


def _bits_from_float32(value: np.float32) -> int:
    return int(np.asarray(value, dtype=np.float32).view(np.uint32).item())


def _bits_from_float64(value: np.float64 | float) -> int:
    return struct.unpack(">Q", struct.pack(">d", float(value)))[0]


def _ieee_to_mpq(bits: int, exponent_bits: int, fraction_bits: int, bias: int) -> gmpy2.mpq:
    sign = -1 if (bits >> (exponent_bits + fraction_bits)) & 1 else 1
    exponent_mask = (1 << exponent_bits) - 1
    fraction_mask = (1 << fraction_bits) - 1
    exponent = (bits >> fraction_bits) & exponent_mask
    fraction = bits & fraction_mask

    if exponent == exponent_mask:
        raise ValueError("NaN and infinity are not supported as exact certification inputs.")

    if exponent == 0:
        if fraction == 0:
            return gmpy2.mpq(0)
        significand = fraction
        two_power = 1 - bias - fraction_bits
    else:
        significand = (1 << fraction_bits) | fraction
        two_power = exponent - bias - fraction_bits

    value = gmpy2.mpq(significand)
    if two_power >= 0:
        value *= 1 << two_power
    else:
        value /= 1 << (-two_power)
    return value if sign > 0 else -value


def exact_mpq(value: Any) -> gmpy2.mpq:
    if isinstance(value, gmpy2.mpq):
        return value
    if isinstance(value, gmpy2.mpz):
        return gmpy2.mpq(value)
    if isinstance(value, (int, np.integer)):
        return gmpy2.mpq(int(value))
    if isinstance(value, np.float16):
        return _ieee_to_mpq(_bits_from_float16(value), 5, 10, 15)
    if isinstance(value, np.float32):
        return _ieee_to_mpq(_bits_from_float32(value), 8, 23, 127)
    if isinstance(value, (float, np.float64)):
        return _ieee_to_mpq(_bits_from_float64(value), 11, 52, 1023)
    raise TypeError(f"Unsupported exact import type: {type(value)!r}")


def exact_float(value: Any) -> float:
    q = exact_mpq(value)
    numerator = int(q.numerator)
    denominator = int(q.denominator)
    return numerator / denominator


def outward_round_float32_upper(value: float) -> np.float32:
    if math.isnan(value) or math.isinf(value):
        raise ValueError("Cannot store NaN or infinity as upward-rounded metadata.")
    if value == 0.0:
        return np.float32(0.0)
    cast = np.float32(value)
    if float(cast) < value:
        return np.nextafter(cast, np.float32(np.inf), dtype=np.float32)
    if float(cast) == value:
        return np.nextafter(cast, np.float32(np.inf), dtype=np.float32)
    return cast


def mpfr_to_proven_float32_upper(value: gmpy2.mpfr) -> np.float32:
    if not gmpy2.is_finite(value):
        raise ValueError("Cannot store a non-finite MPFR bound as float32 metadata.")
    if value < 0:
        raise ValueError("Cannot store a negative upper bound as float32 metadata.")
    if value == 0:
        return np.float32(0.0)

    candidate = np.float32(float(value))
    if not np.isfinite(candidate):
        raise OverflowError("MPFR bound is too large to store as float32 metadata.")

    compare_precision = max(128, int(getattr(value, "precision", 53)) + 32)
    while True:
        ctx = gmpy2.get_context().copy()
        ctx.precision = compare_precision
        ctx.round = gmpy2.RoundToNearest
        ctx.real_round = gmpy2.RoundToNearest
        ctx.imag_round = gmpy2.RoundToNearest
        with gmpy2.context(ctx):
            candidate_mpfr = gmpy2.mpfr(exact_mpq(candidate))
        if candidate_mpfr >= value:
            return candidate
        candidate = np.nextafter(candidate, np.float32(np.inf), dtype=np.float32)
        if not np.isfinite(candidate):
            raise OverflowError("Failed to find a finite float32 upper bound.")
