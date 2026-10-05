from __future__ import annotations

import unittest
from unittest.mock import patch

import gmpy2
import numpy as np

from rack_kv.certificate import (
    _block_logit_cap_interval,
    certify_progressive_skipping,
    exact_reference_output,
    exact_reference_output_mpfr,
    rigorous_attention_output_interval,
    rigorous_output_error_norm,
    rigorous_output_error_upper_from_intervals,
)
from rack_kv.codec import CompressedBlock, encode_block
from rack_kv.ieee import exact_mpq
from rack_kv.rigorous import (
    DEFAULT_PRECISION,
    dot_interval,
    exact_mpfr,
    exact_vector,
    norm_lower,
    lower_sqrt_dimension,
    rounded_add,
    rounded_div,
    rounded_dot,
    rounded_exp,
    rounded_mul,
    rounded_sqrt,
    rounded_sub,
    upper_sqrt_dimension,
)
from rack_kv.types import CertificateMode, DecodeSchedule


def _full_reconstructed_tensors(
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
    blocks: list[CompressedBlock],
) -> tuple[np.ndarray, np.ndarray]:
    all_keys = [np.asarray(recent_keys, dtype=np.float64)]
    all_values = [np.asarray(recent_values, dtype=np.float64)]
    for block in blocks:
        keys, values = block.decode_block()
        all_keys.append(keys)
        all_values.append(values)
    return np.vstack(all_keys), np.vstack(all_values)


def _kept_reconstructed_tensors(
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
    blocks: list[CompressedBlock],
    decoded_block_starts: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    kept_keys = [np.asarray(recent_keys, dtype=np.float64)]
    kept_values = [np.asarray(recent_values, dtype=np.float64)]
    for block in blocks:
        if block.header.block_start in decoded_block_starts:
            keys, values = block.decode_block()
            kept_keys.append(keys)
            kept_values.append(values)
    return np.vstack(kept_keys), np.vstack(kept_values)


def _make_residual_block(
    *,
    block_start: int,
    dimension: int,
    block_len: int,
    anchor0: float,
    residual_scale: float,
    value_base: float,
    value_residual_scale: float,
) -> CompressedBlock:
    keys = []
    values = []
    for index in range(block_len):
        key = np.zeros(dimension, dtype=np.float64)
        key[0] = anchor0 + residual_scale * index
        if dimension > 1:
            key[1] = ((-1) ** index) * residual_scale * 0.5
        if dimension > 2:
            key[2] = residual_scale * (index - block_len / 2) * 0.25
        keys.append(key)
        values.append(
            np.array(
                [
                    value_base + index * value_residual_scale,
                    ((-1) ** index) * (value_base * 0.25 + value_residual_scale),
                ],
                dtype=np.float64,
            )
        )
    return encode_block(np.asarray(keys, dtype=np.float64), np.asarray(values, dtype=np.float64), block_start=block_start)


def _assert_rigorous_certificate_valid(
    test_case: unittest.TestCase,
    *,
    query: np.ndarray,
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
    blocks: list[CompressedBlock],
    result,
    precision: int = 512,
) -> None:
    full_keys, full_values = _full_reconstructed_tensors(recent_keys, recent_values, blocks)
    kept_keys, kept_values = _kept_reconstructed_tensors(recent_keys, recent_values, blocks, result.decoded_block_starts)
    full_interval, _ = rigorous_attention_output_interval(query, full_keys, full_values, precision=precision)
    kept_interval, _ = rigorous_attention_output_interval(query, kept_keys, kept_values, precision=precision)
    rigorous_error_upper = rigorous_output_error_upper_from_intervals(full_interval, kept_interval, precision=precision)
    test_case.assertLessEqual(rigorous_error_upper, result.certificate_value_mpfr)


class RigorousCertificateTests(unittest.TestCase):
    def test_exact_import_preserves_float32_value(self) -> None:
        value = np.float32(0.15625)
        q = exact_mpq(value)
        self.assertEqual(int(q.numerator), 5)
        self.assertEqual(int(q.denominator), 32)

    def test_sqrt_bounds_are_distinct_at_multiple_precisions(self) -> None:
        for precision in (53, 128, 256):
            lower = lower_sqrt_dimension(2, precision=precision)
            upper = upper_sqrt_dimension(2, precision=precision)
            self.assertLess(lower, upper)

    def test_exact_zero_behaviour_for_sqrt_and_norm_lower(self) -> None:
        for precision in (53, 128, 256):
            zero = exact_mpfr(0, precision=precision)
            self.assertEqual(rounded_sqrt(zero, precision=precision, round_mode=gmpy2.RoundDown), 0)
            self.assertEqual(rounded_sqrt(zero, precision=precision, round_mode=gmpy2.RoundUp), 0)
            vector = exact_vector(np.zeros(8, dtype=np.float64), precision=precision)
            self.assertEqual(norm_lower(vector, precision=precision), 0)

    def test_directed_operations_bracket_high_precision_references(self) -> None:
        work_precision = 53
        reference_precision = 4096

        x_work = exact_mpfr(np.float64(0.1), precision=work_precision)
        y_work = exact_mpfr(np.float64(0.3), precision=work_precision)
        x_ref = exact_mpfr(np.float64(0.1), precision=reference_precision)
        y_ref = exact_mpfr(np.float64(0.3), precision=reference_precision)

        add_low = rounded_add(x_work, y_work, precision=work_precision, round_mode=gmpy2.RoundDown)
        add_up = rounded_add(x_work, y_work, precision=work_precision, round_mode=gmpy2.RoundUp)
        add_ref = rounded_add(x_ref, y_ref, precision=reference_precision, round_mode=gmpy2.RoundToNearest)
        self.assertLessEqual(add_low, add_ref)
        self.assertGreaterEqual(add_up, add_ref)

        sub_low = rounded_sub(y_work, x_work, precision=work_precision, round_mode=gmpy2.RoundDown)
        sub_up = rounded_sub(y_work, x_work, precision=work_precision, round_mode=gmpy2.RoundUp)
        sub_ref = rounded_sub(y_ref, x_ref, precision=reference_precision, round_mode=gmpy2.RoundToNearest)
        self.assertLessEqual(sub_low, sub_ref)
        self.assertGreaterEqual(sub_up, sub_ref)

        mul_low = rounded_mul(x_work, y_work, precision=work_precision, round_mode=gmpy2.RoundDown)
        mul_up = rounded_mul(x_work, y_work, precision=work_precision, round_mode=gmpy2.RoundUp)
        mul_ref = rounded_mul(x_ref, y_ref, precision=reference_precision, round_mode=gmpy2.RoundToNearest)
        self.assertLessEqual(mul_low, mul_ref)
        self.assertGreaterEqual(mul_up, mul_ref)

        div_low = rounded_div(y_work, x_work, precision=work_precision, round_mode=gmpy2.RoundDown)
        div_up = rounded_div(y_work, x_work, precision=work_precision, round_mode=gmpy2.RoundUp)
        div_ref = rounded_div(y_ref, x_ref, precision=reference_precision, round_mode=gmpy2.RoundToNearest)
        self.assertLessEqual(div_low, div_ref)
        self.assertGreaterEqual(div_up, div_ref)

        exp_low = rounded_exp(x_work, precision=work_precision, round_mode=gmpy2.RoundDown)
        exp_up = rounded_exp(x_work, precision=work_precision, round_mode=gmpy2.RoundUp)
        exp_ref = rounded_exp(x_ref, precision=reference_precision, round_mode=gmpy2.RoundToNearest)
        self.assertLessEqual(exp_low, exp_ref)
        self.assertGreaterEqual(exp_up, exp_ref)

    def test_dot_interval_contains_reference(self) -> None:
        left = exact_vector([np.float32(0.25), np.float32(-0.5), np.float32(1.5)], precision=DEFAULT_PRECISION)
        right = exact_vector([np.float32(2.0), np.float32(-1.0), np.float32(0.25)], precision=DEFAULT_PRECISION)
        interval = dot_interval(left, right, precision=DEFAULT_PRECISION)
        reference = exact_mpfr(np.float32(1.375), precision=DEFAULT_PRECISION)
        self.assertLessEqual(interval.lower, reference)
        self.assertGreaterEqual(interval.upper, reference)

    def test_directed_operations_keep_requested_precision(self) -> None:
        precision = 211
        left = exact_mpfr(np.float64(1.0), precision=precision)
        right = exact_mpfr(np.float64(2.0**-40), precision=precision)
        result = rounded_sub(left, right, precision=precision, round_mode=gmpy2.RoundUp)
        self.assertEqual(result.precision, precision)
        self.assertEqual(gmpy2.get_context().precision, 53)

    def test_precision_below_53_is_rejected_for_exact_float64_import(self) -> None:
        with self.assertRaises(ValueError):
            exact_mpfr(np.float64(1.25), precision=52)

    def test_old_direct_omitted_contribution_bound_fails(self) -> None:
        query = np.array([0.0], dtype=np.float64)
        kept_keys = np.array([[0.0]], dtype=np.float64)
        kept_values = np.array([[1.0]], dtype=np.float64)
        skipped_keys = np.array([[0.0]], dtype=np.float64)
        skipped_values = np.array([[-1.0]], dtype=np.float64)
        full_output = exact_reference_output(
            query,
            np.vstack([kept_keys, skipped_keys]),
            np.vstack([kept_values, skipped_values]),
        )
        kept_output = exact_reference_output(query, kept_keys, kept_values)
        true_error = float(np.linalg.norm(full_output - kept_output))
        direct_omitted_contribution = 0.5
        self.assertGreater(true_error, direct_omitted_contribution)
        self.assertAlmostEqual(true_error, 1.0, places=12)

    def test_adversarial_um_regression_uses_sign_safe_sqrt_division(self) -> None:
        dimension = 128
        query = np.zeros(dimension, dtype=np.float64)
        query[0] = 1.0
        anchor_value = np.float16(5.424022674560547e-06)
        keys = np.zeros((1, dimension), dtype=np.float64)
        keys[0, 0] = float(anchor_value)
        values = np.zeros((1, 1), dtype=np.float64)
        block = encode_block(keys, values, block_start=0)

        query_exact = exact_vector(query, precision=256)
        beta_upper = _block_logit_cap_interval(query_exact, block, precision=256).upper
        u_m = rounded_exp(beta_upper, precision=256, round_mode=gmpy2.RoundUp)

        reference_precision = 4096
        query_high = exact_vector(query, precision=reference_precision)
        anchor_high = exact_vector(block.header.anchor_key, precision=reference_precision)
        sqrt_high = upper_sqrt_dimension(dimension, precision=reference_precision)
        dot_high = rounded_dot(query_high, anchor_high, precision=reference_precision, round_mode=gmpy2.RoundUp)
        true_logit_upper = rounded_div(dot_high, sqrt_high, precision=reference_precision, round_mode=gmpy2.RoundUp)
        true_mass_upper = rounded_exp(true_logit_upper, precision=reference_precision, round_mode=gmpy2.RoundUp)
        self.assertGreaterEqual(u_m, true_mass_upper)

    def test_rigorous_attention_output_interval_contains_high_precision_reference(self) -> None:
        query = np.array([0.5, -1.0, 0.25], dtype=np.float64)
        keys = np.array([[0.3, -0.2, 0.1], [-0.5, 0.6, 0.2], [0.1, 0.0, -0.2]], dtype=np.float64)
        values = np.array([[0.2, -0.1], [0.05, 0.3], [-0.25, 0.15]], dtype=np.float64)
        interval_output, _ = rigorous_attention_output_interval(query, keys, values, precision=512)
        reference_output = exact_reference_output_mpfr(query, keys, values, precision=4096)
        for interval_component, reference_component in zip(interval_output, reference_output):
            self.assertLessEqual(interval_component.lower, reference_component)
            self.assertGreaterEqual(interval_component.upper, reference_component)

    def test_one_residual_block_full_skip_is_certified_and_rigorously_valid(self) -> None:
        dimension = 128
        query = np.zeros(dimension, dtype=np.float64)
        query[0] = 6.0
        recent_keys = np.zeros((2, dimension), dtype=np.float64)
        recent_keys[0, 0] = 8.0
        recent_keys[1, 0] = 7.5
        recent_values = np.array([[1.0, -0.1], [0.9, 0.05]], dtype=np.float64)
        block = _make_residual_block(
            block_start=0,
            dimension=dimension,
            block_len=4,
            anchor0=-2.0,
            residual_scale=0.15,
            value_base=0.02,
            value_residual_scale=0.005,
        )

        result = certify_progressive_skipping(
            query=query,
            recent_keys=recent_keys,
            recent_values=recent_values,
            historical_blocks=[block],
            tolerance=0.05,
            mode=CertificateMode.RIGOROUS_REFERENCE,
        )
        self.assertTrue(result.certified)
        self.assertIn(result.chosen_certificate, {"cert1", "cert2"})
        self.assertEqual(result.decoded_block_starts, ())
        self.assertEqual(result.skipped_block_starts, (0,))
        self.assertGreater(block.block_len, 1)
        self.assertGreater(float(block.header.rho_upper), 0.0)
        self.assertFalse(result.output_is_formally_certified)
        _assert_rigorous_certificate_valid(
            self,
            query=query,
            recent_keys=recent_keys,
            recent_values=recent_values,
            blocks=[block],
            result=result,
        )

    def test_multi_residual_blocks_allow_partial_decode_and_certified_skip(self) -> None:
        dimension = 5
        query = np.zeros(dimension, dtype=np.float64)
        query[0] = 4.0
        recent_keys = np.array([[5.5, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        recent_values = np.array([[1.0, 0.0]], dtype=np.float64)
        blocks = [
            _make_residual_block(block_start=0, dimension=dimension, block_len=4, anchor0=4.5, residual_scale=0.3, value_base=0.7, value_residual_scale=0.05),
            _make_residual_block(block_start=4, dimension=dimension, block_len=4, anchor0=-2.0, residual_scale=0.2, value_base=0.05, value_residual_scale=0.01),
            _make_residual_block(block_start=8, dimension=dimension, block_len=3, anchor0=-3.0, residual_scale=0.1, value_base=0.03, value_residual_scale=0.005),
        ]

        result = certify_progressive_skipping(
            query=query,
            recent_keys=recent_keys,
            recent_values=recent_values,
            historical_blocks=blocks,
            tolerance=0.05,
            mode=CertificateMode.RIGOROUS_REFERENCE,
            schedule=DecodeSchedule.LARGEST_U_TIMES_NU,
        )
        self.assertTrue(result.certified)
        self.assertEqual(result.decoded_block_starts, (0,))
        self.assertEqual(set(result.skipped_block_starts), {4, 8})
        skipped_blocks = [block for block in blocks if block.header.block_start in result.skipped_block_starts]
        self.assertTrue(any(block.block_len > 1 and float(block.header.rho_upper) > 0.0 for block in skipped_blocks))
        self.assertFalse(result.output_is_formally_certified)
        _assert_rigorous_certificate_valid(
            self,
            query=query,
            recent_keys=recent_keys,
            recent_values=recent_values,
            blocks=blocks,
            result=result,
        )

    def test_randomized_residual_block_certificates_have_no_violations(self) -> None:
        rng = np.random.default_rng(7)
        dimensions = [2, 5, 128]
        checked_cases = 24
        violations = 0
        rigorous_interval_cases = 0
        for case_index in range(checked_cases):
            dimension = dimensions[case_index % len(dimensions)]
            sign = -1.0 if case_index % 2 else 1.0
            query = np.zeros(dimension, dtype=np.float64)
            query[0] = sign * rng.uniform(5.0, 8.0)
            if dimension > 1:
                query[1] = rng.uniform(-0.5, 0.5)

            recent_keys = np.zeros((2, dimension), dtype=np.float64)
            recent_keys[0, 0] = sign * rng.uniform(8.0, 10.0)
            recent_keys[1, 0] = sign * rng.uniform(7.5, 9.5)
            recent_values = rng.normal(scale=0.5, size=(2, 2)).astype(np.float64)

            blocks = []
            for block_start in range(0, 12, 4):
                block_len = int(rng.integers(2, 9))
                anchor0 = sign * rng.uniform(-5.0, -2.5)
                residual_scale = rng.uniform(0.01, 0.12)
                value_base = rng.uniform(0.001, 0.01)
                value_residual_scale = rng.uniform(0.0005, 0.005)
                blocks.append(
                    _make_residual_block(
                        block_start=block_start,
                        dimension=dimension,
                        block_len=block_len,
                        anchor0=anchor0,
                        residual_scale=residual_scale,
                        value_base=value_base,
                        value_residual_scale=value_residual_scale,
                    )
                )

            result = certify_progressive_skipping(
                query=query,
                recent_keys=recent_keys,
                recent_values=recent_values,
                historical_blocks=blocks,
                tolerance=0.5,
                mode=CertificateMode.RIGOROUS_REFERENCE,
                schedule=DecodeSchedule.LARGEST_U_TIMES_NU,
            )
            self.assertTrue(result.certified)
            self.assertNotEqual(result.chosen_certificate, "zero")
            skipped_blocks = [block for block in blocks if block.header.block_start in result.skipped_block_starts]
            self.assertTrue(skipped_blocks)
            self.assertTrue(any(block.block_len > 1 and float(block.header.rho_upper) > 0.0 for block in skipped_blocks))
            rigorous_interval_cases += 1

            try:
                _assert_rigorous_certificate_valid(
                    self,
                    query=query,
                    recent_keys=recent_keys,
                    recent_values=recent_values,
                    blocks=blocks,
                    result=result,
                    precision=384,
                )
            except AssertionError:
                violations += 1
        self.assertEqual(rigorous_interval_cases, checked_cases)
        self.assertEqual(violations, 0)

    def test_zero_remaining_blocks_yields_zero_certificate(self) -> None:
        query = np.array([0.25, -0.75], dtype=np.float64)
        recent_keys = np.array([[0.1, 0.2]], dtype=np.float64)
        recent_values = np.array([[0.3, 0.4]], dtype=np.float64)
        result = certify_progressive_skipping(
            query=query,
            recent_keys=recent_keys,
            recent_values=recent_values,
            historical_blocks=[],
            tolerance=0.0,
            mode=CertificateMode.RIGOROUS_REFERENCE,
        )
        self.assertEqual(result.certificate_value_text, "0.0")
        self.assertEqual(result.certificate_value_upper_float, 0.0)
        self.assertFalse(result.output_is_formally_certified)
        np.testing.assert_allclose(result.output_vector_approx, exact_reference_output(query, recent_keys, recent_values))

    def test_input_validation_and_failure_modes(self) -> None:
        query = np.array([1.0, 0.0], dtype=np.float64)
        recent_keys = np.array([[1.0, 0.0]], dtype=np.float64)
        recent_values = np.array([[1.0]], dtype=np.float64)
        block = _make_residual_block(
            block_start=0,
            dimension=2,
            block_len=2,
            anchor0=-1.0,
            residual_scale=0.1,
            value_base=0.1,
            value_residual_scale=0.01,
        )
        with self.assertRaises(ValueError):
            certify_progressive_skipping(
                query=query,
                recent_keys=recent_keys,
                recent_values=recent_values,
                historical_blocks=[block],
                tolerance=-1.0,
                mode=CertificateMode.RIGOROUS_REFERENCE,
            )
        with self.assertRaises(ValueError):
            certify_progressive_skipping(
                query=query,
                recent_keys=np.array([[1.0]], dtype=np.float64),
                recent_values=recent_values,
                historical_blocks=[block],
                tolerance=1.0,
                mode=CertificateMode.RIGOROUS_REFERENCE,
            )
        fast = certify_progressive_skipping(
            query=query,
            recent_keys=recent_keys,
            recent_values=recent_values,
            historical_blocks=[block],
            tolerance=1.0,
            mode=CertificateMode.FAST_FP64,
        )
        self.assertFalse(fast.certified)
        self.assertEqual(fast.chosen_certificate, "failure")

    def test_all_zero_query_and_sparse_query_paths(self) -> None:
        zero_query = np.zeros(5, dtype=np.float64)
        recent_keys = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        recent_values = np.array([[0.5, -0.25]], dtype=np.float64)
        block = _make_residual_block(
            block_start=0,
            dimension=5,
            block_len=3,
            anchor0=-0.5,
            residual_scale=0.1,
            value_base=0.02,
            value_residual_scale=0.005,
        )
        zero_result = certify_progressive_skipping(
            query=zero_query,
            recent_keys=recent_keys,
            recent_values=recent_values,
            historical_blocks=[block],
            tolerance=0.2,
            mode=CertificateMode.RIGOROUS_REFERENCE,
        )
        self.assertTrue(zero_result.certified)

        sparse_query = np.zeros(128, dtype=np.float64)
        sparse_query[0] = 4.0
        sparse_query[63] = -0.5
        sparse_query[127] = 0.25
        sparse_recent_keys = np.zeros((2, 128), dtype=np.float64)
        sparse_recent_keys[0, 0] = 6.0
        sparse_recent_keys[1, 63] = -0.25
        sparse_recent_values = np.array([[0.7, -0.1], [0.6, 0.2]], dtype=np.float64)
        sparse_block = _make_residual_block(
            block_start=3,
            dimension=128,
            block_len=4,
            anchor0=-1.0,
            residual_scale=0.08,
            value_base=0.015,
            value_residual_scale=0.003,
        )
        sparse_result = certify_progressive_skipping(
            query=sparse_query,
            recent_keys=sparse_recent_keys,
            recent_values=sparse_recent_values,
            historical_blocks=[sparse_block],
            tolerance=0.1,
            mode=CertificateMode.RIGOROUS_REFERENCE,
        )
        self.assertTrue(sparse_result.certified)

    def test_numerical_failure_falls_back_to_decoding_all_remaining_blocks(self) -> None:
        query = np.array([1.0, 0.0], dtype=np.float64)
        recent_keys = np.array([[1.0, 0.0]], dtype=np.float64)
        recent_values = np.array([[1.0, 0.0]], dtype=np.float64)
        blocks = [
            _make_residual_block(block_start=0, dimension=2, block_len=3, anchor0=0.25, residual_scale=0.1, value_base=0.5, value_residual_scale=0.02),
            _make_residual_block(block_start=3, dimension=2, block_len=4, anchor0=-0.5, residual_scale=0.08, value_base=-0.2, value_residual_scale=0.03),
        ]
        with patch("rack_kv.certificate._remaining_upper_bounds", side_effect=OverflowError("forced failure")):
            result = certify_progressive_skipping(
                query=query,
                recent_keys=recent_keys,
                recent_values=recent_values,
                historical_blocks=blocks,
                tolerance=0.1,
                mode=CertificateMode.RIGOROUS_REFERENCE,
            )
        self.assertTrue(result.certified)
        self.assertEqual(result.chosen_certificate, "zero")
        self.assertEqual(set(result.decoded_block_starts), {0, 3})
        self.assertEqual(result.skipped_block_starts, ())
        self.assertIn("fallback", result.message.lower())

    def test_mpfr_arithmetic_error_falls_back_to_decoding_all_remaining_blocks(self) -> None:
        query = np.array([1.0, 0.0], dtype=np.float64)
        recent_keys = np.array([[1.0, 0.0]], dtype=np.float64)
        recent_values = np.array([[1.0, 0.0]], dtype=np.float64)
        blocks = [
            _make_residual_block(block_start=0, dimension=2, block_len=3, anchor0=0.25, residual_scale=0.1, value_base=0.5, value_residual_scale=0.02),
        ]
        with patch("rack_kv.certificate._remaining_upper_bounds", side_effect=ArithmeticError("forced mpfr failure")):
            result = certify_progressive_skipping(
                query=query,
                recent_keys=recent_keys,
                recent_values=recent_values,
                historical_blocks=blocks,
                tolerance=0.1,
                mode=CertificateMode.RIGOROUS_REFERENCE,
            )
        self.assertTrue(result.certified)
        self.assertEqual(result.chosen_certificate, "zero")
        self.assertEqual(set(result.decoded_block_starts), {0})


if __name__ == "__main__":
    unittest.main()
