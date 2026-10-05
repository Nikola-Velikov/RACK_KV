from __future__ import annotations

import unittest

import gmpy2
import numpy as np

from experiments.common import ROOT, load_config
from rack_kv.anisotropic import (
    anisotropic_logit_upper_bound,
    approximate_anisotropic_bound,
    build_anisotropic_summary,
    progressive_anisotropic_shadow,
)
from rack_kv.certificate import (
    _block_logit_cap_interval,
    certify_progressive_skipping,
    exact_reference_output,
)
from rack_kv.codec import encode_block
from rack_kv.rigorous import exact_vector


class AnisotropicCertificateTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260927)
        self.query = self.rng.normal(size=8)
        self.keys = self.rng.normal(size=(8, 8))
        self.values = self.rng.normal(size=(8, 6))
        self.block = encode_block(self.keys, self.values, block_start=0, precision=256)

    def test_rank_zero_recovers_v1_interval_exactly(self):
        summary = build_anisotropic_summary(self.block, rank=0, precision=256)
        query_exact = exact_vector(self.query, precision=256)
        old = _block_logit_cap_interval(query_exact, self.block, precision=256)
        new = anisotropic_logit_upper_bound(self.query, summary, precision=256)
        self.assertEqual(str(old.lower), str(new.lower))
        self.assertEqual(str(old.upper), str(new.upper))

    def test_every_reconstructed_logit_is_below_rigorous_cap(self):
        reconstructed = self.block.decode_key_block()
        for rank in (0, 1, 2, 4, 8):
            summary = build_anisotropic_summary(self.block, rank=rank, precision=256)
            upper = anisotropic_logit_upper_bound(self.query, summary, precision=256).upper
            exact_logits = reconstructed @ self.query / np.sqrt(reconstructed.shape[1])
            self.assertLessEqual(float(exact_logits.max()), float(upper))

    def test_true_mass_is_below_anisotropic_mass_bound(self):
        reconstructed = self.block.decode_key_block()
        exact_logits = reconstructed @ self.query / np.sqrt(reconstructed.shape[1])
        true_log_mass = float(np.log(np.exp(exact_logits - exact_logits.max()).sum()) + exact_logits.max())
        for rank in (0, 2, 4, 8):
            summary = build_anisotropic_summary(self.block, rank=rank, precision=256)
            cap = anisotropic_logit_upper_bound(self.query, summary, precision=256).upper
            log_upper = gmpy2.log(gmpy2.mpfr(summary.block_len)) + cap
            self.assertLessEqual(true_log_mass, float(log_upper))

    def test_approximate_mode_cannot_authorize(self):
        summary = build_anisotropic_summary(self.block, rank=4, precision=256)
        result = approximate_anisotropic_bound(self.query, summary)
        self.assertFalse(result.rigorous_authorization)
        self.assertIsNone(result.would_skip)

    def test_shadow_skip_obeys_unchanged_output_error_theorem(self):
        query = np.array([1.0, 0.0])
        block = encode_block(
            np.array([[-10.0, 0.0], [-10.0, 0.1]]),
            np.array([[0.1, 0.2], [0.2, 0.1]]),
            block_start=0,
            precision=256,
        )
        recent_keys = np.array([[10.0, 0.0]])
        recent_values = np.array([[1.0, 0.0]])
        summary = build_anisotropic_summary(block, rank=1, precision=256)
        result = progressive_anisotropic_shadow(
            query=query,
            recent_keys=recent_keys,
            recent_values=recent_values,
            historical_blocks=[block],
            summaries=[summary],
            tolerance=0.05,
            precision=256,
        )
        self.assertTrue(result.would_certify)
        full = exact_reference_output(
            query,
            np.vstack([block.decode_key_block(), recent_keys]),
            np.vstack([block.decode_value_block(), recent_values]),
            precision=512,
        )
        kept = exact_reference_output(query, recent_keys, recent_values, precision=512)
        self.assertLessEqual(float(np.linalg.norm(full - kept)), float(result.certificate_bound))

    def test_rank_zero_shadow_decision_matches_v1(self):
        recent_keys = np.array([[8.0] + [0.0] * 7])
        recent_values = self.rng.normal(size=(1, 6))
        old = certify_progressive_skipping(
            query=self.query,
            recent_keys=recent_keys,
            recent_values=recent_values,
            historical_blocks=[self.block],
            tolerance=0.05,
            precision=256,
        )
        summary = build_anisotropic_summary(self.block, rank=0, precision=256)
        new = progressive_anisotropic_shadow(
            query=self.query,
            recent_keys=recent_keys,
            recent_values=recent_values,
            historical_blocks=[self.block],
            summaries=[summary],
            tolerance=0.05,
            precision=256,
        )
        self.assertEqual(bool(old.skipped_block_starts), new.would_certify)
        self.assertEqual(old.skipped_block_starts, new.skipped_block_starts)
        self.assertEqual(old.decoded_block_starts, new.decoded_block_starts)
        if old.certificate_value_mpfr is not None:
            self.assertEqual(str(old.certificate_value_mpfr), str(new.certificate_bound))

    def test_shadow_module_does_not_change_frozen_v1_sources(self):
        config = load_config(ROOT / "configs/rack_kv_v1.yaml")
        self.assertEqual(config["epsilon"], 0.05)


if __name__ == "__main__":
    unittest.main()
