import unittest
import numpy as np

from rack_kv.compression_certificate import build_error_metadata, certify_compression, certify_compression_tight


class CompressionCertificateTests(unittest.TestCase):
    def test_metadata_bounds_and_recent_zero(self):
        original = np.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        reconstructed = original + .01
        metadata = build_error_metadata(original, reconstructed, original, reconstructed, block_ranges=((0, 2),), precision=128)
        self.assertTrue(all(float(a) >= np.linalg.norm(x - y) for a, x, y in zip(metadata.kappa, original, reconstructed)))

    def test_certificate_has_nonnegative_bound_without_original_at_query(self):
        original = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        reconstructed = original + .01
        metadata = build_error_metadata(original, reconstructed, original, reconstructed, precision=128)
        result = certify_compression(np.asarray([1.0, 0.5]), reconstructed, reconstructed, metadata, precision=128, attention_scale=1.0)
        self.assertGreaterEqual(float(result.e_comp), 0.0)

    def test_tight_intervals_and_composed_bound_are_safe_on_fixture(self):
        original = np.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        reconstructed = original + .01
        metadata = build_error_metadata(original, reconstructed, original, reconstructed, precision=128)
        result = certify_compression_tight(
            np.asarray([1.0, 0.5]), reconstructed, reconstructed, metadata,
            precision=128, attention_scale=1.0,
            centers={"zero": np.zeros(2), "mean": reconstructed.mean(0), "ohat": np.asarray([.5, .5])},
        )
        logits_exact = original @ np.asarray([1.0, .5])
        p = np.exp(logits_exact - logits_exact.max()); p /= p.sum()
        self.assertTrue(all(float(lo) <= x <= float(hi) for x, lo, hi in zip(p, result.lower_probabilities, result.upper_probabilities)))
        self.assertGreaterEqual(float(result.e_comp_tight), 0.0)
        self.assertLessEqual(float(result.e_probability_tight), float(result.e_probability_coordinate))

    def test_centered_probability_identity(self):
        values = np.asarray([[1.0, 2.0], [3.0, -1.0], [2.0, .5]])
        d = np.asarray([.1, -.04, -.06])
        center = values.mean(0)
        np.testing.assert_allclose(d @ values, d @ (values - center))


if __name__ == "__main__":
    unittest.main()
