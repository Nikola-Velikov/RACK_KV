from __future__ import annotations

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import hashlib
import json
import zipfile

import gmpy2
import numpy as np

from experiments.common import load_config, ROOT, verify_review_archive
from experiments.audit_v1 import finite_tree
from experiments.profiling import geometry, candidate_diagnostics
from experiments.results import summarize
from experiments.compare_to_v1 import compare
from rack_kv.codec import encode_block, serialize_block_container, deserialize_block_container, CompressedBlock, _rigorous_block_metadata
from rack_kv.certificate import certify_progressive_skipping, rigorous_attention_output_interval, rigorous_output_error_upper_from_intervals
from rack_kv.stage5 import _gqa_groups, _certified_attention_for_head, _shared_reconstructed_attention_output_numpy


class FrozenV1Tests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.blocks = [encode_block(self.rng.normal(size=(4, 4)), self.rng.normal(size=(4, 3)), block_start=i * 4, precision=256) for i in range(3)]

    def test_frozen_source_and_configuration(self):
        self.assertEqual(load_config(ROOT / "configs/rack_kv_v1.yaml")["mpfr_precision"], 256)

    def test_arbitrary_byte_offset_decode_equals_complete_decode(self):
        container = serialize_block_container(self.blocks)
        full_k = np.vstack([b.decode_key_block() for b in self.blocks])
        full_v = np.vstack([b.decode_value_block() for b in self.blocks])
        for j in (2, 0, 1):
            start, length = container.block_offsets[j], container.block_lengths[j]
            block = CompressedBlock.deserialize(container.buffer[start:start + length], key_dim=4, value_dim=3)
            np.testing.assert_array_equal(block.decode_key_block(), full_k[j * 4:(j + 1) * 4])
            np.testing.assert_array_equal(block.decode_value_block(), full_v[j * 4:(j + 1) * 4])

    def test_roundtrip_and_disjoint_accounting(self):
        container = serialize_block_container(self.blocks)
        loaded = deserialize_block_container(container.buffer)
        categories = {"anchors": 0, "residuals": 0, "scales": 0, "metadata": 0,
                      "index": container.index_bytes, "header": container.header_bytes}
        for i, block in enumerate(self.blocks):
            self.assertEqual(block.serialize(), loaded.deserialize_block(i).serialize())
            categories["anchors"] += block.header.anchor_key.nbytes + block.header.anchor_value.nbytes
            categories["residuals"] += block.key_residuals.nbytes + block.value_residuals.nbytes
            categories["scales"] += 4
            categories["metadata"] += 16
        self.assertEqual(sum(categories.values()), len(container.buffer))

    def test_outward_geometry_metadata_and_cap(self):
        for block in self.blocks:
            rho, nu = _rigorous_block_metadata(block, precision=512)
            self.assertLessEqual(rho, block.header.rho_upper)
            self.assertLessEqual(nu, block.header.nu_upper)
            d = candidate_diagnostics(np.ones(4), block, .5, 256)
            self.assertGreaterEqual(d["logit_bound_slack"], 0)
            self.assertGreaterEqual(d["mass_bound_ratio"], 1)

    def test_profiler_does_not_mutate_payload(self):
        block = self.blocks[0]
        before = block.serialize()
        result = geometry(block.decode_key_block())
        self.assertEqual(block.serialize(), before)
        self.assertLessEqual(result["effective_rank_energy_entropy"], 4)
        self.assertEqual(result["residual_energy_rank_8"], 0)

    def test_rank_one_and_zero_geometry(self):
        rank1 = geometry(np.arange(8)[:, None] * np.ones((1, 4)))
        self.assertAlmostEqual(rank1["effective_rank_energy_entropy"], 1)
        self.assertLess(rank1["residual_energy_fraction_rank_1"], 1e-25)
        zero = geometry(np.ones((8, 4)))
        self.assertEqual(zero["effective_rank_energy_entropy"], 0)

    def test_rigorous_skip_safety_nonzero_fixture(self):
        query = np.array([1., 0.])
        block = encode_block(np.array([[-10., 0.], [-10., .1]]), np.array([[.1, .2], [.2, .1]]), block_start=0)
        recent_k, recent_v = np.array([[10., 0.]]), np.array([[1., 0.]])
        result = certify_progressive_skipping(query=query, recent_keys=recent_k, recent_values=recent_v,
                                             historical_blocks=[block], tolerance=.05, precision=256)
        self.assertTrue(result.certified)
        self.assertTrue(result.skipped_block_starts)
        self.assertIsNotNone(result.certificate_value_mpfr)
        full, _ = rigorous_attention_output_interval(query, np.vstack([block.decode_key_block(), recent_k]),
                                                     np.vstack([block.decode_value_block(), recent_v]), precision=512)
        kept, _ = rigorous_attention_output_interval(query, recent_k, recent_v, precision=512)
        upper = rigorous_output_error_upper_from_intervals(full, kept, precision=512)
        self.assertLessEqual(upper, result.certificate_value_mpfr)
        self.assertGreater(gmpy2.mpfr(result.z_k_lower_text), 0)

    def test_gqa_all_heads(self):
        groups = _gqa_groups(32, 8)
        self.assertEqual(groups, tuple(tuple(range(4 * i, 4 * i + 4)) for i in range(8)))

    def attention_fixture(self):
        recent_k, recent_v = self.rng.normal(size=(4, 4)), self.rng.normal(size=(4, 3))
        keys = np.vstack([b.decode_key_block() for b in self.blocks] + [recent_k])
        values = np.vstack([b.decode_value_block() for b in self.blocks] + [recent_v])
        return dict(query=np.ones(4), reconstructed_full_keys=keys, reconstructed_full_values=values,
                    prefix_result=SimpleNamespace(blocks=self.blocks), recent_keys=recent_k, recent_values=recent_v,
                    tolerance=.05, precision=256, scaling=.5)

    def test_prefilter_cannot_authorize_and_zero_skip_exact_parity(self):
        args = self.attention_fixture()
        with patch("rack_kv.stage5.certify_progressive_skipping", side_effect=AssertionError("MPFR must not run on rejected prefilter")):
            output, record = _certified_attention_for_head(**args, prefilter_bound_override=1.)
        expected = _shared_reconstructed_attention_output_numpy(query=args["query"], keys=args["reconstructed_full_keys"], values=args["reconstructed_full_values"], scaling=.5)
        np.testing.assert_array_equal(output, expected)
        self.assertFalse(record["mpfr_invoked"])
        self.assertEqual(record["skipped_block_starts"], [])

    def test_passed_prefilter_still_requires_mpfr(self):
        args = self.attention_fixture()
        with patch("rack_kv.stage5.certify_progressive_skipping", side_effect=RuntimeError("required authority")):
            with self.assertRaisesRegex(RuntimeError, "required authority"):
                _certified_attention_for_head(**args, prefilter_bound_override=0.)

    def test_aggregation_rejects_duplicates_and_bad_bytes(self):
        row = dict(case_key="a", method_name="full_kv", mode="native", encoded_key_bytes=4, encoded_value_bytes=4,
                   scales_bytes=0, metadata_bytes=0, indices_bytes=0, block_page_metadata_bytes=0, recent_window_bytes=0,
                   total_serialized_bytes=8, attention_output_l2_error=0., compression_ratio_vs_full_kv=1., memory_saving_fraction_vs_full_kv=0.)
        self.assertEqual(summarize([row])["full_kv:native"]["mean_serialized_bytes"], 8)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize([row, row])
        with self.assertRaisesRegex(ValueError, "storage mismatch"):
            summarize([{**row, "total_serialized_bytes": 16}])

    def test_comparison_rejects_mixed_population_and_flags_regression(self):
        base = {"identity": {"selection": {"layers": [0]}, "method_mode_plan": []},
                "scientific_source_unchanged": True, "representative": {"rack_kv:native": {"case_count": 1, "false_safe_count": 0}}}
        new = copy.deepcopy(base)
        new["representative"]["rack_kv:native"]["false_safe_count"] = 1
        self.assertTrue(compare(base, new)["warnings"])
        new["identity"]["selection"]["layers"] = [31]
        with self.assertRaisesRegex(ValueError, "populations"):
            compare(base, new)

    def test_archive_verification_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "review.zip"
            payload = b"real evidence"
            manifest = {"files": [{"relative_path": "data.txt", "size_bytes": len(payload),
                                   "sha256": hashlib.sha256(payload).hexdigest()}]}
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("data.txt", payload)
                archive.writestr("manifest.json", json.dumps(manifest))
            self.assertEqual(verify_review_archive(path)["mismatch_count"], 0)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("data.txt", b"changed")
                archive.writestr("manifest.json", json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_review_archive(path)

    def test_nonfinite_export_audit(self):
        finite_tree({"missing": None, "zero": 0})
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            finite_tree({"value": float("nan")})

    def test_full_model_storage_scopes_are_not_mixed(self):
        data = json.loads((ROOT / ".tmp/stage5_quality_final/stage5_quality_final_results.json").read_text())
        for method, storage in data["aggregate_storage"].items():
            expected = (storage["cumulative_total_serialized_bytes"] if method == "full_kv"
                        else storage["modified_layers_cumulative_total_serialized_bytes"])
            self.assertEqual(sum(storage["category_sums"].values()), expected)


if __name__ == "__main__":
    unittest.main()
