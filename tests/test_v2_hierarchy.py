from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np

from experiments.common import ROOT, load_config
from rack_kv.anisotropic import anisotropic_logit_upper_bound, execute_anisotropic_flat
from rack_kv.codec import encode_block
from rack_kv.hierarchy import (
    build_hierarchy,
    execute_anisotropic_hierarchical,
    select_kept_leaves,
    validate_hierarchy,
)


class V2HierarchyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260927)
        self.query = np.array([1.0, 0.0, 0.0, 0.0])
        self.recent_keys = np.array([[12.0, 0.0, 0.0, 0.0]])
        self.recent_values = np.array([[1.0, 0.0]])
        self.blocks = []
        for start in (0, 8, 16, 24):
            keys = np.column_stack((np.full(8, -12.0 - start / 100.0), rng.normal(scale=0.1, size=(8, 3))))
            values = rng.normal(scale=0.1, size=(8, 2))
            self.blocks.append(encode_block(keys, values, block_start=start, precision=256))

    def test_tree_ranges_cover_leaves_exactly_and_parent_is_union(self):
        index = build_hierarchy(self.blocks, fanout=2, rank=4, precision=256)
        validate_hierarchy(index)
        leaves = [index.nodes[node_id] for node_id in index.leaf_node_ids]
        self.assertEqual([(leaf.token_start, leaf.token_end) for leaf in leaves], [(0, 8), (8, 16), (16, 24), (24, 32)])
        root = index.nodes[index.root_id]
        self.assertEqual((root.token_start, root.token_end), (0, 32))

    def test_every_node_summary_contains_reconstructed_keys(self):
        index = build_hierarchy(self.blocks, fanout=2, rank=4, precision=256)
        for node in index.nodes.values():
            # Internal node coverage is validated through its descendants.
            rows = []
            for leaf_id in index.descendants(node.node_id):
                leaf = index.nodes[leaf_id]
                rows.append(self.blocks[leaf.leaf_index].decode_key_block())
            keys = np.vstack(rows)
            cap = anisotropic_logit_upper_bound(self.query, node.summary, precision=256).upper
            self.assertLessEqual(float(np.max(keys @ self.query / np.sqrt(keys.shape[1]))), float(cap))

    def test_hierarchy_uses_one_cumulative_mpfr_budget_and_removes_entries(self):
        result = execute_anisotropic_hierarchical(
            query=self.query, recent_keys=self.recent_keys, recent_values=self.recent_values,
            historical_blocks=self.blocks, rank=4, fanout=2, tolerance=0.05, precision=256,
        )
        self.assertGreater(len(result.skipped_block_starts), 0)
        self.assertLess(result.attention_token_count, 33)
        self.assertLessEqual(float(result.traversal.certificate_bound), 0.05)
        self.assertFalse(result.traversal.numerical_fallback_used)

    def test_no_skip_equals_compression_only_output(self):
        result = execute_anisotropic_flat(
            query=np.array([0.0, 1.0, 0.0, 0.0]), recent_keys=self.recent_keys,
            recent_values=self.recent_values, historical_blocks=self.blocks, rank=4,
            tolerance=0.0, precision=256,
        )
        self.assertEqual(result.skipped_block_starts, ())
        np.testing.assert_array_equal(result.output, result.compression_only_output)
        hierarchical = execute_anisotropic_hierarchical(
            query=np.array([0.0, 1.0, 0.0, 0.0]), recent_keys=self.recent_keys,
            recent_values=self.recent_values, historical_blocks=self.blocks, rank=4,
            fanout=2, tolerance=0.0, precision=256,
        )
        self.assertEqual(hierarchical.skipped_block_starts, ())
        np.testing.assert_array_equal(hierarchical.output, hierarchical.compression_only_output)

    def test_flat_skip_has_mpfr_authorization(self):
        result = execute_anisotropic_flat(
            query=self.query, recent_keys=self.recent_keys, recent_values=self.recent_values,
            historical_blocks=self.blocks, rank=4, tolerance=0.05, precision=256,
        )
        self.assertTrue(result.certificate.rigorous)
        self.assertGreater(len(result.skipped_block_starts), 0)

    def test_disabling_hierarchy_delegates_to_flat_mode(self):
        flat = execute_anisotropic_flat(
            query=self.query, recent_keys=self.recent_keys, recent_values=self.recent_values,
            historical_blocks=self.blocks, rank=4, tolerance=0.05, precision=256,
        )
        disabled = execute_anisotropic_hierarchical(
            query=self.query, recent_keys=self.recent_keys, recent_values=self.recent_values,
            historical_blocks=self.blocks, rank=4, fanout=2, tolerance=0.05, precision=256,
            hierarchy_enabled=False,
        )
        self.assertEqual(flat.skipped_block_starts, disabled.skipped_block_starts)
        np.testing.assert_array_equal(flat.output, disabled.output)

    def test_leaf_rank_eight_cannot_exceed_seven_nonzero_residual_directions(self):
        keys = self.blocks[0].decode_key_block()
        self.assertLessEqual(np.linalg.matrix_rank(keys - keys[0]), 7)

    def test_frozen_v1_configuration_is_unchanged(self):
        config = load_config(ROOT / "configs/rack_kv_v1.yaml")
        self.assertEqual((config["recent_window"], config["block_size"], config["epsilon"], config["mpfr_precision"]), (16, 8, 0.05, 256))

    def test_all_locked_v1_sources_remain_identical(self):
        lock = json.loads((ROOT / "configs/rack_kv_v1.lock.json").read_text(encoding="utf-8"))
        for relative, expected in lock["source_files"].items():
            observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(observed, expected, relative)


if __name__ == "__main__":
    unittest.main()
