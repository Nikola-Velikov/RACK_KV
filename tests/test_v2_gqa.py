from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np

from experiments.common import ROOT
from rack_kv.codec import encode_block
from rack_kv.gqa import (
    certify_gqa_hierarchical_group,
    concatenation_bound,
    gqa_groups,
    output_projection_group_columns,
    projected_group_bound,
)
from rack_kv.hierarchy import build_hierarchy
from rack_kv.rigorous import exact_mpfr


class V2GQATests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(441)
        self.blocks = []
        for start in (0, 8):
            keys = np.column_stack((np.full(8, -12.0 - start / 100.0), rng.normal(scale=0.05, size=8)))
            values = rng.normal(scale=0.05, size=(8, 2))
            self.blocks.append(encode_block(keys, values, block_start=start, precision=256))
        self.index = build_hierarchy(self.blocks, fanout=2, rank=2, precision=256)
        payloads = [block.decode_block() for block in self.blocks]
        self.leaf_keys = [payload[0] for payload in payloads]
        self.leaf_values = [payload[1] for payload in payloads]
        self.heads = (0, 1, 2, 3)
        self.queries = {head: np.array([1.0, 0.0]) for head in self.heads}
        self.recent_keys = {head: np.array([[12.0, 0.0]]) for head in self.heads}
        self.recent_values = {head: np.array([[1.0, 0.0]]) for head in self.heads}
        self.w_o = np.eye(8)

    def _run(self, **overrides):
        arguments = dict(
            kv_head=0, query_by_head=self.queries, recent_keys_by_head=self.recent_keys,
            recent_values_by_head=self.recent_values, index=self.index, leaf_keys=self.leaf_keys,
            leaf_values=self.leaf_values, epsilon_head=0.05, precision=256,
            output_projection=self.w_o, num_attention_heads=4, num_key_value_heads=1,
            head_dim=2, leaf_payload_bytes={0: 100, 8: 100},
        )
        arguments.update(overrides)
        return certify_gqa_hierarchical_group(**arguments)

    def test_mapping_comes_from_configured_gqa_ratio(self):
        groups = gqa_groups(num_attention_heads=32, num_key_value_heads=8)
        self.assertEqual(groups[0], (0, 1, 2, 3))
        self.assertEqual(groups[1], (4, 5, 6, 7))
        self.assertEqual(groups[-1], (28, 29, 30, 31))

    def test_all_heads_pass_produces_physical_eligibility(self):
        result = self._run()
        self.assertGreater(len(result.physical_eligible_regions), 0)
        decision = next(item for item in result.region_decisions if item.all_heads_passed)
        self.assertEqual(decision.gqa_skip_vote_count, 4)
        self.assertEqual(set(decision.mapped_query_heads), set(self.heads))
        self.assertIsNotNone(decision.b_concat)
        self.assertIsNotNone(decision.b_group)
        self.assertGreater(result.potential_payload_bytes_avoided, 0)

    def test_one_failing_head_prevents_physical_eligibility(self):
        queries = dict(self.queries)
        queries[3] = np.array([-1.0, 0.0])
        result = self._run(query_by_head=queries)
        root = next(item for item in result.region_decisions if item.region_id == self.index.root_id)
        self.assertFalse(root.all_heads_passed)
        self.assertLess(root.gqa_skip_vote_count, 4)
        self.assertEqual(result.physical_eligible_regions, ())

    def test_cumulative_bounds_and_descendant_coverage(self):
        result = self._run()
        for decision in result.region_decisions:
            for head in decision.head_decisions:
                if head.passed:
                    self.assertLessEqual(float(head.candidate_bound_after), 0.05)
                    self.assertTrue(head.rigorous)
        for region in result.physical_eligible_regions:
            self.assertTrue(set(self.index.descendants(region)).issubset(set(result.physical_eligible_leaf_blocks)))

    def test_concat_and_projected_frobenius_bounds_contain_fixture_errors(self):
        deltas = [np.array([0.1, -0.2]), np.array([0.0, 0.3]), np.array([-0.1, 0.1]), np.array([0.05, 0.0])]
        actual_concat = float(np.linalg.norm(np.concatenate(deltas)))
        head_bounds = [exact_mpfr(np.linalg.norm(delta) + 1e-4, precision=256) for delta in deltas]
        concat, projected = projected_group_bound(head_bounds, output_projection_group_columns(self.w_o, query_heads=self.heads, head_dim=2), precision=256)
        actual_projected = float(np.linalg.norm(self.w_o @ np.concatenate(deltas)))
        self.assertLessEqual(actual_concat, float(concat))
        self.assertLessEqual(actual_projected, float(projected))
        self.assertEqual(str(concat), str(concatenation_bound(head_bounds, precision=256)))

    def test_v1_lock_sources_are_unchanged(self):
        lock = json.loads((ROOT / "configs/rack_kv_v1.lock.json").read_text(encoding="utf-8"))
        for relative, expected in lock["source_files"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected, relative)


if __name__ == "__main__":
    unittest.main()
