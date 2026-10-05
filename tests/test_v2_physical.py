from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rack_kv.codec import encode_block
from rack_kv.physical import KVPayloadStore, execute_logical_gqa, execute_physical_gqa


class V2PhysicalPayloadTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.blocks = [
            encode_block(rng.normal(size=(4, 3)), rng.normal(size=(4, 2)), block_start=0, precision=256),
            encode_block(rng.normal(size=(4, 3)), rng.normal(size=(4, 2)), block_start=4, precision=256),
        ]
        self.queries = {head: np.array([0.2, -0.1, 0.3]) for head in range(4)}
        self.recent_keys = {head: np.array([[0.1, 0.0, 0.2]]) for head in range(4)}
        self.recent_values = {head: np.array([[0.4, -0.2]]) for head in range(4)}

    def _store(self):
        temp = tempfile.NamedTemporaryFile(suffix=".rackv2", delete=False)
        temp.close()
        self.addCleanup(lambda: Path(temp.name).unlink(missing_ok=True))
        return KVPayloadStore.create(Path(temp.name), self.blocks)

    def _run(self, store, **kwargs):
        arguments = dict(
            store=store,
            queries_by_head=self.queries,
            recent_keys_by_head=self.recent_keys,
            recent_values_by_head=self.recent_values,
            required_leaf_blocks=(0,),
            physically_eligible_blocks=(1,),
            complete_gqa_group=True,
            all_heads_represented=True,
            mpfr_authorized=True,
            numerical_fallback=False,
            key_dim=3,
            value_dim=2,
        )
        arguments.update(kwargs)
        return execute_physical_gqa(**arguments)

    def test_metadata_open_does_not_read_payload(self):
        store = self._store()
        self.assertEqual(store.stats.payload_bytes_read, 0)
        self.assertEqual(store.stats.payload_read_calls, 0)
        self.assertEqual(store.get_metadata(1).block_id, 1)
        self.assertEqual(store.stats.payload_bytes_read, 0)

    def test_forbidden_eligible_block_is_never_read_or_decoded(self):
        store = self._store()
        store.forbidden_block_ids.add(1)
        result = self._run(store)
        self.assertEqual(result.physically_omitted_block_ids, (1,))
        self.assertEqual(store.stats.blocks_decoded, 1)
        self.assertEqual(store.stats.payload_read_calls, 1)

    def test_required_block_is_loaded_once_for_all_four_heads(self):
        store = self._store()
        result = self._run(store)
        self.assertEqual(set(result.outputs_by_head), {0, 1, 2, 3})
        self.assertEqual(store.stats.blocks_loaded, 1)
        self.assertEqual(store.stats.blocks_decoded, 1)

    def test_logical_and_physical_outputs_match(self):
        logical_store = self._store()
        logical = execute_logical_gqa(
            store=logical_store, required_leaf_blocks=(0,), queries_by_head=self.queries,
            recent_keys_by_head=self.recent_keys, recent_values_by_head=self.recent_values,
            key_dim=3, value_dim=2,
        )
        physical = self._run(self._store())
        for head in self.queries:
            np.testing.assert_allclose(logical[head], physical.outputs_by_head[head], rtol=0, atol=1e-12)

    def test_incomplete_group_fails_closed(self):
        store = self._store()
        result = self._run(store, complete_gqa_group=False)
        self.assertEqual(result.physically_omitted_block_ids, ())
        self.assertEqual(result.required_block_ids, (0, 1))
        self.assertIn("incomplete_gqa_group", result.fallback_reasons)
        self.assertEqual(store.stats.blocks_decoded, 2)

    def test_mpfr_uncertainty_fails_closed(self):
        store = self._store()
        result = self._run(store, mpfr_authorized=False)
        self.assertEqual(result.physically_omitted_block_ids, ())
        self.assertIn("mpfr_not_authorized", result.fallback_reasons)
        self.assertEqual(store.stats.blocks_loaded, 2)

    def test_missing_eligibility_set_fails_closed(self):
        store = self._store()
        result = self._run(store, physically_eligible_blocks=())
        self.assertEqual(result.physically_omitted_block_ids, ())
        self.assertIn("eligibility_required_set_mismatch", result.fallback_reasons)


if __name__ == "__main__":
    unittest.main()
