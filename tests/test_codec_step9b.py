import unittest
import numpy as np

from rack_kv.codec_step9b import encode_step9b, Step9BBlock, _anchor_index


class Step9BTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.keys = rng.normal(size=(8, 16))
        self.values = rng.normal(size=(8, 16))

    def test_anchor_is_deterministic_and_in_block(self):
        self.assertEqual(_anchor_index(self.keys, "medoid"), _anchor_index(self.keys, "medoid"))
        self.assertIn(_anchor_index(self.keys, "minimax"), range(8))

    def test_mixed_roundtrip_and_anchor_index(self):
        block = encode_step9b(self.keys, self.values, key_policy="mixed", value_policy="per_token_int4", group_size=8, mixed_threshold=.02, block_start=4)
        restored = Step9BBlock.deserialize(block.serialize())
        keys, values = restored.decode_block()
        self.assertEqual(restored.block_start, 4)
        self.assertEqual(restored.key_anchor_index, block.key_anchor_index)
        self.assertEqual(restored.value_anchor_index, block.value_anchor_index)
        self.assertTrue(np.all(np.isfinite(keys)))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_int4_is_packed(self):
        block = encode_step9b(self.keys, self.values, key_policy="per_token_int4", value_policy="per_token_int4")
        self.assertLess(len(block.key_payload), (len(self.keys) - 1) * self.keys.shape[1])

    def test_kv_policies_are_independent(self):
        block = encode_step9b(self.keys, self.values, key_policy="per_token_int8", value_policy="per_token_int4")
        self.assertEqual(block.key_policy, "per_token_int8")
        self.assertEqual(block.value_policy, "per_token_int4")


if __name__ == "__main__":
    unittest.main()
