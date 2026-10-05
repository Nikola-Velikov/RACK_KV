from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rack_kv.codec_v2 import encode_v2_block, hadamard_transform, pack_int4, unpack_int4
from rack_kv.physical import KVPayloadStore


class V2CodecTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.keys = rng.normal(size=(8, 8))
        self.values = rng.normal(size=(8, 4))

    def test_hadamard_is_orthogonal_and_deterministic(self):
        transformed = hadamard_transform(self.keys)
        np.testing.assert_allclose(hadamard_transform(transformed), self.keys, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(transformed, hadamard_transform(self.keys))

    def test_int4_round_trip(self):
        values = np.array([-8, -7, -1, 0, 1, 7, 6], dtype=np.int8)
        np.testing.assert_array_equal(unpack_int4(pack_int4(values), len(values)), values)

    def test_groupwise_scales_and_v2_round_trip(self):
        block = encode_v2_block(self.keys, self.values, key_codec="hadamard_int8", value_codec="hadamard_int4", group_size=4, block_start=8)
        restored = type(block).deserialize(block.serialize())
        keys, values = restored.decode_block()
        self.assertEqual(restored.block_start, 8)
        self.assertEqual(len(restored.key_scales), 2)
        self.assertEqual(len(restored.value_scales), 1)
        self.assertEqual(restored.key_codec, "hadamard_int8")
        self.assertEqual(restored.value_codec, "hadamard_int4")
        self.assertEqual(keys.shape, self.keys.shape)
        self.assertEqual(values.shape, self.values.shape)

    def test_int4_is_packed_not_int8(self):
        block = encode_v2_block(self.keys, self.values, key_codec="hadamard_int4", value_codec="hadamard_int4", group_size=4)
        self.assertEqual(len(block.key_payload), ((7 * 8) + 1) // 2)
        self.assertEqual(len(block.value_payload), ((7 * 4) + 1) // 2)

    def test_random_access_reads_only_requested_v2_block(self):
        blocks = [encode_v2_block(self.keys, self.values, key_codec="hadamard_int4", value_codec="hadamard_int4", group_size=4, block_start=i * 8) for i in range(2)]
        with tempfile.NamedTemporaryFile(suffix=".rackv2", delete=False) as temp:
            path = Path(temp.name)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        store = KVPayloadStore.create(path, blocks)
        decoded = store.decode_block(1, key_dim=8, value_dim=4, decoder=type(blocks[1]).deserialize)
        self.assertEqual(decoded.block_start, 8)
        self.assertEqual(store.stats.blocks_loaded, 1)
        self.assertEqual(store.stats.blocks_decoded, 1)

    def test_different_blocks_are_independent(self):
        first = encode_v2_block(self.keys, self.values, key_codec="hadamard_int8", value_codec="hadamard_int8", group_size=4, block_start=0)
        second = encode_v2_block(self.keys + 100, self.values - 100, key_codec="hadamard_int8", value_codec="hadamard_int8", group_size=4, block_start=8)
        self.assertFalse(np.allclose(first.decode_block()[0], second.decode_block()[0]))


if __name__ == "__main__":
    unittest.main()
