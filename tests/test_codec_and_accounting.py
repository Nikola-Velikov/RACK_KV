from __future__ import annotations

import unittest

import numpy as np

from rack_kv.accounting import compressed_block_bytes, historical_memory_report, original_block_bytes
from rack_kv.codec import (
    BlockHeader,
    CompressedBlock,
    _quantize_residuals,
    _rigorous_block_metadata,
    deserialize_block_container,
    encode_block,
    serialize_block_container,
)
from rack_kv.types import LLAMA_3_1_8B_GEOMETRY


class CodecAndAccountingTests(unittest.TestCase):
    def test_block_decode_is_local_and_shapes_match(self) -> None:
        keys = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.5, 2.5, 3.5, 4.5],
                [0.5, 1.5, 2.5, 3.5],
            ],
            dtype=np.float64,
        )
        values = np.array(
            [
                [0.2, -0.1, 0.0, 1.0],
                [0.1, -0.2, 0.3, 0.9],
                [0.3, -0.4, 0.5, 0.8],
            ],
            dtype=np.float64,
        )
        block = encode_block(keys, values, block_start=32)
        self.assertEqual(block.block_len, 3)
        decoded_keys, decoded_values = block.decode_block()
        self.assertEqual(decoded_keys.shape, keys.shape)
        self.assertEqual(decoded_values.shape, values.shape)
        token_key, token_value = block.decode_token(2)
        np.testing.assert_allclose(token_key, decoded_keys[2])
        np.testing.assert_allclose(token_value, decoded_values[2])
        self.assertGreaterEqual(float(block.header.rho_upper), np.linalg.norm(decoded_keys[2] - decoded_keys[0]))
        self.assertGreaterEqual(float(block.header.nu_upper), np.linalg.norm(decoded_values[2]))

    def test_serializer_matches_estimated_packed_bytes_and_roundtrips(self) -> None:
        keys = np.array([[1.0, -2.0], [1.25, -2.25], [0.75, -1.75]], dtype=np.float64)
        values = np.array([[0.1, 0.2], [0.15, 0.1], [0.05, 0.25]], dtype=np.float64)
        block = encode_block(keys, values, block_start=7)
        payload = block.serialize()
        self.assertEqual(len(payload), block.estimated_packed_bytes())
        self.assertEqual(len(payload), block.actual_serialized_bytes())
        roundtrip = CompressedBlock.deserialize(payload, key_dim=2, value_dim=2)
        self.assertEqual(roundtrip.header.block_start, 7)
        self.assertEqual(roundtrip.header.block_len, 3)
        self.assertEqual(roundtrip.header.anchor_key.tobytes(order="C"), block.header.anchor_key.tobytes(order="C"))
        self.assertEqual(roundtrip.header.anchor_value.tobytes(order="C"), block.header.anchor_value.tobytes(order="C"))
        self.assertEqual(
            int(np.asarray([roundtrip.header.key_scale], dtype=np.float16).view(np.uint16)[0]),
            int(np.asarray([block.header.key_scale], dtype=np.float16).view(np.uint16)[0]),
        )
        self.assertEqual(
            int(np.asarray([roundtrip.header.value_scale], dtype=np.float16).view(np.uint16)[0]),
            int(np.asarray([block.header.value_scale], dtype=np.float16).view(np.uint16)[0]),
        )
        self.assertEqual(
            int(np.asarray([roundtrip.header.rho_upper], dtype=np.float32).view(np.uint32)[0]),
            int(np.asarray([block.header.rho_upper], dtype=np.float32).view(np.uint32)[0]),
        )
        self.assertEqual(
            int(np.asarray([roundtrip.header.nu_upper], dtype=np.float32).view(np.uint32)[0]),
            int(np.asarray([block.header.nu_upper], dtype=np.float32).view(np.uint32)[0]),
        )
        np.testing.assert_array_equal(roundtrip.key_residuals, block.key_residuals)
        np.testing.assert_array_equal(roundtrip.value_residuals, block.value_residuals)
        self.assertEqual(roundtrip.serialize(), payload)
        recomputed_rho, recomputed_nu = _rigorous_block_metadata(roundtrip, precision=256)
        self.assertGreaterEqual(float(roundtrip.header.rho_upper), float(recomputed_rho))
        self.assertGreaterEqual(float(roundtrip.header.nu_upper), float(recomputed_nu))
        self.assertGreater(roundtrip.actual_python_object_bytes(), roundtrip.actual_serialized_bytes())

    def test_key_and_value_blocks_decode_independently(self) -> None:
        keys = np.array([[1.0, -2.0], [1.25, -2.25], [0.75, -1.75]], dtype=np.float64)
        values = np.array([[0.1, 0.2], [0.15, 0.1], [0.05, 0.25]], dtype=np.float64)
        block = encode_block(keys, values, block_start=7)
        decoded_keys = block.decode_key_block()
        decoded_values = block.decode_value_block()
        combined_keys, combined_values = block.decode_block()
        np.testing.assert_array_equal(decoded_keys, combined_keys)
        np.testing.assert_array_equal(decoded_values, combined_values)
        np.testing.assert_array_equal(block.decode_key_token(1), combined_keys[1])
        np.testing.assert_array_equal(block.decode_value_token(2), combined_values[2])

    def test_serialized_block_container_supports_independent_random_access(self) -> None:
        block0 = encode_block(
            np.array([[1.0, 0.0], [1.2, -0.1]], dtype=np.float64),
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
            block_start=0,
        )
        block1 = encode_block(
            np.array([[2.0, 0.5], [2.1, 0.25], [1.8, 0.0]], dtype=np.float64),
            np.array([[1.0, -0.2], [0.9, -0.1], [0.8, 0.0]], dtype=np.float64),
            block_start=2,
        )
        container = serialize_block_container([block0, block1])
        self.assertEqual(
            container.total_bytes,
            container.header_bytes + container.index_bytes + block0.actual_serialized_bytes() + block1.actual_serialized_bytes(),
        )
        roundtrip_container = deserialize_block_container(container.buffer)
        roundtrip_block1 = roundtrip_container.deserialize_block(1)
        roundtrip_keys, roundtrip_values = roundtrip_block1.decode_block()
        expected_keys, expected_values = block1.decode_block()
        np.testing.assert_array_equal(roundtrip_keys, expected_keys)
        np.testing.assert_array_equal(roundtrip_values, expected_values)

        corrupted = bytearray(container.buffer)
        corrupted[container.block_offsets[0]] ^= 0x01
        corrupted_container = deserialize_block_container(bytes(corrupted))
        unaffected = corrupted_container.deserialize_block(1)
        unaffected_keys, unaffected_values = unaffected.decode_block()
        np.testing.assert_array_equal(unaffected_keys, expected_keys)
        np.testing.assert_array_equal(unaffected_values, expected_values)

    def test_empty_and_malformed_serialized_block_containers_are_rejected(self) -> None:
        block = encode_block(
            np.array([[1.0, 0.0], [1.2, -0.1]], dtype=np.float64),
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
            block_start=0,
        )
        container = serialize_block_container([block])

        with self.assertRaises(ValueError):
            serialize_block_container([])

        with self.assertRaises(ValueError):
            deserialize_block_container(container.buffer[:10])

        malformed_zero_count = bytearray(container.buffer)
        malformed_zero_count[8:12] = (0).to_bytes(4, byteorder="little", signed=False)
        with self.assertRaises(ValueError):
            deserialize_block_container(bytes(malformed_zero_count))

        overlapping = bytearray(container.buffer)
        payload_start = container.header_bytes + container.index_bytes
        overlapping[12:16] = (payload_start - 1).to_bytes(4, byteorder="little", signed=False)
        with self.assertRaises(ValueError):
            deserialize_block_container(bytes(overlapping))

        out_of_range = bytearray(container.buffer)
        out_of_range[16:20] = (len(container.buffer) + 1).to_bytes(4, byteorder="little", signed=False)
        with self.assertRaises(ValueError):
            deserialize_block_container(bytes(out_of_range))

    def test_memory_accounting_includes_recent_exact_and_total(self) -> None:
        self.assertEqual(original_block_bytes(32, 128, 128), 16384)
        self.assertEqual(compressed_block_bytes(32, 128, 128), 8468)
        report = historical_memory_report(
            geometry=LLAMA_3_1_8B_GEOMETRY,
            total_context_tokens=1024,
            recent_window=128,
            block_size=32,
        )
        self.assertEqual(report.historical_tokens, 896)
        self.assertEqual(report.recent_exact_bytes, 128 * LLAMA_3_1_8B_GEOMETRY.original_bytes_per_token_total)
        self.assertEqual(
            report.estimated_packed_total_bytes,
            report.recent_exact_bytes + report.estimated_packed_historical_bytes,
        )
        self.assertGreater(report.original_historical_bytes, report.estimated_packed_historical_bytes)
        self.assertGreater(report.compression_ratio, 1.0)

    def test_invalid_block_header_and_payload_validation(self) -> None:
        with self.assertRaises(ValueError):
            BlockHeader(
                block_start=0,
                block_len=1,
                anchor_key=np.array([1.0], dtype=np.float16),
                anchor_value=np.array([1.0], dtype=np.float16),
                key_scale=np.float16(0.0),
                value_scale=np.float16(1.0),
                rho_upper=np.float32(0.0),
                nu_upper=np.float32(0.0),
            )
        with self.assertRaises(ValueError):
            CompressedBlock(
                header=BlockHeader(
                    block_start=0,
                    block_len=2,
                    anchor_key=np.array([1.0], dtype=np.float16),
                    anchor_value=np.array([1.0], dtype=np.float16),
                    key_scale=np.float16(1.0),
                    value_scale=np.float16(1.0),
                    rho_upper=np.float32(0.0),
                    nu_upper=np.float32(0.0),
                ),
                key_residuals=np.zeros((2, 1), dtype=np.int8),
                value_residuals=np.zeros((1, 1), dtype=np.int8),
            )

    def test_quantization_scales_handle_zero_tiny_and_overflow_cases(self) -> None:
        zero_scale, zero_quantized = _quantize_residuals(np.zeros((3, 2), dtype=np.float64))
        self.assertEqual(float(zero_scale), 1.0)
        np.testing.assert_array_equal(zero_quantized, np.zeros((3, 2), dtype=np.int8))

        tiny_scale, tiny_quantized = _quantize_residuals(np.array([[1e-12, -2e-12]], dtype=np.float64))
        self.assertTrue(np.isfinite(tiny_scale))
        self.assertGreater(float(tiny_scale), 0.0)
        self.assertEqual(tiny_quantized.dtype, np.int8)

        with self.assertRaises(OverflowError):
            _quantize_residuals(np.array([[1e308, -1e308]], dtype=np.float64))

    def test_encode_block_raises_on_residual_scale_overflow(self) -> None:
        keys = np.array([[0.0, 0.0], [1e308, 0.0]], dtype=np.float64)
        values = np.array([[0.0], [1.0]], dtype=np.float64)
        with self.assertRaises(OverflowError):
            encode_block(keys, values, block_start=0)


if __name__ == "__main__":
    unittest.main()
