from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from rack_kv.llama_trace import (
    load_attention_trace,
    query_head_to_kv_head,
    replay_compact_trace_outputs,
    replay_head_attention_output,
    run_minimal_llama31_capture,
    save_attention_trace,
)


class LlamaTraceTests(unittest.TestCase):
    def test_query_head_to_kv_head_respects_gqa_groups(self) -> None:
        self.assertEqual(query_head_to_kv_head(0, num_attention_heads=32, num_key_value_heads=8), 0)
        self.assertEqual(query_head_to_kv_head(3, num_attention_heads=32, num_key_value_heads=8), 0)
        self.assertEqual(query_head_to_kv_head(4, num_attention_heads=32, num_key_value_heads=8), 1)
        self.assertEqual(query_head_to_kv_head(31, num_attention_heads=32, num_key_value_heads=8), 7)

    def test_trace_roundtrip_preserves_tensors_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.safetensors"
            tensors = {
                "queries": torch.tensor([[[1.0, 2.0]]], dtype=torch.bfloat16),
                "final_keys": torch.tensor([[[1.0, 0.0], [0.5, -0.5]]], dtype=torch.bfloat16),
                "final_values": torch.tensor([[[0.25, -0.25], [0.5, 0.5]]], dtype=torch.bfloat16),
                "model_head_outputs": torch.tensor([[[0.3125, 0.0625]]], dtype=torch.bfloat16),
                "visible_lengths": torch.tensor([2], dtype=torch.int64),
                "record_token_ids": torch.tensor([17], dtype=torch.int64),
                "record_token_positions": torch.tensor([1], dtype=torch.int64),
                "record_layer_indices": torch.tensor([3], dtype=torch.int64),
                "selected_query_heads": torch.tensor([0], dtype=torch.int64),
                "selected_kv_heads": torch.tensor([0], dtype=torch.int64),
            }
            metadata = {"prompt": "trace", "query_to_kv_heads": [0], "scaling": 0.5, "trace_schema": "compact_final_cache_v1"}
            save_attention_trace(trace_path, tensors, metadata)
            loaded_tensors, loaded_metadata = load_attention_trace(trace_path)

            for key, value in tensors.items():
                self.assertTrue(torch.equal(value, loaded_tensors[key]))
            self.assertEqual(loaded_metadata["prompt"], "trace")
            self.assertEqual(loaded_metadata["query_to_kv_heads"], [0])
            self.assertEqual(loaded_metadata["scaling"], 0.5)
            self.assertEqual(int(loaded_tensors["record_layer_indices"][0].item()), 3)

    def _assert_selected_head_roundtrip(self, selected_query_heads: tuple[int, ...]) -> None:
        num_attention_heads = 32
        num_key_value_heads = 8
        query_to_kv_heads = tuple(
            query_head_to_kv_head(
                query_head_index=head,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            )
            for head in selected_query_heads
        )
        selected_kv_heads = tuple(sorted(set(query_to_kv_heads)))
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.safetensors"
            tensors = {
                "queries": torch.tensor([[[1.0, 2.0]]], dtype=torch.bfloat16).repeat(1, len(selected_query_heads), 1),
                "final_keys": torch.tensor([[[1.0, 0.0], [0.5, -0.5]]], dtype=torch.bfloat16).repeat(len(selected_kv_heads), 1, 1),
                "final_values": torch.tensor([[[0.25, -0.25], [0.5, 0.5]]], dtype=torch.bfloat16).repeat(len(selected_kv_heads), 1, 1),
                "model_head_outputs": torch.tensor([[[0.3125, 0.0625]]], dtype=torch.bfloat16).repeat(1, len(selected_query_heads), 1),
                "visible_lengths": torch.tensor([2], dtype=torch.int64),
                "record_token_ids": torch.tensor([17], dtype=torch.int64),
                "record_token_positions": torch.tensor([1], dtype=torch.int64),
                "record_layer_indices": torch.tensor([0], dtype=torch.int64),
                "selected_query_heads": torch.tensor(selected_query_heads, dtype=torch.int64),
                "selected_kv_heads": torch.tensor(selected_kv_heads, dtype=torch.int64),
            }
            metadata = {
                "prompt": "trace",
                "query_to_kv_heads": list(query_to_kv_heads),
                "selected_query_heads": list(selected_query_heads),
                "selected_kv_heads": list(selected_kv_heads),
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": num_key_value_heads,
                "scaling": 0.5,
                "trace_schema": "compact_final_cache_v1",
            }
            save_attention_trace(trace_path, tensors, metadata)
            loaded_tensors, loaded_metadata = load_attention_trace(trace_path)
            self.assertEqual(loaded_tensors["selected_query_heads"].tolist(), list(selected_query_heads))
            self.assertEqual(loaded_metadata["selected_query_heads"], list(selected_query_heads))
            self.assertEqual(loaded_tensors["selected_kv_heads"].tolist(), list(selected_kv_heads))
            self.assertEqual(loaded_metadata["selected_kv_heads"], list(selected_kv_heads))
            self.assertEqual(loaded_metadata["query_to_kv_heads"], list(query_to_kv_heads))
            for query_head, kv_head in zip(selected_query_heads, query_to_kv_heads):
                self.assertEqual(
                    query_head // (num_attention_heads // num_key_value_heads),
                    kv_head,
                )
                self.assertIn(kv_head, selected_kv_heads)

    def test_selected_query_heads_roundtrip_preserves_global_indices_014(self) -> None:
        selected_query_heads = (0, 1, 4)
        self._assert_selected_head_roundtrip(selected_query_heads)
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.safetensors"
            tensors = {
                "queries": torch.tensor([[[1.0, 2.0]]], dtype=torch.bfloat16).repeat(1, len(selected_query_heads), 1),
                "final_keys": torch.tensor([[[1.0, 0.0], [0.5, -0.5]]], dtype=torch.bfloat16).repeat(2, 1, 1),
                "final_values": torch.tensor([[[0.25, -0.25], [0.5, 0.5]]], dtype=torch.bfloat16).repeat(2, 1, 1),
                "model_head_outputs": torch.tensor([[[0.3125, 0.0625]]], dtype=torch.bfloat16).repeat(1, len(selected_query_heads), 1),
                "visible_lengths": torch.tensor([2], dtype=torch.int64),
                "record_token_ids": torch.tensor([17], dtype=torch.int64),
                "record_token_positions": torch.tensor([1], dtype=torch.int64),
                "record_layer_indices": torch.tensor([0], dtype=torch.int64),
                "selected_query_heads": torch.tensor(selected_query_heads, dtype=torch.int64),
                "selected_kv_heads": torch.tensor([0, 1], dtype=torch.int64),
            }
            metadata = {
                "prompt": "trace",
                "query_to_kv_heads": [0, 0, 1],
                "selected_query_heads": [0, 1, 4],
                "selected_kv_heads": [0, 1],
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "scaling": 0.5,
                "trace_schema": "compact_final_cache_v1",
            }
            save_attention_trace(trace_path, tensors, metadata)
            loaded_tensors, loaded_metadata = load_attention_trace(trace_path)
            self.assertEqual(loaded_tensors["selected_query_heads"].tolist(), [0, 1, 4])
            self.assertNotEqual(loaded_tensors["selected_query_heads"].tolist(), [0, 1, 2])
            self.assertEqual(loaded_metadata["selected_query_heads"], [0, 1, 4])

    def test_selected_query_heads_roundtrip_preserves_global_indices_3712(self) -> None:
        self._assert_selected_head_roundtrip((3, 7, 12))

    def test_offline_replay_matches_manual_attention(self) -> None:
        query = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)
        keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16)
        values = torch.tensor([[0.5, 0.0], [0.0, 1.0]], dtype=torch.bfloat16)
        replayed = replay_head_attention_output(query, keys, values, scaling=0.5)

        scores = torch.matmul(query.unsqueeze(0), keys.transpose(0, 1)) * 0.5
        weights = torch.softmax(scores.to(torch.float32), dim=-1).to(torch.bfloat16)
        manual = torch.matmul(weights, values).squeeze(0)
        self.assertTrue(torch.equal(replayed, manual))

    def test_compact_trace_replay_uses_final_cache_prefixes(self) -> None:
        tensors = {
            "queries": torch.tensor(
                [
                    [[1.0, -1.0]],
                    [[0.5, 0.25]],
                ],
                dtype=torch.bfloat16,
            ),
            "final_keys": torch.tensor(
                [
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ]
                ],
                dtype=torch.bfloat16,
            ),
            "final_values": torch.tensor(
                [
                    [
                        [0.5, 0.0],
                        [0.0, 1.0],
                    ]
                ],
                dtype=torch.bfloat16,
            ),
            "visible_lengths": torch.tensor([1, 2], dtype=torch.int64),
        }
        replayed = replay_compact_trace_outputs(
            tensors,
            query_to_kv_heads=(0,),
            selected_kv_heads=(0,),
            scaling=0.5,
        )
        expected0 = replay_head_attention_output(tensors["queries"][0, 0], tensors["final_keys"][0, :1, :], tensors["final_values"][0, :1, :], scaling=0.5)
        expected1 = replay_head_attention_output(tensors["queries"][1, 0], tensors["final_keys"][0, :2, :], tensors["final_values"][0, :2, :], scaling=0.5)
        self.assertTrue(torch.equal(replayed[0, 0], expected0))
        self.assertTrue(torch.equal(replayed[1, 0], expected1))

    def test_layer_one_capture_is_rejected_before_network_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                run_minimal_llama31_capture(output_dir=Path(temp_dir), layer_index=1)


if __name__ == "__main__":
    unittest.main()
