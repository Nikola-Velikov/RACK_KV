from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from rack_kv.llama_trace import replay_compact_trace_outputs, save_attention_trace
from rack_kv.stage2 import validate_compact_trace
from rack_kv.stage3 import result_to_dict, run_stage3_reduced_layer_case


def _build_synthetic_trace(trace_path: Path, *, layer_index: int, scaling: float) -> None:
    query_records = 8
    head_dim = 2
    selected_query_heads = (0, 1, 4)
    selected_kv_heads = (0, 1)
    query_to_kv_heads = (0, 0, 1)
    queries = torch.zeros((query_records, len(selected_query_heads), head_dim), dtype=torch.bfloat16)
    final_keys = torch.zeros((len(selected_kv_heads), query_records, head_dim), dtype=torch.bfloat16)
    final_values = torch.zeros((len(selected_kv_heads), query_records, head_dim), dtype=torch.bfloat16)
    for record_index in range(query_records):
        for query_local_index in range(len(selected_query_heads)):
            queries[record_index, query_local_index, 0] = torch.tensor(0.02 * (query_local_index + 1), dtype=torch.bfloat16)
            queries[record_index, query_local_index, 1] = torch.tensor(-0.01 * record_index, dtype=torch.bfloat16)
        for kv_local_index in range(len(selected_kv_heads)):
            final_keys[kv_local_index, record_index, 0] = torch.tensor(0.05 * (record_index + 1), dtype=torch.bfloat16)
            final_keys[kv_local_index, record_index, 1] = torch.tensor(-0.015 * (kv_local_index + 1), dtype=torch.bfloat16)
            final_values[kv_local_index, record_index, 0] = torch.tensor(0.03 * ((record_index % 3) - 1), dtype=torch.bfloat16)
            final_values[kv_local_index, record_index, 1] = torch.tensor(0.02 * (kv_local_index + 1), dtype=torch.bfloat16)

    tensors = {
        "queries": queries,
        "final_keys": final_keys,
        "final_values": final_values,
        "visible_lengths": torch.arange(1, query_records + 1, dtype=torch.int64),
        "record_token_ids": torch.arange(query_records, dtype=torch.int64),
        "record_token_positions": torch.arange(query_records, dtype=torch.int64),
        "record_layer_indices": torch.full((query_records,), layer_index, dtype=torch.int64),
        "selected_query_heads": torch.tensor(selected_query_heads, dtype=torch.int64),
        "selected_kv_heads": torch.tensor(selected_kv_heads, dtype=torch.int64),
    }
    tensors["model_head_outputs"] = replay_compact_trace_outputs(
        tensors,
        query_to_kv_heads=query_to_kv_heads,
        selected_kv_heads=selected_kv_heads,
        scaling=scaling,
    ).to(torch.bfloat16)
    metadata = {
        "trace_schema": "compact_final_cache_v1",
        "checkpoint_repo": "synthetic/stage3-test",
        "checkpoint_revision": "synthetic-revision",
        "selected_query_heads": list(selected_query_heads),
        "selected_kv_heads": list(selected_kv_heads),
        "query_to_kv_heads": list(query_to_kv_heads),
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "gqa_group_size": 4,
        "head_dim": head_dim,
        "layer_index": layer_index,
        "scaling": scaling,
        "source_dtype": "torch.bfloat16",
        "storage_dtype": "torch.bfloat16",
    }
    save_attention_trace(trace_path, tensors, metadata)


class Stage3MultilayerTests(unittest.TestCase):
    def test_nonzero_layer_trace_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "layer31_trace.safetensors"
            _build_synthetic_trace(trace_path, layer_index=31, scaling=0.5)
            with self.assertRaises(ValueError):
                validate_compact_trace(trace_path)
            trace = validate_compact_trace(trace_path, allow_nonzero_layer=True)
            self.assertEqual(trace.layer_index, 31)

    def test_reduced_layer_case_runs_on_nonzero_layer_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "layer31_trace.safetensors"
            _build_synthetic_trace(trace_path, layer_index=31, scaling=0.5)
            result = run_stage3_reduced_layer_case(
                trace_path=trace_path,
                recent_window=2,
                block_size=2,
                tolerances=(0.0, 0.1),
                precision=64,
                record_index=7,
                query_local_index=0,
            )
            payload = result_to_dict(result)
            self.assertEqual(result.layer_index, 31)
            self.assertEqual(payload["layer_index"], 31)
            self.assertEqual(payload["record_index"], 7)
            self.assertEqual(payload["query_local_index"], 0)
            self.assertEqual(len(payload["results"]), 2)


if __name__ == "__main__":
    unittest.main()
