from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from rack_kv.llama_trace import replay_compact_trace_outputs, save_attention_trace
from rack_kv.stage2 import validate_compact_trace
from rack_kv.stage2b import result_to_dict, run_stage2b_pilot, select_stage2b_query_positions


def _load_stage2b_script_module():
    script_path = Path("scripts") / "run_stage2b_layer0_pilot.py"
    spec = importlib.util.spec_from_file_location("stage2b_runner", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_synthetic_trace(trace_path: Path, *, scaling: float) -> None:
    query_records = 256
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
            queries[record_index, query_local_index, 1] = torch.tensor(-0.015 * ((record_index % 5) - 2), dtype=torch.bfloat16)
        for kv_local_index in range(len(selected_kv_heads)):
            final_keys[kv_local_index, record_index, 0] = torch.tensor(0.01 * (record_index % 7) + 0.001 * kv_local_index, dtype=torch.bfloat16)
            final_keys[kv_local_index, record_index, 1] = torch.tensor(-0.02 * ((record_index + kv_local_index) % 5), dtype=torch.bfloat16)
            final_values[kv_local_index, record_index, 0] = torch.tensor(0.03 * ((record_index % 3) - 1), dtype=torch.bfloat16)
            final_values[kv_local_index, record_index, 1] = torch.tensor(0.01 * (kv_local_index + 1) + 0.005 * (record_index % 4), dtype=torch.bfloat16)

    tensors = {
        "queries": queries,
        "final_keys": final_keys,
        "final_values": final_values,
        "visible_lengths": torch.arange(1, query_records + 1, dtype=torch.int64),
        "record_token_ids": torch.arange(query_records, dtype=torch.int64),
        "record_token_positions": torch.arange(query_records, dtype=torch.int64),
        "record_layer_indices": torch.zeros((query_records,), dtype=torch.int64),
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
        "checkpoint_repo": "synthetic/stage2b-test",
        "checkpoint_revision": "synthetic-revision",
        "selected_query_heads": list(selected_query_heads),
        "selected_kv_heads": list(selected_kv_heads),
        "query_to_kv_heads": list(query_to_kv_heads),
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "gqa_group_size": 4,
        "head_dim": head_dim,
        "layer_index": 0,
        "scaling": scaling,
        "source_dtype": "torch.bfloat16",
        "storage_dtype": "torch.bfloat16",
    }
    save_attention_trace(trace_path, tensors, metadata)


class Stage2BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls._temp_dir.name)
        cls.trace_path = cls.temp_path / "synthetic_trace_256.safetensors"
        _build_synthetic_trace(cls.trace_path, scaling=0.5)
        cls.validated = validate_compact_trace(cls.trace_path)
        cls.result = run_stage2b_pilot(
            trace_path=cls.trace_path,
            recent_windows=(16,),
            block_sizes=(32,),
            tolerances=(0.1,),
            precision=64,
            random_seed=0,
        )
        cls.result_dict = result_to_dict(cls.result)
        cls.scaled_trace_path = cls.temp_path / "synthetic_trace_scaled_256.safetensors"
        _build_synthetic_trace(cls.scaled_trace_path, scaling=1.0)
        cls.scaled_result = run_stage2b_pilot(
            trace_path=cls.scaled_trace_path,
            recent_windows=(16,),
            block_sizes=(32,),
            tolerances=(0.1,),
            precision=64,
            random_seed=0,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()

    def test_exact_query_position_sampling_rule(self) -> None:
        self.assertEqual(
            select_stage2b_query_positions(256),
            (31, 47, 63, 79, 95, 111, 127, 143, 159, 175, 191, 207, 223, 239, 248, 249, 250, 251, 252, 253, 254, 255),
        )

    def test_synthetic_256_trace_is_compact_ot_and_loadable(self) -> None:
        trace = self.validated
        self.assertEqual(trace.sequence_length, 256)
        self.assertEqual(trace.query_records, 256)
        self.assertEqual(tuple(trace.queries.shape), (256, 3, 2))
        self.assertEqual(tuple(trace.final_keys.shape), (2, 256, 2))
        self.assertEqual(tuple(trace.final_values.shape), (2, 256, 2))
        self.assertEqual(tuple(trace.model_head_outputs.shape), (256, 3, 2))
        self.assertEqual(trace.selected_query_heads, (0, 1, 4))
        self.assertEqual(trace.selected_kv_heads, (0, 1))
        self.assertEqual(trace.query_to_kv_heads, (0, 0, 1))
        self.assertEqual(trace.visible_lengths[0], 1)
        self.assertEqual(trace.visible_lengths[-1], 256)

    def test_stage2b_result_has_one_memory_row_per_unique_wb_and_no_old_metric(self) -> None:
        result_dict = self.result_dict
        self.assertEqual(len(result_dict["memory_by_wb"]), 1)
        self.assertEqual(len(result_dict["configurations"]), 1)
        memory_row = result_dict["memory_by_wb"][0]
        self.assertEqual(memory_row["recent_window"], 16)
        self.assertEqual(memory_row["block_size"], 32)
        config = result_dict["configurations"][0]
        self.assertIn("max_true_compression_error", config)
        self.assertNotIn("max_compression_error", config)
        self.assertNotIn("serialized_compressed_bytes", config)
        self.assertEqual(config["rigorous_interval_violation_count"], 0)
        self.assertEqual(config["reference_decomposition_violation_count"], 0)
        self.assertEqual(config["model_relative_decomposition_violation_count"], 0)

    def test_saved_scaling_changes_stage2b_reference_metrics(self) -> None:
        base_config = self.result_dict["configurations"][0]
        scaled_config = result_to_dict(self.scaled_result)["configurations"][0]
        self.assertNotEqual(base_config["max_model_reference_gap"], scaled_config["max_model_reference_gap"])

    def test_stage2b_report_uses_unique_memory_rows_and_no_definition_change_section(self) -> None:
        module = _load_stage2b_script_module()
        fake_results = {
            "prompt": {
                "prompt_source_sha256": "abc",
                "prompt_token_count": 256,
                "seed": 0,
                "prompt_source_path": "prompt.txt",
                "prompt_decoded_path": "decoded.txt",
                "prompt_metadata_path": "metadata.json",
            },
            "capture": {
                "checkpoint_repo": "repo",
                "checkpoint_revision": "rev",
                "tls_verification": True,
                "tls_fallback_reason": None,
                "stock_projected_output_max_abs_diff": 0.0,
                "stock_projected_output_max_rel_diff": 0.0,
                "stock_decoder_output_max_abs_diff": 0.0,
                "stock_decoder_output_max_rel_diff": 0.0,
                "stock_cache_key_max_abs_diff": 0.0,
                "stock_cache_key_max_rel_diff": 0.0,
                "stock_cache_value_max_abs_diff": 0.0,
                "stock_cache_value_max_rel_diff": 0.0,
                "max_compact_replay_abs_diff": 0.0,
            },
            "trace": {
                "trace_sha256": "sha",
                "trace_path": "trace.safetensors",
                "trace_size_bytes": 123,
                "queries_shape": [256, 3, 128],
                "final_keys_shape": [2, 256, 128],
                "final_values_shape": [2, 256, 128],
                "model_head_outputs_shape": [256, 3, 128],
                "selected_query_heads": [0, 1, 4],
                "selected_kv_heads": [0, 1],
                "query_to_kv_heads": [0, 0, 1],
                "evaluated_query_positions": [31, 47],
                "scaling": 0.08838834764831845,
                "canonical_one_over_sqrt_head_dim": 0.08838834764831845,
                "scaling_minus_canonical": 0.0,
            },
            "memory_by_wb": [
                {
                    "recent_window": 16,
                    "block_size": 32,
                    "aggregate_prefix_compression_ratio": 1.25,
                    "min_prefix_compression_ratio": 0.95,
                    "mean_prefix_compression_ratio": 1.1,
                    "median_prefix_compression_ratio": 1.15,
                    "max_prefix_compression_ratio": 1.4,
                    "final_prefix_compression_ratio": 1.3,
                    "bytes_per_historical_token": 5.0,
                }
            ],
            "configurations": [
                {
                    "recent_window": 16,
                    "block_size": 32,
                    "tolerance": 0.05,
                    "eligible_query_cases": 10,
                    "weighted_skipped_block_fraction": 0.2,
                    "mean_case_skipped_fraction": 0.1,
                    "median_case_skipped_fraction": 0.1,
                    "max_case_skipped_fraction": 0.3,
                    "fraction_cases_with_any_skipped_block": 0.5,
                    "max_true_compression_error": 0.01,
                    "max_observed_reconstructed_skipping_error": 0.02,
                    "max_rigorous_skip_error_upper": 0.03,
                    "rigorous_interval_violation_count": 0,
                    "fallback_count": 0,
                },
                {
                    "recent_window": 16,
                    "block_size": 32,
                    "tolerance": 0.1,
                    "eligible_query_cases": 10,
                    "weighted_skipped_block_fraction": 0.4,
                    "mean_case_skipped_fraction": 0.2,
                    "median_case_skipped_fraction": 0.2,
                    "max_case_skipped_fraction": 0.5,
                    "fraction_cases_with_any_skipped_block": 0.7,
                    "max_true_compression_error": 0.01,
                    "max_observed_reconstructed_skipping_error": 0.02,
                    "max_rigorous_skip_error_upper": 0.03,
                    "rigorous_interval_violation_count": 0,
                    "fallback_count": 0,
                },
            ],
            "stage2a_comparison": {
                "scope_note": "comparison",
                "stage2a_accepted": {
                    "trace_tokens": 12,
                    "final_prefix_compression_ratio_min": 1.0,
                    "final_prefix_compression_ratio_max": 1.2,
                    "global_prefix_ratio_summary": {},
                    "mean_skipped_block_fraction": 0.0,
                    "max_skipped_block_fraction": 0.0,
                    "fraction_cases_with_any_skipped_block": 0.0,
                    "max_true_compression_error": 0.0,
                    "max_observed_skipping_error": 0.0,
                    "rigorous_interval_violation_count": 0,
                    "runtime_seconds_rerun_current_code": 1.0,
                    "peak_rss_bytes_rerun_current_code": 2,
                },
                "stage2b_pilot": {
                    "trace_tokens": 256,
                    "final_prefix_compression_ratio_min": 1.2,
                    "final_prefix_compression_ratio_max": 1.3,
                    "global_prefix_ratio_summary": {},
                    "mean_skipped_block_fraction": 0.3,
                    "max_skipped_block_fraction": 0.4,
                    "fraction_cases_with_any_skipped_block": 0.5,
                    "max_true_compression_error": 0.01,
                    "max_observed_skipping_error": 0.02,
                    "rigorous_interval_violation_count": 0,
                    "runtime_seconds": 3.0,
                    "peak_rss_bytes": 4,
                },
            },
            "capture_runtime_seconds": 1.0,
            "capture_peak_rss_bytes": 2,
            "experiment_runtime_seconds": 3.0,
            "experiment_peak_rss_bytes": 4,
            "review_package_size_bytes": 5,
            "source_snapshot_sha256": "snap",
        }
        report = module._render_report(fake_results)
        self.assertNotIn("Definition Change From Previous Report", report)
        self.assertIn("0.9500000000000000", report)
        self.assertEqual(report.count("| 16 | 32 | 1.250000 |"), 1)
        self.assertIn("| 16 | 32 | 0.050000 |", report)
        self.assertIn("| 16 | 32 | 0.100000 |", report)


if __name__ == "__main__":
    unittest.main()
