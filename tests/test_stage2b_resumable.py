from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import psutil
import torch

from rack_kv.llama_trace import replay_compact_trace_outputs, save_attention_trace
from rack_kv.stage2 import validate_compact_trace
from rack_kv.stage2b import result_to_dict, run_stage2b_pilot


def _load_stage2b_script_module():
    script_path = Path("scripts") / "run_stage2b_layer0_pilot.py"
    spec = importlib.util.spec_from_file_location("stage2b_runner_resumable", script_path)
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


class Stage2BResumableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_stage2b_script_module()
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls._temp_dir.name)
        cls.trace_path = cls.temp_path / "synthetic_trace_256.safetensors"
        _build_synthetic_trace(cls.trace_path, scaling=0.5)
        cls.trace = validate_compact_trace(cls.trace_path)
        cls.tolerances = (0.0, 0.01, 0.05, 0.1)
        cls.prompt_sha = "synthetic-prompt-sha"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()

    def test_case_equivalence_no_historical_blocks(self) -> None:
        case_record, prefix_record, _ = self.module._run_case_with_equivalence(
            trace=self.trace,
            record_index=0,
            query_local_index=0,
            recent_window=16,
            block_size=8,
            tolerances=self.tolerances,
            precision=64,
        )
        self.assertTrue(case_record["equivalence"]["exact_match"])
        self.assertEqual(prefix_record["historical_tokens"], 0)
        for tolerance_key in case_record["results_by_tolerance"]:
            result = case_record["results_by_tolerance"][tolerance_key]
            self.assertEqual(result["candidate_blocks"], 0)
            self.assertEqual(result["certified_skipped_blocks"], 0)

    def test_case_equivalence_with_history_and_shared_kv_mapping(self) -> None:
        first_case, _, _ = self.module._run_case_with_equivalence(
            trace=self.trace,
            record_index=255,
            query_local_index=0,
            recent_window=16,
            block_size=8,
            tolerances=self.tolerances,
            precision=64,
        )
        second_case, _, _ = self.module._run_case_with_equivalence(
            trace=self.trace,
            record_index=255,
            query_local_index=1,
            recent_window=16,
            block_size=8,
            tolerances=self.tolerances,
            precision=64,
        )
        self.assertTrue(first_case["equivalence"]["exact_match"])
        self.assertTrue(second_case["equivalence"]["exact_match"])
        self.assertEqual(first_case["kv_head_global"], 0)
        self.assertEqual(second_case["kv_head_global"], 0)
        self.assertGreater(
            max(
                result["candidate_blocks"]
                for result in first_case["results_by_tolerance"].values()
            ),
            0,
        )

    def test_case_equivalence_supports_mocked_numerical_fallback(self) -> None:
        fallback_projection = {
            "record_index": 31,
            "query_local_index": 0,
            "query_head_global": 0,
            "kv_head_global": 0,
            "visible_length": 32,
            "query_position": 31,
            "historical_tokens": 16,
            "recent_exact_tokens": 16,
            "candidate_blocks": 2,
            "certified_skipped_blocks": 0,
            "decoded_blocks": 2,
            "decoded_block_starts": (0, 8),
            "skipped_block_starts": (),
            "certificate_name": "zero",
            "z_k_lower_text": None,
            "z_k_lower_upper_float": None,
            "u_s_upper_text": None,
            "u_s_upper_float": None,
            "nu_s_upper_text": None,
            "nu_s_upper_float": None,
            "kept_output_norm_upper_text": None,
            "kept_output_norm_upper_float": None,
            "certificate_bound_text": "0.0",
            "certificate_bound_upper_float": 0.0,
            "observed_skip_error": 0.0,
            "rigorous_skip_error_upper_text": "0.0",
            "rigorous_skip_error_upper_float": 0.0,
            "bound_to_observed_ratio": None,
            "approximate_observed_violation": False,
            "rigorous_interval_violation": False,
            "numerical_fallback_used": True,
            "model_reference_gap": 0.0,
            "compression_error": 0.0,
            "reference_total_error": 0.0,
            "captured_model_total_gap": 0.0,
            "reference_decomposition_lhs": 0.0,
            "reference_decomposition_rhs": 0.0,
            "model_relative_decomposition_lhs": 0.0,
            "model_relative_decomposition_rhs": 0.0,
        }

        def _fake_case(*args, **kwargs):
            return SimpleNamespace(**fallback_projection.copy())

        with patch.object(self.module, "_run_query_case_tolerance_bundle", side_effect=lambda **kwargs: tuple(_fake_case() for _ in kwargs["tolerances"])):
            with patch.object(self.module, "_run_query_case", side_effect=lambda **kwargs: _fake_case()):
                case_record, _, _ = self.module._run_case_with_equivalence(
                    trace=self.trace,
                    record_index=31,
                    query_local_index=0,
                    recent_window=16,
                    block_size=8,
                    tolerances=self.tolerances,
                    precision=64,
                )
        self.assertTrue(case_record["equivalence"]["exact_match"])
        self.assertTrue(all(result["numerical_fallback_used"] for result in case_record["results_by_tolerance"].values()))

    def test_partial_tmp_checkpoint_is_ignored(self) -> None:
        checkpoint_path = self.temp_path / "pair_checkpoints" / "pair_w16_b8.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.with_name(checkpoint_path.name + ".tmp").write_text("partial", encoding="utf-8")
        settings = self.module._pair_checkpoint_settings(
            trace=self.trace,
            prompt_source_sha256=self.prompt_sha,
            recent_window=16,
            block_size=8,
            tolerances=self.tolerances,
            precision=64,
            random_seed=0,
            query_positions=(31, 255),
        )
        payload = self.module._load_pair_checkpoint(
            checkpoint_path=checkpoint_path,
            settings=settings,
            tolerances=self.tolerances,
        )
        self.assertIsNone(payload)

    def test_checkpoint_with_wrong_trace_hash_is_rejected(self) -> None:
        checkpoint_path = self.temp_path / "pair_checkpoints" / "pair_w16_b16_wrong.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        settings = self.module._pair_checkpoint_settings(
            trace=self.trace,
            prompt_source_sha256=self.prompt_sha,
            recent_window=16,
            block_size=16,
            tolerances=self.tolerances,
            precision=64,
            random_seed=0,
            query_positions=(31, 255),
        )
        payload = self.module._new_pair_checkpoint_payload(
            settings=settings,
            case_order=self.module._pair_case_order(
                trace=self.trace,
                recent_window=16,
                block_size=16,
                query_positions=(31, 255),
            ),
        )
        payload["trace_sha256"] = "wrong"
        checkpoint_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        with self.assertRaises(self.module.Stage2BExecutionError):
            self.module._load_pair_checkpoint(
                checkpoint_path=checkpoint_path,
                settings=settings,
                tolerances=self.tolerances,
            )

    def test_interrupted_run_resumes_without_recomputing_completed_cases(self) -> None:
        output_dir = self.temp_path / "resume_case_test"
        checkpoint_path = output_dir / "pair_checkpoints" / "pair_w16_b32.json"
        first = self.module._run_pair_checkpoint_internal(
            trace_path=self.trace_path,
            checkpoint_path=checkpoint_path,
            prompt_source_sha256=self.prompt_sha,
            recent_window=16,
            block_size=32,
            tolerances=self.tolerances,
            precision=64,
            random_seed=0,
            max_rss_bytes=16 * 1024 ** 3,
            min_free_memory_bytes=256 * 1024 ** 2,
            query_positions=(31, 255),
            stop_after_completed_cases=1,
        )
        self.assertEqual(first["status"], "stopped")
        self.assertEqual(first["payload"]["metrics"]["cases_completed"], 1)

        second = self.module._resume_or_run_pair_checkpoint(
            output_dir=output_dir,
            trace_path=self.trace_path,
            prompt_source_sha256=self.prompt_sha,
            recent_window=16,
            block_size=32,
            tolerances=self.tolerances,
            precision=64,
            random_seed=0,
            max_rss_bytes=16 * 1024 ** 3,
            min_free_memory_bytes=256 * 1024 ** 2,
            query_positions=(31, 255),
        )
        payload = second["payload"]
        self.assertEqual(payload["status"], "completed")
        self.assertGreaterEqual(payload["metrics"]["cases_reused_from_checkpoint"], 1)
        self.assertEqual(
            payload["metrics"]["cases_newly_computed"] + payload["metrics"]["cases_reused_from_checkpoint"],
            payload["metrics"]["cases_total"],
        )

    def test_memory_guard_stops_before_next_case_and_preserves_checkpoint(self) -> None:
        output_dir = self.temp_path / "memory_guard_test"
        checkpoint_path = output_dir / "pair_checkpoints" / "pair_w16_b8.json"
        rss_values = [
            100 * 1024 ** 2,
            100 * 1024 ** 2,
            100 * 1024 ** 2,
            8_450_000_000,
            8_450_000_000,
            8_450_000_000,
        ]
        rss_state = {"index": 0}

        def _next_rss() -> int:
            index = rss_state["index"]
            if index < len(rss_values):
                value = rss_values[index]
                rss_state["index"] += 1
                return value
            return rss_values[-1]

        def _fake_run_case(**kwargs):
            query_local_index = int(kwargs["query_local_index"])
            record_index = int(kwargs["record_index"])
            kv_head_global = int(self.trace.query_to_kv_heads[query_local_index])
            query_head_global = int(self.trace.selected_query_heads[query_local_index])
            prefix_record = {
                "record_index": record_index,
                "kv_head_global": kv_head_global,
                "visible_length": 32,
                "recent_window": 16,
                "block_size": 8,
                "historical_tokens": 16,
                "recent_exact_tokens": 16,
                "original_kv_bytes": 256,
                "recent_exact_bytes": 128,
                "compressed_historical_bytes": 64,
                "total_compressed_bytes": 192,
                "bytes_per_historical_token": 4.0,
                "compression_ratio": 256.0 / 192.0,
                "key_anchor_bytes": 16,
                "value_anchor_bytes": 16,
                "quantized_key_residual_bytes": 8,
                "quantized_value_residual_bytes": 8,
                "scale_bytes": 4,
                "certificate_metadata_bytes": 8,
                "block_header_bytes": 8,
                "container_header_bytes": 8,
                "container_index_bytes": 8,
                "padding_alignment_bytes": 0,
                "independent_blocks_tested": 1,
                "independent_decode_max_diff": 0.0,
                "unaffected_block_decode_tested": True,
                "unaffected_block_decode_passed": True,
                "max_key_abs_error": 0.0,
                "key_rmse": 0.0,
                "key_squared_error_sum": 0.0,
                "key_element_count": 1,
                "max_value_abs_error": 0.0,
                "value_rmse": 0.0,
                "value_squared_error_sum": 0.0,
                "value_element_count": 1,
                "number_of_blocks": 1,
                "authoritative_serialized_roundtrip_used": True,
            }
            result_projection = {
                "record_index": record_index,
                "query_local_index": query_local_index,
                "query_head_global": query_head_global,
                "kv_head_global": kv_head_global,
                "visible_length": 32,
                "query_position": record_index,
                "historical_tokens": 16,
                "recent_exact_tokens": 16,
                "candidate_blocks": 1,
                "certified_skipped_blocks": 0,
                "decoded_blocks": 1,
                "decoded_block_starts": (0,),
                "skipped_block_starts": (),
                "certificate_name": "zero",
                "z_k_lower_text": None,
                "z_k_lower_upper_float": None,
                "u_s_upper_text": None,
                "u_s_upper_float": None,
                "nu_s_upper_text": None,
                "nu_s_upper_float": None,
                "kept_output_norm_upper_text": None,
                "kept_output_norm_upper_float": None,
                "certificate_bound_text": "0.0",
                "certificate_bound_upper_float": 0.0,
                "observed_skip_error": 0.0,
                "rigorous_skip_error_upper_text": "0.0",
                "rigorous_skip_error_upper_float": 0.0,
                "bound_to_observed_ratio": None,
                "approximate_observed_violation": False,
                "rigorous_interval_violation": False,
                "numerical_fallback_used": False,
                "model_reference_gap": 0.0,
                "compression_error": 0.0,
                "reference_total_error": 0.0,
                "captured_model_total_gap": 0.0,
                "reference_decomposition_lhs": 0.0,
                "reference_decomposition_rhs": 0.0,
                "model_relative_decomposition_lhs": 0.0,
                "model_relative_decomposition_rhs": 0.0,
            }
            case_record = {
                "status": "completed",
                "record_index": record_index,
                "query_local_index": query_local_index,
                "query_head_global": query_head_global,
                "kv_head_global": kv_head_global,
                "query_position": record_index,
                "results_by_tolerance": {
                    self.module._tolerance_key(tolerance): result_projection
                    for tolerance in self.tolerances
                },
                "equivalence": {"exact_match": True, "first_difference": None},
                "metrics": {"runtime_seconds": 0.0},
            }
            return case_record, prefix_record, case_record["metrics"]

        with patch.object(self.module, "_run_case_with_equivalence", side_effect=_fake_run_case):
            with patch.object(self.module, "_rss_bytes", side_effect=_next_rss):
                with patch.object(self.module, "_available_memory_bytes", return_value=8 * 1024 ** 3):
                    with self.assertRaises(self.module.MemoryGuardExceeded):
                        self.module._run_pair_checkpoint_internal(
                            trace_path=self.trace_path,
                            checkpoint_path=checkpoint_path,
                            prompt_source_sha256=self.prompt_sha,
                            recent_window=16,
                            block_size=8,
                            tolerances=self.tolerances,
                            precision=64,
                            random_seed=0,
                            max_rss_bytes=8 * 1024 ** 3,
                            min_free_memory_bytes=256 * 1024 ** 2,
                            query_positions=(31, 255),
                        )
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "stopped")
        self.assertEqual(payload["metrics"]["cases_completed"], 1)

    def test_stale_lock_removed_but_live_lock_respected(self) -> None:
        lock_path = self.temp_path / "stage2b_resumable.lock.json"
        lock_path.write_text(
            json.dumps(
                {
                    "pid": 999999,
                    "process_create_time": 0.0,
                    "command": "python stale.py",
                    "hostname": "stale-host",
                    "start_timestamp": "2026-07-21T00:00:00Z",
                }
            ),
            encoding="utf-8",
        )
        lock_payload = self.module._acquire_resume_lock(lock_path)
        self.assertEqual(int(json.loads(lock_path.read_text(encoding="utf-8"))["pid"]), lock_payload["pid"])
        self.module._release_resume_lock(lock_path, lock_payload)
        self.assertFalse(lock_path.exists())

        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            sleeper_lock = {
                "pid": sleeper.pid,
                "process_create_time": psutil.Process(sleeper.pid).create_time(),
                "command": "python sleep",
                "hostname": "live-host",
                "start_timestamp": "2026-07-21T00:00:01Z",
            }
            lock_path.write_text(json.dumps(sleeper_lock), encoding="utf-8")
            with self.assertRaises(self.module.Stage2BExecutionError):
                self.module._acquire_resume_lock(lock_path)
        finally:
            sleeper.terminate()
            sleeper.wait(timeout=10)
            if lock_path.exists():
                lock_path.unlink()

    def test_aggregated_checkpoint_matches_in_memory_reference(self) -> None:
        output_dir = self.temp_path / "aggregate_match_test"
        checkpoint_path = output_dir / "pair_checkpoints" / "pair_w16_b32.json"
        summary = self.module._run_pair_checkpoint_internal(
            trace_path=self.trace_path,
            checkpoint_path=checkpoint_path,
            prompt_source_sha256=self.prompt_sha,
            recent_window=16,
            block_size=32,
            tolerances=self.tolerances,
            precision=64,
            random_seed=0,
            max_rss_bytes=16 * 1024 ** 3,
            min_free_memory_bytes=256 * 1024 ** 2,
            query_positions=(31, 255),
        )
        payload = summary["payload"]
        streamed = payload["aggregated_results"]
        reference = result_to_dict(
            run_stage2b_pilot(
                trace_path=self.trace_path,
                recent_windows=(16,),
                block_sizes=(32,),
                tolerances=self.tolerances,
                precision=64,
                random_seed=0,
                evaluated_query_positions=(31, 255),
            )
        )
        self.assertEqual(streamed["memory_by_wb"], reference["memory_by_wb"])
        self.assertEqual(streamed["configurations"], reference["configurations"])


if __name__ == "__main__":
    unittest.main()
