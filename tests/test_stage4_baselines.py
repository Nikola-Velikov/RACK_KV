from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

from rack_kv.llama_trace import replay_compact_trace_outputs, save_attention_trace
from rack_kv.stage4 import (
    BASELINE_DEFINITIONS,
    STAGE4_BUDGET_MATCHING_POLICY_VERSION,
    STAGE4_METHOD_VERSION,
    STAGE4_RACK_TOLERANCE,
    STAGE4_RESULT_VERSION,
    build_stage4_case_context,
    run_baseline_case,
    validate_stage4_capture_inputs,
)


def _load_stage4_script_module():
    script_path = Path("scripts") / "run_stage4_baselines_smoke.py"
    spec = importlib.util.spec_from_file_location("stage4_runner_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_synthetic_trace(
    trace_path: Path,
    *,
    layer_index: int,
    sequence_length: int = 16,
    scaling: float = 0.5,
    future_offset: float = 0.0,
) -> str:
    head_dim = 4
    selected_query_heads = (0, 1, 4)
    selected_kv_heads = (0, 1)
    query_to_kv_heads = (0, 0, 1)
    queries = torch.zeros((sequence_length, len(selected_query_heads), head_dim), dtype=torch.bfloat16)
    final_keys = torch.zeros((len(selected_kv_heads), sequence_length, head_dim), dtype=torch.bfloat16)
    final_values = torch.zeros((len(selected_kv_heads), sequence_length, head_dim), dtype=torch.bfloat16)
    for record_index in range(sequence_length):
        for query_local_index in range(len(selected_query_heads)):
            queries[record_index, query_local_index, 0] = torch.tensor(0.05 * (query_local_index + 1), dtype=torch.bfloat16)
            queries[record_index, query_local_index, 1] = torch.tensor(-0.01 * ((record_index % 5) - 2), dtype=torch.bfloat16)
            queries[record_index, query_local_index, 2] = torch.tensor(0.015 * ((record_index + query_local_index) % 4), dtype=torch.bfloat16)
            queries[record_index, query_local_index, 3] = torch.tensor(-0.02 * (query_local_index + 1), dtype=torch.bfloat16)
        for kv_local_index in range(len(selected_kv_heads)):
            base = 0.01 * record_index + 0.002 * kv_local_index
            if record_index >= sequence_length // 2:
                base += future_offset
            final_keys[kv_local_index, record_index, 0] = torch.tensor(base, dtype=torch.bfloat16)
            final_keys[kv_local_index, record_index, 1] = torch.tensor(-0.03 * ((record_index + kv_local_index) % 5), dtype=torch.bfloat16)
            final_keys[kv_local_index, record_index, 2] = torch.tensor(0.02 * ((record_index % 3) - 1), dtype=torch.bfloat16)
            final_keys[kv_local_index, record_index, 3] = torch.tensor(0.01 * (kv_local_index + 1), dtype=torch.bfloat16)
            final_values[kv_local_index, record_index, 0] = torch.tensor(0.04 * ((record_index % 3) - 1), dtype=torch.bfloat16)
            final_values[kv_local_index, record_index, 1] = torch.tensor(0.02 * (kv_local_index + 1), dtype=torch.bfloat16)
            final_values[kv_local_index, record_index, 2] = torch.tensor(-0.015 * ((record_index + 1) % 4), dtype=torch.bfloat16)
            final_values[kv_local_index, record_index, 3] = torch.tensor(0.01 * ((record_index % 2) * 2 - 1), dtype=torch.bfloat16)
    tensors = {
        "queries": queries,
        "final_keys": final_keys,
        "final_values": final_values,
        "visible_lengths": torch.arange(1, sequence_length + 1, dtype=torch.int64),
        "record_token_ids": torch.arange(sequence_length, dtype=torch.int64),
        "record_token_positions": torch.arange(sequence_length, dtype=torch.int64),
        "record_layer_indices": torch.full((sequence_length,), layer_index, dtype=torch.int64),
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
        "checkpoint_repo": "synthetic/stage4-test",
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
    return metadata["checkpoint_revision"]


def _build_capture_dir(
    root: Path,
    *,
    layer_indices: tuple[int, ...] = (0, 8, 16, 24, 31),
    sequence_length: int = 16,
    future_offset: float = 0.0,
) -> Path:
    capture_dir = root / "capture"
    capture_dir.mkdir(parents=True, exist_ok=True)
    revision = None
    layers = []
    for layer_index in layer_indices:
        trace_path = capture_dir / f"llama31_layer{layer_index}_trace.safetensors"
        revision = _build_synthetic_trace(
            trace_path,
            layer_index=layer_index,
            sequence_length=sequence_length,
            future_offset=future_offset,
        )
        tensors_shape = [sequence_length, 3, 4]
        layers.append(
            {
                "layer_index": layer_index,
                "trace_path": str(trace_path),
                "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                "queries_shape": tensors_shape,
                "final_keys_shape": [2, sequence_length, 4],
                "final_values_shape": [2, sequence_length, 4],
                "model_head_outputs_shape": tensors_shape,
                "selected_query_heads": [0, 1, 4],
                "selected_kv_heads": [0, 1],
                "query_to_kv_heads": [0, 0, 1],
                "stock_forward_comparison": {
                    "projected_output_max_abs_diff": 0.0,
                    "decoder_output_max_abs_diff": 0.0,
                },
                "cache_comparison": {
                    "key_max_abs_diff": 0.0,
                    "value_max_abs_diff": 0.0,
                },
                "compact_replay": {
                    "max_abs_diff": 0.0,
                },
            }
        )
    payload = {
        "tls_verification": True,
        "selected_query_heads": [0, 1, 4],
        "selected_kv_heads": [0, 1],
        "query_to_kv_heads": [0, 0, 1],
        "checkpoint": {
            "repo_id": "synthetic/stage4-test",
            "revision": revision,
        },
        "layers": layers,
    }
    (capture_dir / "capture_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return capture_dir


class Stage4BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = _load_stage4_script_module()

    def _build_context(self, *, sequence_length: int = 16, future_offset: float = 0.0, layer_index: int = 0):
        self.temp_dir = tempfile.TemporaryDirectory()
        capture_dir = _build_capture_dir(Path(self.temp_dir.name), sequence_length=sequence_length, future_offset=future_offset)
        traces, _ = validate_stage4_capture_inputs(capture_dir=capture_dir, layer_indices=(0, 8, 16, 24, 31))
        trace = traces[layer_index]
        context = build_stage4_case_context(
            trace=trace,
            record_index=sequence_length - 1,
            query_local_index=0,
            recent_window=4,
            precision=64,
        )
        return context, capture_dir

    def tearDown(self) -> None:
        temp_dir = getattr(self, "temp_dir", None)
        if temp_dir is not None:
            temp_dir.cleanup()
            self.temp_dir = None

    def test_validate_stage4_capture_inputs_accepts_arbitrary_layer_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            capture_dir = _build_capture_dir(Path(temp_dir))
            traces, _ = validate_stage4_capture_inputs(capture_dir=capture_dir, layer_indices=(8, 31))
            self.assertEqual(tuple(sorted(traces.keys())), (8, 31))

    def test_full_kv_reproduces_reference_output(self) -> None:
        context, _ = self._build_context()
        result, payload = run_baseline_case(
            context=context,
            method_name="full_kv",
            mode="native",
            precision=64,
        )
        self.assertEqual(len(payload), result.total_serialized_bytes)
        self.assertAlmostEqual(result.attention_output_l2_error, 0.0, places=12)
        self.assertAlmostEqual(result.relative_l2_error, 0.0, places=12)

    def test_matched_budget_only_applies_to_tunable_methods(self) -> None:
        context, _ = self._build_context()
        rack_native, _ = run_baseline_case(
            context=context,
            method_name="rack_kv",
            mode="native",
            recent_window=4,
            block_size=2,
            precision=64,
        )
        target = rack_native.total_serialized_bytes
        for method_name in ("full_kv", "rack_kv", "uniform_int8_kv", "kivi_style"):
            with self.assertRaises(ValueError):
                run_baseline_case(
                    context=context,
                    method_name=method_name,
                    mode="matched_budget",
                    recent_window=4,
                    block_size=2,
                    precision=64,
                    rack_target_bytes=target,
                )
        for method_name in ("snapkv_style", "quest_style"):
            matched, _ = run_baseline_case(
                context=context,
                method_name=method_name,
                mode="matched_budget",
                recent_window=4,
                block_size=2,
                precision=64,
                rack_target_bytes=target,
            )
            self.assertTrue(matched.budget_tunable)
            self.assertTrue(matched.budget_match_attempted)
            self.assertEqual(matched.budget_target_bytes, target)
            self.assertEqual(matched.parameters["budget_matching_policy_version"], STAGE4_BUDGET_MATCHING_POLICY_VERSION)

    def test_stage4_method_mode_plan_is_exact(self) -> None:
        plan = self.script._method_mode_plan()
        self.assertEqual(
            plan,
            (
                ("rack_kv", "native"),
                ("full_kv", "native"),
                ("uniform_int8_kv", "native"),
                ("kivi_style", "native"),
                ("snapkv_style", "native"),
                ("snapkv_style", "matched_budget"),
                ("quest_style", "native"),
                ("quest_style", "matched_budget"),
            ),
        )

    def test_all_methods_are_causal_on_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_a, tempfile.TemporaryDirectory() as temp_dir_b:
            capture_a = _build_capture_dir(Path(temp_dir_a), sequence_length=16, future_offset=0.0)
            capture_b = _build_capture_dir(Path(temp_dir_b), sequence_length=16, future_offset=10.0)
            traces_a, _ = validate_stage4_capture_inputs(capture_dir=capture_a, layer_indices=(0, 8, 16, 24, 31))
            traces_b, _ = validate_stage4_capture_inputs(capture_dir=capture_b, layer_indices=(0, 8, 16, 24, 31))
            context_a = build_stage4_case_context(trace=traces_a[0], record_index=7, query_local_index=0, recent_window=4, precision=64)
            context_b = build_stage4_case_context(trace=traces_b[0], record_index=7, query_local_index=0, recent_window=4, precision=64)
            rack_a, _ = run_baseline_case(context=context_a, method_name="rack_kv", mode="native", recent_window=4, block_size=2, precision=64, trace=traces_a[0])
            rack_b, _ = run_baseline_case(context=context_b, method_name="rack_kv", mode="native", recent_window=4, block_size=2, precision=64, trace=traces_b[0])
            self.assertAlmostEqual(rack_a.attention_output_l2_error, rack_b.attention_output_l2_error, places=12)
            self.assertEqual(rack_a.total_serialized_bytes, rack_b.total_serialized_bytes)
            rack_target = rack_a.total_serialized_bytes

            cases = (
                ("full_kv", "native"),
                ("rack_kv", "native"),
                ("uniform_int8_kv", "native"),
                ("kivi_style", "native"),
                ("snapkv_style", "native"),
                ("snapkv_style", "matched_budget"),
                ("quest_style", "native"),
                ("quest_style", "matched_budget"),
            )
            for method_name, mode in cases:
                result_a, payload_a = run_baseline_case(
                    context=context_a,
                    method_name=method_name,
                    mode=mode,
                    recent_window=4,
                    block_size=2,
                    precision=64,
                    trace=traces_a[0] if method_name == "rack_kv" else None,
                    rack_target_bytes=rack_target if mode == "matched_budget" else None,
                )
                result_b, payload_b = run_baseline_case(
                    context=context_b,
                    method_name=method_name,
                    mode=mode,
                    recent_window=4,
                    block_size=2,
                    precision=64,
                    trace=traces_b[0] if method_name == "rack_kv" else None,
                    rack_target_bytes=rack_target if mode == "matched_budget" else None,
                )
                self.assertEqual(payload_a, payload_b, msg=f"{method_name}:{mode} payload changed after future perturbation")
                self.assertAlmostEqual(result_a.attention_output_l2_error, result_b.attention_output_l2_error, places=12, msg=f"{method_name}:{mode} changed after future perturbation")

    def test_serialized_byte_accounting_matches_payload_length(self) -> None:
        context, _ = self._build_context()
        rack_native, _ = run_baseline_case(
            context=context,
            method_name="rack_kv",
            mode="native",
            recent_window=4,
            block_size=2,
            tolerance=STAGE4_RACK_TOLERANCE,
            precision=64,
        )
        target = rack_native.total_serialized_bytes
        for method_name, mode in self.script._method_mode_plan():
            result, payload = run_baseline_case(
                context=context,
                method_name=method_name,
                mode=mode,
                recent_window=4,
                block_size=2,
                tolerance=STAGE4_RACK_TOLERANCE,
                precision=64,
                rack_target_bytes=target if mode == "matched_budget" else None,
            )
            accounted = (
                result.encoded_key_bytes
                + result.encoded_value_bytes
                + result.scales_bytes
                + result.metadata_bytes
                + result.indices_bytes
                + result.block_page_metadata_bytes
                + result.recent_window_bytes
            )
            self.assertEqual(len(payload), result.total_serialized_bytes)
            self.assertEqual(accounted, result.total_serialized_bytes)

    def test_source_snapshot_manifest_uses_actual_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "src" / "a.py").write_text("print('a')\n", encoding="utf-8")
            (root / "src" / "b.py").write_text("print('b')\n", encoding="utf-8")
            (root / "out.txt").write_text("out\n", encoding="utf-8")
            manifest = self.script._build_manifest(
                file_map={"out.txt": root / "out.txt", "src/a.py": root / "src" / "a.py", "src/b.py": root / "src" / "b.py"},
                source_relative_paths=("src/a.py", "src/b.py"),
                source_snapshot_base_dir=root,
            )
            expected_lines = [
                f"src/a.py\t{hashlib.sha256((root / 'src' / 'a.py').read_bytes()).hexdigest()}",
                f"src/b.py\t{hashlib.sha256((root / 'src' / 'b.py').read_bytes()).hexdigest()}",
            ]
            expected = hashlib.sha256("\n".join(expected_lines).encode("utf-8")).hexdigest()
            self.assertEqual(manifest["source_snapshot_sha256"], expected)
            self.assertEqual(manifest["file_count"], 3)

    def test_resume_matches_uninterrupted_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            capture_dir = _build_capture_dir(root, sequence_length=16)
            output_dir = root / "out"
            first = self.script._run_benchmark(
                capture_dir=capture_dir,
                output_dir=output_dir,
                run_label="Test Benchmark",
                layer_indices=(0, 31),
                query_positions=(7, 15),
                query_local_indices=(0,),
                recent_window=4,
                block_size=2,
                tolerance=0.05,
                precision=64,
                seed=0,
                max_rss_bytes=8 * 1024 ** 3,
                min_free_bytes=2 * 1024 ** 3,
            )
            second = self.script._run_benchmark(
                capture_dir=capture_dir,
                output_dir=output_dir,
                run_label="Test Benchmark",
                layer_indices=(0, 31),
                query_positions=(7, 15),
                query_local_indices=(0,),
                recent_window=4,
                block_size=2,
                tolerance=0.05,
                precision=64,
                seed=0,
                max_rss_bytes=8 * 1024 ** 3,
                min_free_bytes=2 * 1024 ** 3,
            )
            self.assertEqual(first["case_results"], second["case_results"])
            self.assertEqual(first["aggregate_by_method_mode"], second["aggregate_by_method_mode"])
            self.assertEqual(first["actual_result_count"], 2 * 2 * 1 * 8)

    def test_completed_cases_reuse_checkpoints_without_rebuilding_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            capture_dir = _build_capture_dir(root, sequence_length=16)
            output_dir = root / "out"
            first = self.script._run_benchmark(
                capture_dir=capture_dir,
                output_dir=output_dir,
                run_label="Checkpoint Reuse Test",
                layer_indices=(0, 31),
                query_positions=(7, 15),
                query_local_indices=(0,),
                recent_window=4,
                block_size=2,
                tolerance=0.05,
                precision=64,
                seed=0,
                max_rss_bytes=8 * 1024 ** 3,
                min_free_bytes=2 * 1024 ** 3,
            )
            original_builder = self.script.build_stage4_case_context

            def _unexpected_rebuild(*args, **kwargs):
                raise AssertionError("Completed Stage 4 cases should reuse checkpoints without rebuilding full MPFR context.")

            self.script.build_stage4_case_context = _unexpected_rebuild
            try:
                second = self.script._run_benchmark(
                    capture_dir=capture_dir,
                    output_dir=output_dir,
                    run_label="Checkpoint Reuse Test",
                    layer_indices=(0, 31),
                    query_positions=(7, 15),
                    query_local_indices=(0,),
                    recent_window=4,
                    block_size=2,
                    tolerance=0.05,
                    precision=64,
                    seed=0,
                    max_rss_bytes=8 * 1024 ** 3,
                    min_free_bytes=2 * 1024 ** 3,
                )
            finally:
                self.script.build_stage4_case_context = original_builder

            self.assertEqual(first["case_results"], second["case_results"])
            self.assertEqual(first["aggregate_by_method_mode"], second["aggregate_by_method_mode"])

    def test_stale_stage4a_checkpoint_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            capture_dir = _build_capture_dir(root, sequence_length=16)
            traces, _ = validate_stage4_capture_inputs(capture_dir=capture_dir, layer_indices=(0, 8, 16, 24, 31))
            context = build_stage4_case_context(trace=traces[0], record_index=15, query_local_index=0, recent_window=4, precision=64)
            result, payload = run_baseline_case(
                context=context,
                method_name="snapkv_style",
                mode="matched_budget",
                recent_window=4,
                block_size=2,
                precision=64,
                rack_target_bytes=100,
            )
            output_dir = root / "out"
            checkpoint_path = self.script._result_checkpoint_path(output_dir=output_dir, case_key="stale", method_name="snapkv_style", mode="matched_budget")
            payload_path = self.script._payload_path(output_dir=output_dir, case_key="stale", method_name="snapkv_style", mode="matched_budget")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_bytes(payload)
            stale_settings = {
                "schema": "stage4_baseline_case_checkpoint_v1",
                "run_schema": "stage4_baseline_run_v1",
                "result_version": "stage4_smoke_v1",
                "method_version": "stage4_baselines_v1",
                "trace_path": str(context.trace_path),
                "trace_sha256": context.trace_sha256,
                "layer_index": context.layer_index,
                "record_index": context.record_index,
                "query_position": context.query_position,
                "query_local_index": context.query_local_index,
                "query_head_global": context.query_head_global,
                "kv_head_global": context.kv_head_global,
                "visible_length": context.visible_length,
                "historical_tokens": context.historical_tokens,
                "recent_window": 4,
                "block_size": 2,
                "tolerance": 0.05,
                "precision": 64,
                "seed": 0,
                "method_name": "snapkv_style",
                "mode": "matched_budget",
                "rack_target_bytes": 100,
                "budget_matching_policy_version": "nearest_serialized_bytes_v1",
                "selected_keep_count": result.parameters["keep_count"],
                "budget_tunable": True,
                "budget_match_attempted": True,
            }
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "settings": stale_settings,
                        "result": self.script._json_normalize(self.script.stage4_case_result_to_dict(result)),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            expected_settings = self.script._base_case_settings(
                context=context,
                method_name="snapkv_style",
                mode="matched_budget",
                recent_window=4,
                block_size=2,
                tolerance=0.05,
                precision=64,
                seed=0,
                rack_target_bytes=100,
            )
            with self.assertRaises(self.script.Stage4ExecutionError):
                self.script._load_checkpoint(
                    checkpoint_path=checkpoint_path,
                    payload_path=payload_path,
                    expected_settings=expected_settings,
                )

    def test_selection_validation_accepts_arbitrary_positions_and_query_locals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            capture_dir = _build_capture_dir(Path(temp_dir), sequence_length=16)
            traces, _ = validate_stage4_capture_inputs(capture_dir=capture_dir, layer_indices=(0, 8, 16, 24, 31))
            info = self.script._validate_requested_selection(
                traces=traces,
                layer_indices=(0, 31),
                query_positions=(3, 7, 15),
                query_local_indices=(0, 2),
            )
            self.assertEqual(info["selected_query_heads"], [0, 1, 4])
            self.assertEqual(info["selected_kv_heads"], [0, 1])
            self.assertEqual(info["query_to_kv_heads"], [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
