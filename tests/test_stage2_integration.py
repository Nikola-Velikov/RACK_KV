from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gmpy2
import numpy as np
import torch

from rack_kv.certificate import CertificationResult, certify_progressive_skipping as rigorous_certify_progressive_skipping
from rack_kv.codec import deserialize_block_container as codec_deserialize_block_container
from rack_kv.llama_trace import load_attention_trace, save_attention_trace
from rack_kv.stage2 import (
    TRACE_SCHEMA_COMPACT_V1,
    _build_prefix_result,
    _build_query_case_base,
    _run_query_case,
    _skip_validation_flags,
    result_to_dict,
    run_stage2_smoke_experiment,
    validate_compact_trace,
)
from rack_kv.types import CertificateMode


TRACE_PATH = Path(".tmp/llama31_capture/llama31_layer0_trace.safetensors")


class Stage2IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trace = validate_compact_trace(TRACE_PATH)
        cls.result_w4b4 = run_stage2_smoke_experiment(
            trace_path=TRACE_PATH,
            recent_windows=(4,),
            block_sizes=(4,),
            tolerances=(0.05,),
            precision=256,
            random_seed=0,
        )
        cls.result_w2b2 = run_stage2_smoke_experiment(
            trace_path=TRACE_PATH,
            recent_windows=(2,),
            block_sizes=(2,),
            tolerances=(0.1,),
            precision=256,
            random_seed=0,
        )

    def test_real_compact_trace_loading_uses_saved_metadata_and_provenance(self) -> None:
        trace = self.trace
        self.assertEqual(trace.trace_schema, TRACE_SCHEMA_COMPACT_V1)
        self.assertEqual(trace.trace_sha256, "97cb97e55e9c1f121fa06d250f6e03b8d6d92f2c044fa96e0f11d8c893b0099e")
        self.assertEqual(trace.trace_sha256_from_report, trace.trace_sha256)
        self.assertEqual(trace.selected_query_heads, (0, 1, 4))
        self.assertEqual(trace.selected_kv_heads, (0, 1))
        self.assertEqual(trace.query_to_kv_heads, (0, 0, 1))
        self.assertEqual(trace.visible_lengths, tuple(range(1, 13)))
        self.assertEqual(trace.query_positions, tuple(range(12)))
        self.assertEqual(tuple(trace.queries.shape), (12, 3, 128))
        self.assertEqual(tuple(trace.final_keys.shape), (2, 12, 128))
        self.assertEqual(tuple(trace.final_values.shape), (2, 12, 128))
        self.assertEqual(tuple(trace.model_head_outputs.shape), (12, 3, 128))
        self.assertAlmostEqual(trace.scaling, 1.0 / math.sqrt(trace.head_dim), places=15)

    def test_invalid_trace_rejection_does_not_repair_inconsistent_mapping(self) -> None:
        tensors, metadata = load_attention_trace(TRACE_PATH)
        metadata["query_to_kv_heads"] = [0, 1, 1]
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "broken_trace.safetensors"
            save_attention_trace(trace_path, tensors, metadata)
            with self.assertRaises(ValueError):
                validate_compact_trace(trace_path)

    def test_real_trace_smoke_configuration_has_empty_and_multi_block_cases(self) -> None:
        result = self.result_w4b4
        self.assertEqual(len(result.configs), 1)
        config = result.configs[0]
        self.assertEqual(config.evaluated_query_cases, 36)
        self.assertGreater(config.eligible_query_cases, 0)
        self.assertTrue(any(case.candidate_blocks == 0 for case in config.query_case_results))
        self.assertTrue(any(case.candidate_blocks >= 2 for case in config.query_case_results))
        self.assertEqual(config.rigorous_interval_violation_count, 0)
        self.assertEqual(config.approximate_observed_violation_count, 0)
        self.assertEqual(config.false_safe_violation_count, 0)
        self.assertEqual(config.reference_decomposition_violation_count, 0)
        self.assertEqual(config.model_relative_decomposition_violation_count, 0)
        self.assertGreaterEqual(config.aggregate_prefix_serialized_bytes, 1)

    def test_real_trace_smoke_configuration_keeps_error_terms_separated(self) -> None:
        result = self.result_w2b2
        config = result.configs[0]
        self.assertGreater(config.unique_prefix_slices_with_history, 0)
        self.assertGreater(config.number_of_blocks, 0)
        self.assertGreater(config.independently_decoded_blocks_tested, 0)
        self.assertEqual(config.independent_versus_full_decode_max_difference, 0.0)
        self.assertTrue(config.unaffected_block_decoding_passed)
        self.assertGreater(config.max_model_reference_gap, 0.0)
        self.assertEqual(config.false_safe_violation_count, 0)
        for case in config.query_case_results:
            self.assertLessEqual(case.reference_decomposition_lhs, case.reference_decomposition_rhs + 1e-9)
            self.assertLessEqual(case.model_relative_decomposition_lhs, case.model_relative_decomposition_rhs + 1e-9)
            self.assertGreaterEqual(case.model_reference_gap, 0.0)
            self.assertGreaterEqual(case.compression_error, 0.0)
            self.assertGreaterEqual(case.observed_skip_error, 0.0)
            self.assertGreaterEqual(case.reference_total_error, 0.0)
            self.assertGreaterEqual(case.captured_model_total_gap, 0.0)
            if case.candidate_blocks == 0:
                self.assertEqual(case.observed_skip_error, 0.0)
                self.assertEqual(case.compression_error, 0.0)

    def test_zero_compression_path_has_zero_true_compression_error_but_nonzero_model_gap(self) -> None:
        record_index = self.trace.query_records - 1
        kv_head_global = self.trace.query_to_kv_heads[0]
        prefix_keys, prefix_values = self.trace.kv_prefix(record_index=record_index, kv_head_global=kv_head_global)
        prefix_result = _build_prefix_result(
            record_index=record_index,
            kv_head_global=kv_head_global,
            prefix_keys=prefix_keys,
            prefix_values=prefix_values,
            recent_window=int(prefix_keys.shape[0]),
            block_size=2,
            precision=256,
        )
        base_case = _build_query_case_base(
            trace=self.trace,
            prefix_result=prefix_result,
            query_local_index=0,
            precision=256,
        )
        self.assertEqual(prefix_result.blocks, ())
        self.assertEqual(base_case.compression_error, 0.0)
        self.assertGreaterEqual(base_case.model_reference_gap, 0.0)

    def test_saved_scaling_is_used_by_replay_outputs(self) -> None:
        tensors, metadata = load_attention_trace(TRACE_PATH)
        metadata["scaling"] = float(metadata["scaling"]) * 0.5
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "scaled_trace.safetensors"
            save_attention_trace(trace_path, tensors, metadata)
            modified_trace = validate_compact_trace(trace_path)
            record_index = modified_trace.query_records - 1
            kv_head_global = modified_trace.query_to_kv_heads[0]
            prefix_keys, prefix_values = modified_trace.kv_prefix(record_index=record_index, kv_head_global=kv_head_global)
            prefix_result = _build_prefix_result(
                record_index=record_index,
                kv_head_global=kv_head_global,
                prefix_keys=prefix_keys,
                prefix_values=prefix_values,
                recent_window=2,
                block_size=2,
                precision=256,
            )
            modified_base = _build_query_case_base(
                trace=modified_trace,
                prefix_result=prefix_result,
                query_local_index=0,
                precision=256,
            )
        original_trace = self.trace
        original_prefix_keys, original_prefix_values = original_trace.kv_prefix(record_index=record_index, kv_head_global=kv_head_global)
        original_prefix_result = _build_prefix_result(
            record_index=record_index,
            kv_head_global=kv_head_global,
            prefix_keys=original_prefix_keys,
            prefix_values=original_prefix_values,
            recent_window=2,
            block_size=2,
            precision=256,
        )
        original_base = _build_query_case_base(
            trace=original_trace,
            prefix_result=original_prefix_result,
            query_local_index=0,
            precision=256,
        )
        self.assertFalse(
            np.allclose(
                original_base.original_reference_output,
                modified_base.original_reference_output,
                atol=0.0,
                rtol=0.0,
            )
        )

    def test_rigorous_interval_upper_is_authoritative_validation_condition(self) -> None:
        certificate_value = gmpy2.mpfr("1.0")
        observed_skip_error_mpfr = gmpy2.mpfr("0.5")
        rigorous_skip_error_upper = gmpy2.mpfr("1.25")
        approximate_violation, rigorous_violation = _skip_validation_flags(
            certificate_value=certificate_value,
            observed_skip_error_mpfr=observed_skip_error_mpfr,
            rigorous_skip_error_upper=rigorous_skip_error_upper,
        )
        self.assertFalse(approximate_violation)
        self.assertTrue(rigorous_violation)

    def test_result_dict_uses_corrected_memory_and_error_field_names(self) -> None:
        result_dict = result_to_dict(self.result_w2b2)
        config = result_dict["configurations"][0]
        self.assertIn("aggregate_prefix_original_bytes", config)
        self.assertIn("final_prefix_original_bytes", config)
        self.assertIn("max_true_compression_error", config)
        self.assertIn("max_model_reference_gap", config)
        self.assertIn("global_prefix_ratio_summary", result_dict)
        self.assertNotIn("original_kv_bytes", config)
        self.assertNotIn("serialized_compressed_bytes", config)
        self.assertNotIn("max_compression_error", config)
        self.assertNotIn("max_total_observed_error", config)

    def test_stage2_prefix_uses_reloaded_container_blocks_as_authoritative(self) -> None:
        class TaggedContainer:
            def __init__(self, inner) -> None:
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def deserialize_block(self, block_index: int):
                block = self._inner.deserialize_block(block_index)
                object.__setattr__(block, "_stage2_roundtrip_tag", True)
                return block

        trace = self.trace
        record_index = trace.query_records - 1
        kv_head_global = trace.query_to_kv_heads[0]
        prefix_keys, prefix_values = trace.kv_prefix(record_index=record_index, kv_head_global=kv_head_global)

        with patch("rack_kv.stage2.deserialize_block_container", side_effect=lambda payload: TaggedContainer(codec_deserialize_block_container(payload))):
            prefix_result = _build_prefix_result(
                record_index=record_index,
                kv_head_global=kv_head_global,
                prefix_keys=prefix_keys,
                prefix_values=prefix_values,
                recent_window=2,
                block_size=2,
                precision=256,
            )

        self.assertTrue(prefix_result.authoritative_serialized_roundtrip_used)
        self.assertIsNotNone(prefix_result.serialized_container)
        self.assertGreater(len(prefix_result.blocks), 0)
        self.assertTrue(all(getattr(block, "_stage2_roundtrip_tag", False) for block in prefix_result.blocks))
        for block_index, block in enumerate(prefix_result.blocks):
            payload = prefix_result.serialized_container.payload_for_block(block_index)
            self.assertEqual(block.serialize(), payload)

    def test_query_case_certification_receives_reloaded_serialized_blocks(self) -> None:
        class TaggedContainer:
            def __init__(self, inner) -> None:
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def deserialize_block(self, block_index: int):
                block = self._inner.deserialize_block(block_index)
                object.__setattr__(block, "_stage2_roundtrip_tag", True)
                return block

        trace = self.trace
        record_index = trace.query_records - 1
        query_local_index = 0
        kv_head_global = trace.query_to_kv_heads[query_local_index]
        prefix_keys, prefix_values = trace.kv_prefix(record_index=record_index, kv_head_global=kv_head_global)
        with patch("rack_kv.stage2.deserialize_block_container", side_effect=lambda payload: TaggedContainer(codec_deserialize_block_container(payload))):
            prefix_result = _build_prefix_result(
                record_index=record_index,
                kv_head_global=kv_head_global,
                prefix_keys=prefix_keys,
                prefix_values=prefix_values,
                recent_window=2,
                block_size=2,
                precision=256,
            )
        base_case = _build_query_case_base(
            trace=trace,
            prefix_result=prefix_result,
            query_local_index=query_local_index,
            precision=256,
        )

        seen = {"checked": False}

        def wrapped_certify(**kwargs):
            historical_blocks = kwargs["historical_blocks"]
            self.assertTrue(historical_blocks)
            self.assertTrue(all(getattr(block, "_stage2_roundtrip_tag", False) for block in historical_blocks))
            self.assertEqual(
                [block.header.rho_upper for block in historical_blocks],
                [block.header.rho_upper for block in prefix_result.blocks],
            )
            self.assertEqual(
                [block.header.nu_upper for block in historical_blocks],
                [block.header.nu_upper for block in prefix_result.blocks],
            )
            seen["checked"] = True
            return rigorous_certify_progressive_skipping(**kwargs)

        with patch("rack_kv.stage2.certify_progressive_skipping", side_effect=wrapped_certify):
            case = _run_query_case(
                trace=trace,
                prefix_result=prefix_result,
                query_local_index=query_local_index,
                tolerance=0.1,
                precision=256,
                base_case=base_case,
            )
        self.assertTrue(seen["checked"])
        self.assertIsInstance(case.certificate_name, str)

    def test_real_trace_loading_rejects_bad_selected_query_head_tensor(self) -> None:
        tensors, metadata = load_attention_trace(TRACE_PATH)
        tensors["selected_query_heads"] = torch.tensor([0, 1, 2], dtype=torch.int64)
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "broken_heads.safetensors"
            save_attention_trace(trace_path, tensors, metadata)
            with self.assertRaises(ValueError):
                validate_compact_trace(trace_path)


if __name__ == "__main__":
    unittest.main()
