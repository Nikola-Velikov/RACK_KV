from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import weakref
from unittest import mock
import zipfile

import numpy as np
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from rack_kv.certificate import prepare_certification_inputs, prepared_full_exact_rows
from rack_kv.llama_trace import _RemoteTensorSpec, query_head_to_kv_head
from rack_kv.stage5 import (
    MemoryGuard,
    PersistentTensorRangeCache,
    STAGE5_EXPERIMENT_SCOPE_LAYER0_SAFETY_VALIDATION,
    STAGE5_MODIFIED_LAYERS,
    STAGE5_PROMPT_CORPUS_VERSION,
    STAGE5_REPORT_MD,
    STAGE5_RESULTS_JSON,
    STAGE5_PROMPT_TOKEN_COUNT,
    STAGE5_PROMPT_CHECKPOINT_SCHEMA,
    STAGE5_STORAGE_ACCOUNTING_SCHEMA,
    STAGE5_STREAM_CHECKPOINT_SCHEMA,
    Stage5ExecutionError,
    _aggregate_certificate_records,
    _accumulate_stage5_storage_records,
    _block_lengths_for_historical_tokens,
    _normalize_stage5_methods,
    _build_prompt_texts,
    _certified_attention_for_head,
    _empty_storage_category_sums,
    _gqa_groups,
    _load_prompt_checkpoint,
    _load_stream_checkpoint,
    _manual_attention,
    _method_storage_summary,
    _prompt_method_metrics,
    _precompute_qkv,
    _prefix_context,
    _build_stage5_rack_prefix_payload,
    _run_modified_layer_stream,
    _save_prompt_checkpoint,
    _save_stream_checkpoint,
    _shared_reconstructed_attention_output_numpy,
    _source_snapshot_sha256,
    _stage4_artifact_hashes,
    _stage5_wrapper_total_bytes,
    _stock_layer_forward_with_intermediates,
    _stream_hidden_state_path,
    _stream_settings_payload,
    _save_tensor_artifact,
    _validate_stage4_artifacts_unchanged,
    build_stage5_prompt_corpus,
)
from scripts.run_stage5_quality_smoke import (
    _finalize_from_checkpoints,
    _repair_storage_summary_from_checkpoints,
    _layer0_storage_summary,
)


def _tiny_config() -> LlamaConfig:
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=512,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
    )
    config._attn_implementation = "eager"
    return config


def _tiny_layer() -> LlamaDecoderLayer:
    torch.manual_seed(0)
    layer = LlamaDecoderLayer(_tiny_config(), layer_idx=0).to(dtype=torch.bfloat16)
    layer.eval()
    return layer


def _tiny_hidden(sequence_length: int = 10) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(sequence_length, 128, dtype=torch.float32).to(torch.bfloat16)


def _synthetic_rack_storage_record(
    *,
    layer_index: int,
    token_index: int,
    kv_head_global: int,
    visible_length: int,
    recent_window: int,
    block_size: int,
    head_dim: int,
    value_dim: int,
    precision: int,
) -> dict[str, object]:
    recent_exact_tokens = min(int(visible_length), int(recent_window))
    historical_tokens = max(int(visible_length) - recent_exact_tokens, 0)
    block_lengths = _block_lengths_for_historical_tokens(historical_tokens, block_size)
    block_count = len(block_lengths)
    anchor_bytes = block_count * (2 * head_dim + 2 * value_dim)
    encoded_key_bytes = sum(max(block_len - 1, 0) * head_dim for block_len in block_lengths)
    encoded_value_bytes = sum(max(block_len - 1, 0) * value_dim for block_len in block_lengths)
    quantization_scale_bytes = block_count * 4
    block_metadata_bytes = block_count * 16
    index_bytes = block_count * 8
    recent_window_bytes = recent_exact_tokens * head_dim * 2 + recent_exact_tokens * value_dim * 2
    historical_container_bytes = 0
    if block_count > 0:
        historical_container_bytes = (
            16
            + index_bytes
            + block_metadata_bytes
            + anchor_bytes
            + quantization_scale_bytes
            + encoded_key_bytes
            + encoded_value_bytes
        )
    total_serialized_bytes, _wrapper_header_bytes = _stage5_wrapper_total_bytes(
        recent_keys_shape=(recent_exact_tokens, head_dim),
        recent_values_shape=(recent_exact_tokens, value_dim),
        recent_keys_bytes=recent_exact_tokens * head_dim * 2,
        recent_values_bytes=recent_exact_tokens * value_dim * 2,
        historical_container_bytes=historical_container_bytes,
        recent_window=recent_window,
        block_size=block_size,
        precision=precision,
        historical_tokens=historical_tokens,
    )
    return {
        "layer_index": int(layer_index),
        "token_index": int(token_index),
        "prompt_name": "synthetic",
        "method_name": "rack_kv_compression_only",
        "kv_head_global": int(kv_head_global),
        "visible_length": int(visible_length),
        "historical_tokens": int(historical_tokens),
        "byte_breakdown": {
            "total_serialized_bytes": int(total_serialized_bytes),
        },
    }


def _synthetic_rack_storage_records(
    *,
    sequence_length: int,
    modified_layers: tuple[int, ...],
    num_kv_heads: int,
    recent_window: int,
    block_size: int,
    head_dim: int,
    value_dim: int,
    precision: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for layer_index in modified_layers:
        for token_index in range(sequence_length):
            visible_length = token_index + 1
            for kv_head_global in range(num_kv_heads):
                records.append(
                    _synthetic_rack_storage_record(
                        layer_index=layer_index,
                        token_index=token_index,
                        kv_head_global=kv_head_global,
                        visible_length=visible_length,
                        recent_window=recent_window,
                        block_size=block_size,
                        head_dim=head_dim,
                        value_dim=value_dim,
                        precision=precision,
                    )
                )
    return records


def _synthetic_metric_record(
    *,
    token_index: int,
    target_token_id: int,
    nll: float,
    delta_nll_vs_full: float = 0.0,
    kl_divergence: float = 0.0,
    js_divergence: float = 0.0,
    logit_l2_error: float = 0.0,
    relative_logit_l2_error: float = 0.0,
    max_abs_logit_component_error: float = 0.0,
    cosine_similarity: float = 1.0,
    top1_agreement_with_full: bool = True,
    top1_token: int = 1,
) -> dict[str, object]:
    return {
        "schema": "stage5_metric_record_v1",
        "token_index": int(token_index),
        "target_token_id": int(target_token_id),
        "nll": float(nll),
        "delta_nll_vs_full": float(delta_nll_vs_full),
        "full_target_log_probability": float(-nll + delta_nll_vs_full),
        "method_target_log_probability": float(-nll),
        "full_top1_token": int(top1_token),
        "full_top5_token_ids": [int(top1_token), 2, 3, 4, 5],
        "top1_token": int(top1_token if top1_agreement_with_full else top1_token + 1),
        "top5_token_ids": [int(top1_token), 2, 3, 4, 5],
        "top1_agreement_with_full": bool(top1_agreement_with_full),
        "top5_contains_full_top1": True,
        "top5_set_overlap_count": 5,
        "kl_divergence": float(kl_divergence),
        "jensen_shannon_divergence": float(js_divergence),
        "logit_l2_error": float(logit_l2_error),
        "relative_logit_l2_error": float(relative_logit_l2_error),
        "max_abs_logit_component_error": float(max_abs_logit_component_error),
        "cosine_similarity": float(cosine_similarity),
        "target_token_rank_full": 1,
        "target_token_rank_method": 1,
        "target_token_rank_change": 0,
    }


class _FakeShard:
    def __init__(self, *, spec: _RemoteTensorSpec, url: str = "memory://fake") -> None:
        self._spec = spec
        self.url = url

    def tensor_spec(self, tensor_name: str) -> _RemoteTensorSpec:
        return self._spec


class Stage5QualityTests(unittest.TestCase):
    def test_all_head_gqa_geometry_is_preserved(self) -> None:
        groups = _gqa_groups(32, 8)
        self.assertEqual(len(groups), 8)
        self.assertEqual(groups[0], (0, 1, 2, 3))
        self.assertEqual(groups[1], (4, 5, 6, 7))
        self.assertEqual(groups[-1], (28, 29, 30, 31))
        self.assertEqual(query_head_to_kv_head(0, num_attention_heads=32, num_key_value_heads=8), 0)
        self.assertEqual(query_head_to_kv_head(4, num_attention_heads=32, num_key_value_heads=8), 1)
        self.assertEqual(query_head_to_kv_head(31, num_attention_heads=32, num_key_value_heads=8), 7)

    def test_precompute_qkv_processes_all_heads_and_kv_heads(self) -> None:
        config = _tiny_config()
        layer = _tiny_layer()
        hidden = _tiny_hidden(6)
        queries, keys, values = _precompute_qkv(layer=layer, config=config, hidden_states=hidden)
        self.assertEqual(tuple(queries.shape), (6, 32, 4))
        self.assertEqual(tuple(keys.shape), (6, 8, 4))
        self.assertEqual(tuple(values.shape), (6, 8, 4))

    def test_full_kv_custom_modified_layer_matches_stock_layer_output(self) -> None:
        config = _tiny_config()
        layer = _tiny_layer()
        hidden = _tiny_hidden(12)
        stock_attention, stock_projected, stock_post_attn, stock_mlp, stock_decoder = _stock_layer_forward_with_intermediates(
            layer=layer,
            config=config,
            hidden_states=hidden,
        )
        custom_decoder, custom_outputs, _storage, _certs = _run_modified_layer_stream(
            method_name="full_kv",
            layer=layer,
            config=config,
            hidden_states=hidden,
            layer_index=0,
            recent_window=16,
            block_size=8,
            tolerance=0.05,
            precision=128,
            prompt_name="synthetic",
            certificate_records=[],
        )
        self.assertLess(
            torch.max(torch.abs(stock_attention.to(torch.float32) - custom_outputs["attention_output"].to(torch.float32))).item(),
            6e-2,
        )
        self.assertLess(
            torch.max(torch.abs(stock_projected.to(torch.float32) - custom_outputs["projected_output"].to(torch.float32))).item(),
            6e-2,
        )
        self.assertLess(
            torch.max(torch.abs(stock_post_attn.to(torch.float32) - custom_outputs["post_attention_residual"].to(torch.float32))).item(),
            6e-2,
        )
        self.assertLess(
            torch.max(torch.abs(stock_mlp.to(torch.float32) - custom_outputs["mlp_output"].to(torch.float32))).item(),
            5e-2,
        )
        self.assertLess(
            torch.max(torch.abs(stock_decoder.to(torch.float32) - custom_decoder.to(torch.float32))).item(),
            1e-1,
        )

    def test_rack_path_disabled_matches_full_kv(self) -> None:
        config = _tiny_config()
        layer = _tiny_layer()
        hidden = _tiny_hidden(8)
        full_decoder, _full_outputs, _full_storage, _full_certs = _run_modified_layer_stream(
            method_name="full_kv",
            layer=layer,
            config=config,
            hidden_states=hidden,
            layer_index=0,
            recent_window=16,
            block_size=8,
            tolerance=0.05,
            precision=128,
            prompt_name="synthetic",
            certificate_records=[],
        )
        rack_decoder, _rack_outputs, _rack_storage, certs = _run_modified_layer_stream(
            method_name="rack_kv_compression_only",
            layer=layer,
            config=config,
            hidden_states=hidden,
            layer_index=0,
            recent_window=32,
            block_size=8,
            tolerance=0.05,
            precision=128,
            prompt_name="synthetic",
            certificate_records=[],
        )
        self.assertEqual(certs, [])
        self.assertLess(
            torch.max(torch.abs(full_decoder.to(torch.float32) - rack_decoder.to(torch.float32))).item(),
            1e-5,
        )

    def test_compression_only_incremental_path_matches_stateless_reference_tiny_case(self) -> None:
        config = _tiny_config()
        layer = _tiny_layer()
        hidden = _tiny_hidden(10)
        reference_decoder, reference_outputs, reference_storage, _ = _run_modified_layer_stream(
            method_name="rack_kv_compression_only",
            layer=layer,
            config=config,
            hidden_states=hidden,
            layer_index=0,
            recent_window=4,
            block_size=4,
            tolerance=0.05,
            precision=128,
            prompt_name="synthetic",
            certificate_records=[],
            token_chunk_size=10,
            use_incremental_rack_prefix=False,
        )
        incremental_decoder, incremental_outputs, incremental_storage, _ = _run_modified_layer_stream(
            method_name="rack_kv_compression_only",
            layer=layer,
            config=config,
            hidden_states=hidden,
            layer_index=0,
            recent_window=4,
            block_size=4,
            tolerance=0.05,
            precision=128,
            prompt_name="synthetic",
            certificate_records=[],
            token_chunk_size=1,
            use_incremental_rack_prefix=True,
        )
        self.assertTrue(torch.equal(reference_decoder, incremental_decoder))
        for key in reference_outputs:
            self.assertTrue(torch.equal(reference_outputs[key], incremental_outputs[key]))
        self.assertEqual(reference_storage, incremental_storage)

    def test_certified_mode_never_skips_without_mpfr_record(self) -> None:
        config = _tiny_config()
        layer = _tiny_layer()
        hidden = _tiny_hidden(12)
        certificate_records: list[dict] = []
        _decoder, _outputs, _storage, local_records = _run_modified_layer_stream(
            method_name="rack_kv_certified",
            layer=layer,
            config=config,
            hidden_states=hidden,
            layer_index=0,
            recent_window=4,
            block_size=4,
            tolerance=0.05,
            precision=128,
            prompt_name="synthetic",
            certificate_records=certificate_records,
        )
        for record in local_records:
            if record["mpfr_certified_skipped_blocks"] > 0:
                self.assertTrue(record["mpfr_invoked"])
                self.assertGreater(len(record["skipped_block_starts"]), 0)

    def test_certified_attention_prepared_path_matches_array_path(self) -> None:
        prefix_keys = torch.randn((10, 4), dtype=torch.float32).to(torch.bfloat16)
        prefix_values = torch.randn((10, 4), dtype=torch.float32).to(torch.bfloat16)
        context = _prefix_context(
            prefix_keys_tensor=prefix_keys,
            prefix_values_tensor=prefix_values,
            record_index=9,
            kv_head_global=0,
            recent_window=4,
            include_float64_arrays=True,
        )
        payload = _build_stage5_rack_prefix_payload(
            prefix_context=context,
            recent_window=4,
            block_size=4,
            precision=128,
            retain_blocks=True,
        )
        self.assertTrue(payload.blocks)
        query = np.asarray(prefix_keys[-1].to(torch.float32).numpy(), dtype=np.float64)
        baseline_output, baseline_record = _certified_attention_for_head(
            query=query,
            reconstructed_full_keys=payload.reconstructed_keys,
            reconstructed_full_values=payload.reconstructed_values,
            prefix_result=payload,
            recent_keys=payload.recent_keys,
            recent_values=payload.recent_values,
            tolerance=0.05,
            precision=128,
            scaling=0.5,
        )
        prepared_inputs = prepare_certification_inputs(
            payload.recent_keys,
            payload.recent_values,
            payload.blocks,
            precision=128,
        )
        prepared_full_rows = prepared_full_exact_rows(prepared_inputs)
        prepared_output, prepared_record = _certified_attention_for_head(
            query=query,
            reconstructed_full_keys=payload.reconstructed_keys,
            reconstructed_full_values=payload.reconstructed_values,
            prefix_result=payload,
            recent_keys=payload.recent_keys,
            recent_values=payload.recent_values,
            tolerance=0.05,
            precision=128,
            scaling=0.5,
            prepared_inputs=prepared_inputs,
            prepared_full_rows=prepared_full_rows,
        )
        np.testing.assert_allclose(baseline_output, prepared_output, rtol=0.0, atol=0.0)
        self.assertEqual(baseline_record, prepared_record)

    def test_zero_skip_certified_output_equals_compression_only_shared_path(self) -> None:
        prefix_keys = torch.randn((10, 4), dtype=torch.float32).to(torch.bfloat16)
        prefix_values = torch.randn((10, 4), dtype=torch.float32).to(torch.bfloat16)
        context = _prefix_context(
            prefix_keys_tensor=prefix_keys,
            prefix_values_tensor=prefix_values,
            record_index=9,
            kv_head_global=0,
            recent_window=4,
            include_float64_arrays=True,
        )
        payload = _build_stage5_rack_prefix_payload(
            prefix_context=context,
            recent_window=4,
            block_size=4,
            precision=128,
            retain_blocks=True,
        )
        query = np.asarray(prefix_keys[-1].to(torch.float32).numpy(), dtype=np.float64)
        compression_only = _shared_reconstructed_attention_output_numpy(
            query=query,
            keys=payload.reconstructed_keys,
            values=payload.reconstructed_values,
            scaling=0.5,
        )
        certified_output, record = _certified_attention_for_head(
            query=query,
            reconstructed_full_keys=payload.reconstructed_keys,
            reconstructed_full_values=payload.reconstructed_values,
            prefix_result=payload,
            recent_keys=payload.recent_keys,
            recent_values=payload.recent_values,
            tolerance=0.0,
            precision=128,
            scaling=0.5,
        )
        np.testing.assert_allclose(certified_output, compression_only, rtol=0.0, atol=0.0)
        self.assertEqual(record["mpfr_certified_skipped_blocks"], 0)

    def test_mpfr_rejection_matches_compression_only_and_cannot_skip(self) -> None:
        prefix_keys = torch.randn((10, 4), dtype=torch.float32).to(torch.bfloat16)
        prefix_values = torch.randn((10, 4), dtype=torch.float32).to(torch.bfloat16)
        context = _prefix_context(
            prefix_keys_tensor=prefix_keys,
            prefix_values_tensor=prefix_values,
            record_index=9,
            kv_head_global=0,
            recent_window=4,
            include_float64_arrays=True,
        )
        payload = _build_stage5_rack_prefix_payload(
            prefix_context=context,
            recent_window=4,
            block_size=4,
            precision=128,
            retain_blocks=True,
        )
        query = np.asarray(prefix_keys[-1].to(torch.float32).numpy(), dtype=np.float64)
        compression_only = _shared_reconstructed_attention_output_numpy(
            query=query,
            keys=payload.reconstructed_keys,
            values=payload.reconstructed_values,
            scaling=0.5,
        )
        rejection = SimpleNamespace(
            chosen_certificate=None,
            certificate_value_text=None,
            certificate_value_upper_float=None,
            decoded_block_starts=[int(payload.blocks[0].header.block_start)],
            skipped_block_starts=[],
            numerical_fallback_used=False,
        )
        with mock.patch("rack_kv.stage5.certify_progressive_skipping", return_value=rejection):
            certified_output, record = _certified_attention_for_head(
                query=query,
                reconstructed_full_keys=payload.reconstructed_keys,
                reconstructed_full_values=payload.reconstructed_values,
                prefix_result=payload,
                recent_keys=payload.recent_keys,
                recent_values=payload.recent_values,
                tolerance=1.0,
                precision=128,
                scaling=0.5,
                prefilter_bound_override=0.0,
            )
        np.testing.assert_allclose(certified_output, compression_only, rtol=0.0, atol=0.0)
        self.assertTrue(record["mpfr_invoked"])
        self.assertEqual(record["mpfr_certified_skipped_blocks"], 0)
        self.assertEqual(record["mpfr_rejected_candidates"], 1)
        self.assertEqual(record["skipped_block_starts"], [])

    def test_aggregate_certificate_records_distinguishes_prefilter_and_mpfr(self) -> None:
        records = [
            {
                "prompt_name": "synthetic",
                "layer_index": 0,
                "token_index": 16,
                "query_head_global": 1,
                "kv_head_global": 0,
                "eligible_blocks": 1,
                "mpfr_invoked": False,
                "fast_prefilter_rejected_blocks": 1,
                "mpfr_rejected_skip_candidates": 1,
                "mpfr_certified_skipped_blocks": 0,
                "potential_skip_candidates_sent_to_mpfr": 0,
                "rigorous_interval_violation": 0,
                "approximate_observed_violation": 0,
                "false_safe_count": 0,
                "numerical_fallback_used": 0,
                "skipped_block_starts": [],
            },
            {
                "prompt_name": "synthetic",
                "layer_index": 0,
                "token_index": 16,
                "query_head_global": 0,
                "kv_head_global": 0,
                "eligible_blocks": 1,
                "mpfr_invoked": True,
                "fast_prefilter_rejected_blocks": 0,
                "mpfr_rejected_skip_candidates": 0,
                "mpfr_certified_skipped_blocks": 1,
                "potential_skip_candidates_sent_to_mpfr": 1,
                "rigorous_interval_violation": 0,
                "approximate_observed_violation": 0,
                "false_safe_count": 0,
                "numerical_fallback_used": 0,
                "certificate_bound_text": "0.1",
                "skipped_block_starts": [0],
                "decoded_block_starts": [],
            },
            {
                "prompt_name": "synthetic",
                "layer_index": 0,
                "token_index": 16,
                "query_head_global": 2,
                "kv_head_global": 0,
                "eligible_blocks": 1,
                "mpfr_invoked": True,
                "fast_prefilter_rejected_blocks": 0,
                "mpfr_rejected_skip_candidates": 0,
                "mpfr_certified_skipped_blocks": 1,
                "potential_skip_candidates_sent_to_mpfr": 1,
                "rigorous_interval_violation": 0,
                "approximate_observed_violation": 0,
                "false_safe_count": 0,
                "numerical_fallback_used": 0,
                "certificate_bound_text": "0.1",
                "skipped_block_starts": [0],
                "decoded_block_starts": [],
            },
        ]
        aggregate = _aggregate_certificate_records(records, num_attention_heads=32, num_key_value_heads=8)
        self.assertEqual(aggregate["total_certificate_records"], 3)
        self.assertEqual(aggregate["total_eligible_query_head_block_decisions"], 3)
        self.assertEqual(aggregate["prefilter_rejected_query_head_block_decisions"], 1)
        self.assertEqual(aggregate["candidates_sent_to_mpfr"], 2)
        self.assertEqual(aggregate["mpfr_certified_query_head_block_skips"], 2)
        self.assertEqual(aggregate["mpfr_rejected_candidates"], 0)
        self.assertEqual(aggregate["unique_tokens_with_any_skip"], 1)
        self.assertEqual(aggregate["unique_query_heads_with_any_skip"], 2)
        self.assertEqual(aggregate["unique_logical_kv_blocks_with_any_skip"], 1)
        self.assertEqual(aggregate["gqa_physical_blocks_skippable_by_all_mapped_query_heads"], 0)
        self.assertEqual(aggregate["skips_lacking_proof_records"], 0)

    def test_future_token_changes_do_not_affect_earlier_outputs(self) -> None:
        config = _tiny_config()
        layer = _tiny_layer()
        hidden_a = _tiny_hidden(10)
        hidden_b = hidden_a.clone()
        hidden_b[7:, :] += torch.ones_like(hidden_b[7:, :], dtype=torch.bfloat16)
        for method_name in ("full_kv", "rack_kv_compression_only", "uniform_int8_kv", "rack_kv_certified"):
            dec_a, _out_a, _storage_a, _certs_a = _run_modified_layer_stream(
                method_name=method_name,
                layer=layer,
                config=config,
                hidden_states=hidden_a,
                layer_index=0,
                recent_window=4,
                block_size=4,
                tolerance=0.05,
                precision=128,
                prompt_name="a",
                certificate_records=[],
            )
            dec_b, _out_b, _storage_b, _certs_b = _run_modified_layer_stream(
                method_name=method_name,
                layer=layer,
                config=config,
                hidden_states=hidden_b,
                layer_index=0,
                recent_window=4,
                block_size=4,
                tolerance=0.05,
                precision=128,
                prompt_name="b",
                certificate_records=[],
            )
            self.assertLess(
                torch.max(torch.abs(dec_a[:6].to(torch.float32) - dec_b[:6].to(torch.float32))).item(),
                1e-5,
            )

    def test_compression_only_chunk_diagnostics_are_recorded(self) -> None:
        config = _tiny_config()
        layer = _tiny_layer()
        hidden = _tiny_hidden(10)
        diagnostics: list[dict] = []
        guard = MemoryGuard(max_rss_bytes=8 * 1024**3, min_free_bytes=0)
        _decoder, _outputs, _storage, _certs = _run_modified_layer_stream(
            method_name="rack_kv_compression_only",
            layer=layer,
            config=config,
            hidden_states=hidden,
            layer_index=0,
            recent_window=4,
            block_size=4,
            tolerance=0.05,
            precision=128,
            prompt_name="synthetic",
            certificate_records=[],
            memory_guard=guard,
            token_chunk_size=3,
            diagnostic_records=diagnostics,
        )
        stages = [record["stage"] for record in diagnostics]
        self.assertIn("enter_stream", stages)
        self.assertIn("after_token_chunk", stages)
        self.assertIn("after_chunk_cleanup", stages)
        chunk_records = [record for record in diagnostics if record["stage"] == "after_token_chunk"]
        self.assertTrue(any(int(record["serialization_buffer_bytes"]) > 0 for record in chunk_records))

    def test_manual_attention_matches_direct_formula(self) -> None:
        queries = torch.tensor([[1.0, -1.0], [0.5, 0.5]], dtype=torch.bfloat16)
        keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16)
        values = torch.tensor([[0.25, 0.5], [1.0, -0.5]], dtype=torch.bfloat16)
        actual = _manual_attention(queries=queries, keys=keys, values=values, scaling=0.5)
        scores = torch.matmul(queries.to(torch.float32), keys.to(torch.float32).transpose(0, 1)) * 0.5
        expected = torch.matmul(torch.softmax(scores, dim=-1), values.to(torch.float32)).to(torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))

    def test_prompt_corpus_is_stable_and_not_stage3_prompt(self) -> None:
        asset_dir = Path(".tmp/stage3_multilayer_full/capture/llama31_base_assets/1f47e50cdbe801ad8a5174156ec3a0655108fb9f")
        if not asset_dir.exists():
            self.skipTest("Stage 3 tokenizer assets are not present in this workspace.")
        with tempfile.TemporaryDirectory() as temp_dir_a, tempfile.TemporaryDirectory() as temp_dir_b:
            corpus_a = build_stage5_prompt_corpus(asset_dir=asset_dir, output_dir=Path(temp_dir_a), revision="1f47e50cdbe801ad8a5174156ec3a0655108fb9f")
            corpus_b = build_stage5_prompt_corpus(asset_dir=asset_dir, output_dir=Path(temp_dir_b), revision="1f47e50cdbe801ad8a5174156ec3a0655108fb9f")
            self.assertEqual(
                [prompt.name for prompt in corpus_a],
                [
                    "natural_language_128",
                    "passkey_retrieval_128",
                    "structured_reasoning_128",
                    "code_context_128",
                ],
            )
            self.assertEqual([prompt.source_sha256 for prompt in corpus_a], [prompt.source_sha256 for prompt in corpus_b])
            self.assertEqual([prompt.token_ids for prompt in corpus_a], [prompt.token_ids for prompt in corpus_b])
            self.assertTrue(all(prompt.token_count == STAGE5_PROMPT_TOKEN_COUNT for prompt in corpus_a))
            stage3_prompt_path = Path(".tmp/stage2b_layer0_pilot/prompt/stage2b_prompt_source.txt")
            if stage3_prompt_path.exists():
                stage3_text = stage3_prompt_path.read_text(encoding="utf-8")
                for prompt in corpus_a:
                    self.assertNotEqual(prompt.source_text.strip(), stage3_text.strip())

    def test_prompt_corpus_supports_reduced_token_count(self) -> None:
        asset_dir = Path(".tmp/stage3_multilayer_full/capture/llama31_base_assets/1f47e50cdbe801ad8a5174156ec3a0655108fb9f")
        if not asset_dir.exists():
            self.skipTest("Stage 3 tokenizer assets are not present in this workspace.")
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = build_stage5_prompt_corpus(
                asset_dir=asset_dir,
                output_dir=Path(temp_dir),
                revision="1f47e50cdbe801ad8a5174156ec3a0655108fb9f",
                target_token_count=32,
                prompt_names=("natural_language_128",),
            )
            self.assertTrue(all(prompt.token_count == 32 for prompt in corpus))

    def test_normalize_stage5_methods_adds_full_kv_for_partial_runs(self) -> None:
        self.assertEqual(
            _normalize_stage5_methods(("rack_kv_certified", "uniform_int8_kv")),
            ("full_kv", "rack_kv_certified", "uniform_int8_kv"),
        )
        self.assertEqual(
            _normalize_stage5_methods(("full_kv", "rack_kv_certified")),
            ("full_kv", "rack_kv_certified"),
        )

    def test_prompt_checkpoint_roundtrip_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = {"schema": STAGE5_PROMPT_CHECKPOINT_SCHEMA, "value": 17}
            _save_prompt_checkpoint(root, "natural_language_128", payload)
            loaded = _load_prompt_checkpoint(root, "natural_language_128")
            self.assertEqual(payload, loaded)

    def test_stream_checkpoint_settings_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompt = type("Prompt", (), {"name": "natural_language_128", "source_sha256": "abc", "token_count": 128})()
            settings = _stream_settings_payload(
                method_name="full_kv",
                prompt=prompt,
                repo_id="repo",
                repo_revision="rev",
                recent_window=16,
                block_size=8,
                tolerance=0.05,
                precision=256,
                seed=0,
                total_layers_to_run=32,
            )
            hidden_path = _stream_hidden_state_path(root, "natural_language_128", "full_kv")
            _save_tensor_artifact(hidden_path, {"hidden_states": torch.zeros((2, 4), dtype=torch.bfloat16)})
            _save_stream_checkpoint(
                root,
                "natural_language_128",
                "full_kv",
                {
                    "schema": STAGE5_STREAM_CHECKPOINT_SCHEMA,
                    "status": "in_progress",
                    "method_name": "full_kv",
                    "settings": settings,
                    "next_layer_index": 3,
                    "hidden_state_path": str(hidden_path),
                },
            )
            loaded = _load_stream_checkpoint(
                root,
                "natural_language_128",
                "full_kv",
                expected_settings=settings,
            )
            self.assertIsNotNone(loaded)
            mismatched = dict(settings)
            mismatched["precision"] = 128
            rejected = _load_stream_checkpoint(
                root,
                "natural_language_128",
                "full_kv",
                expected_settings=mismatched,
            )
            self.assertIsNone(rejected)

    def test_stream_checkpoint_missing_method_implementation_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompt = type("Prompt", (), {"name": "natural_language_128", "source_sha256": "abc", "token_count": 128})()
            settings = _stream_settings_payload(
                method_name="full_kv",
                prompt=prompt,
                repo_id="repo",
                repo_revision="rev",
                recent_window=16,
                block_size=8,
                tolerance=0.05,
                precision=256,
                seed=0,
                total_layers_to_run=1,
            )
            hidden_path = _stream_hidden_state_path(root, "natural_language_128", "full_kv")
            _save_tensor_artifact(hidden_path, {"hidden_states": torch.zeros((2, 4), dtype=torch.bfloat16)})
            legacy_settings = dict(settings)
            legacy_settings.pop("method_implementation_version", None)
            _save_stream_checkpoint(
                root,
                "natural_language_128",
                "full_kv",
                {
                    "schema": STAGE5_STREAM_CHECKPOINT_SCHEMA,
                    "status": "complete",
                    "method_name": "full_kv",
                    "settings": legacy_settings,
                    "next_layer_index": 1,
                    "hidden_state_path": str(hidden_path),
                },
            )
            self.assertIsNone(
                _load_stream_checkpoint(
                    root,
                    "natural_language_128",
                    "full_kv",
                    expected_settings=settings,
                )
            )

    def test_outputs_do_not_require_grad(self) -> None:
        config = _tiny_config()
        layer = _tiny_layer()
        hidden = _tiny_hidden(10)
        stock_outputs = _stock_layer_forward_with_intermediates(
            layer=layer,
            config=config,
            hidden_states=hidden,
        )
        self.assertTrue(all(not tensor.requires_grad for tensor in stock_outputs))
        decoder, outputs, _storage, _certs = _run_modified_layer_stream(
            method_name="uniform_int8_kv",
            layer=layer,
            config=config,
            hidden_states=hidden,
            layer_index=0,
            recent_window=4,
            block_size=4,
            tolerance=0.05,
            precision=128,
            prompt_name="synthetic",
            certificate_records=[],
        )
        self.assertFalse(decoder.requires_grad)
        self.assertTrue(all(not tensor.requires_grad for tensor in outputs.values()))

    def test_layer_object_is_releasable_after_use(self) -> None:
        layer = _tiny_layer()
        layer_ref = weakref.ref(layer)
        del layer
        import gc
        gc.collect()
        self.assertIsNone(layer_ref())

    def test_metric_records_do_not_embed_full_logit_vectors(self) -> None:
        logits = torch.randn(40, 32, dtype=torch.float32)
        records, aggregate, _ = _prompt_method_metrics(
            prompt=type("Prompt", (), {"token_ids": tuple(index % 32 for index in range(40)), "answer_token_ids": None, "final_answer_start_token_index": None})(),
            method_name="full_kv",
            logits=logits,
            full_logits=logits,
        )
        self.assertIsInstance(aggregate, dict)
        self.assertTrue(records)
        record = records[0]
        self.assertNotIn("full_logits", record)
        self.assertNotIn("method_logits", record)

    def test_memory_guard_raises_on_intentionally_tiny_limit(self) -> None:
        guard = MemoryGuard(max_rss_bytes=1, min_free_bytes=0)
        with self.assertRaises(Exception):
            guard.check("synthetic_over_limit")

    def test_source_snapshot_uses_actual_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.py").write_text("print('a')\n", encoding="utf-8")
            (root / "b.py").write_text("print('b')\n", encoding="utf-8")
            expected_lines = [
                f"a.py\t{hashlib.sha256((root / 'a.py').read_bytes()).hexdigest()}",
                f"b.py\t{hashlib.sha256((root / 'b.py').read_bytes()).hexdigest()}",
            ]
            expected = hashlib.sha256("\n".join(expected_lines).encode("utf-8")).hexdigest()
            actual = _source_snapshot_sha256(base_dir=root, relative_paths=("a.py", "b.py"))
            self.assertEqual(expected, actual)

    def test_tensor_cache_rejects_corrupted_entry(self) -> None:
        payload = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16).tobytes(order="C")
        fetches: list[tuple[int, int]] = []

        def _fetcher(_url: str, start: int, end_inclusive: int) -> bytes:
            fetches.append((start, end_inclusive))
            return payload

        with tempfile.TemporaryDirectory() as temp_dir:
            cache = PersistentTensorRangeCache(
                root=Path(temp_dir),
                repo_id="synthetic/repo",
                revision="synthetic-revision",
                download_policy=type("Policy", (), {"verify_tls": True})(),  # simple stub
                range_fetcher=_fetcher,
            )
            spec = _RemoteTensorSpec(dtype_code="F16", shape=(2, 2), byte_start=0, byte_end_exclusive=len(payload))
            cache._shards["model-00001-of-00001.safetensors"] = _FakeShard(spec=spec)
            weight_index = {"weight_map": {"tensor.weight": "model-00001-of-00001.safetensors"}}
            first = cache.fetch_tensor(weight_index=weight_index, tensor_name="tensor.weight")
            self.assertEqual(len(fetches), 1)
            payload_path = next(Path(temp_dir).rglob("*.bin"))
            payload_path.write_bytes(b"corrupt")
            second = cache.fetch_tensor(weight_index=weight_index, tensor_name="tensor.weight")
            self.assertEqual(len(fetches), 2)
            self.assertTrue(torch.equal(first.to(torch.float32), second.to(torch.float32)))

    def test_stage4_artifacts_hash_check_is_stable(self) -> None:
        before = _stage4_artifact_hashes(Path.cwd())
        after = _stage4_artifact_hashes(Path.cwd())
        _validate_stage4_artifacts_unchanged(before, after)
        self.assertEqual(before, after)

    def test_prefix_context_counts_visible_recent_and_historical_tokens(self) -> None:
        prefix_keys = torch.zeros((10, 4), dtype=torch.bfloat16)
        prefix_values = torch.zeros((10, 4), dtype=torch.bfloat16)
        context = _prefix_context(
            prefix_keys_tensor=prefix_keys,
            prefix_values_tensor=prefix_values,
            record_index=9,
            kv_head_global=0,
            recent_window=4,
        )
        self.assertEqual(context.visible_length, 10)
        self.assertEqual(context.recent_exact_tokens, 4)
        self.assertEqual(context.historical_tokens, 6)

    def test_rack_prefix_payload_works_without_prefix_float64_copies(self) -> None:
        prefix_keys = torch.randn((10, 4), dtype=torch.float32).to(torch.bfloat16)
        prefix_values = torch.randn((10, 4), dtype=torch.float32).to(torch.bfloat16)
        context = _prefix_context(
            prefix_keys_tensor=prefix_keys,
            prefix_values_tensor=prefix_values,
            record_index=9,
            kv_head_global=0,
            recent_window=4,
            include_float64_arrays=False,
        )
        self.assertIsNone(context.prefix_keys)
        self.assertIsNone(context.prefix_values)
        payload = _build_stage5_rack_prefix_payload(
            prefix_context=context,
            recent_window=4,
            block_size=4,
            precision=128,
            retain_blocks=False,
        )
        self.assertEqual(payload.blocks, tuple())
        self.assertEqual(payload.reconstructed_keys.shape, (10, 4))
        self.assertEqual(payload.reconstructed_values.shape, (10, 4))

    def test_layer0_storage_summary_category_sum_matches_total_on_historical_package(self) -> None:
        output_dir = Path(".tmp/stage5_quality_validation_layer0")
        checkpoint_root = output_dir / "checkpoints"
        if not checkpoint_root.exists():
            self.skipTest("Historical Stage 5 layer-0 validation package is not present in this workspace.")
        checkpoint_summary = json.loads((output_dir / "stage5_quality_final_results.json").read_text(encoding="utf-8"))["checkpoint_summary"]
        for method_name in ("full_kv", "rack_kv_compression_only", "rack_kv_certified", "uniform_int8_kv"):
            payload = json.loads((checkpoint_root / f"natural_language_128__{method_name}.json").read_text(encoding="utf-8"))
            storage = _layer0_storage_summary(
                method_name=method_name,
                checkpoint_payload=payload,
                checkpoint_summary=checkpoint_summary,
            )
            self.assertEqual(sum(storage["category_sums"].values()), sum(storage["token_totals_bytes"]))
            if method_name in {"rack_kv_compression_only", "rack_kv_certified"}:
                self.assertEqual(storage["category_sums"]["certificate_metadata_bytes"], 0)

    def test_historical_layer0_certificate_counts_match_reviewed_records(self) -> None:
        cert_path = Path(".tmp/stage5_quality_validation_layer0/certificate_records/natural_language_128.json")
        if not cert_path.exists():
            self.skipTest("Historical Stage 5 layer-0 certificate records are not present in this workspace.")
        records = json.loads(cert_path.read_text(encoding="utf-8"))["records"]
        aggregate = _aggregate_certificate_records(records, num_attention_heads=32, num_key_value_heads=8)
        self.assertEqual(aggregate["total_certificate_records"], 3584)
        self.assertEqual(aggregate["candidates_sent_to_mpfr"], 4)
        self.assertEqual(aggregate["mpfr_certified_query_head_block_skips"], 4)
        self.assertEqual(aggregate["mpfr_rejected_candidates"], 0)
        self.assertEqual(aggregate["unique_tokens_with_any_skip"], 1)
        self.assertEqual(aggregate["unique_query_heads_with_any_skip"], 4)
        self.assertEqual(aggregate["unique_logical_kv_blocks_with_any_skip"], 3)
        self.assertEqual(aggregate["gqa_physical_blocks_skippable_by_all_mapped_query_heads"], 0)
        self.assertEqual(aggregate["physical_block_decodes_actually_avoided"], 0)

    def test_runtime_storage_accumulation_uses_disjoint_categories_for_compression_only(self) -> None:
        config = _tiny_config()
        layer = _tiny_layer()
        hidden = _tiny_hidden(12)
        _decoder, _outputs, layer_storage, _certs = _run_modified_layer_stream(
            method_name="rack_kv_compression_only",
            layer=layer,
            config=config,
            hidden_states=hidden,
            layer_index=0,
            recent_window=4,
            block_size=4,
            tolerance=0.05,
            precision=128,
            prompt_name="synthetic",
            certificate_records=[],
            token_chunk_size=2,
            use_incremental_rack_prefix=True,
        )
        token_totals = [0 for _ in range(hidden.shape[0])]
        category_sums = _empty_storage_category_sums()
        _accumulate_stage5_storage_records(
            method_name="rack_kv_compression_only",
            storage_records=layer_storage,
            storage_modified_token_totals=token_totals,
            storage_category_sums=category_sums,
            head_dim=4,
            value_dim=4,
            recent_window=4,
            block_size=4,
            precision=128,
        )
        self.assertEqual(category_sums["certificate_metadata_bytes"], 0)
        self.assertEqual(sum(category_sums.values()), sum(token_totals))
        self.assertGreater(category_sums["block_metadata_bytes"], 0)
        self.assertGreater(category_sums["container_header_bytes"], 0)
        self.assertGreater(category_sums["anchor_bytes"], 0)

    def test_runtime_storage_accumulation_uses_disjoint_categories_for_certified(self) -> None:
        config = _tiny_config()
        layer = _tiny_layer()
        hidden = _tiny_hidden(12)
        cert_records: list[dict] = []
        _decoder, _outputs, layer_storage, _local_certs = _run_modified_layer_stream(
            method_name="rack_kv_certified",
            layer=layer,
            config=config,
            hidden_states=hidden,
            layer_index=0,
            recent_window=4,
            block_size=4,
            tolerance=0.05,
            precision=128,
            prompt_name="synthetic",
            certificate_records=cert_records,
            token_chunk_size=1,
            use_incremental_rack_prefix=True,
        )
        token_totals = [0 for _ in range(hidden.shape[0])]
        category_sums = _empty_storage_category_sums()
        _accumulate_stage5_storage_records(
            method_name="rack_kv_certified",
            storage_records=layer_storage,
            storage_modified_token_totals=token_totals,
            storage_category_sums=category_sums,
            head_dim=4,
            value_dim=4,
            recent_window=4,
            block_size=4,
            precision=128,
        )
        self.assertEqual(category_sums["certificate_metadata_bytes"], 0)
        self.assertEqual(sum(category_sums.values()), sum(token_totals))
        self.assertGreaterEqual(category_sums["block_metadata_bytes"], 0)
        self.assertGreater(category_sums["container_header_bytes"], 0)

    def test_method_storage_summary_uses_one_authoritative_scope_for_runtime_records(self) -> None:
        sequence_length = 5
        modified_layers = (0, 2)
        total_layers = 4
        num_kv_heads = 2
        head_dim = 4
        records = _synthetic_rack_storage_records(
            sequence_length=sequence_length,
            modified_layers=modified_layers,
            num_kv_heads=num_kv_heads,
            recent_window=2,
            block_size=2,
            head_dim=head_dim,
            value_dim=head_dim,
            precision=128,
        )
        token_totals = [0 for _ in range(sequence_length)]
        category_sums = _empty_storage_category_sums()
        _accumulate_stage5_storage_records(
            method_name="rack_kv_compression_only",
            storage_records=records,
            storage_modified_token_totals=token_totals,
            storage_category_sums=category_sums,
            head_dim=head_dim,
            value_dim=head_dim,
            recent_window=2,
            block_size=2,
            precision=128,
        )
        summary_from_records = _method_storage_summary(
            method_name="rack_kv_compression_only",
            storage_records=records,
            total_layers=total_layers,
            modified_layers=modified_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sequence_length=sequence_length,
            recent_window=2,
            block_size=2,
            precision=128,
        )
        summary_from_checkpoint = _method_storage_summary(
            method_name="rack_kv_compression_only",
            modified_token_totals=token_totals,
            category_sums_override=category_sums,
            total_layers=total_layers,
            modified_layers=modified_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sequence_length=sequence_length,
            recent_window=2,
            block_size=2,
            precision=128,
        )
        self.assertEqual(summary_from_records["storage_accounting_schema"], STAGE5_STORAGE_ACCOUNTING_SCHEMA)
        self.assertEqual(summary_from_records["modified_layers_token_totals_bytes"], token_totals)
        self.assertEqual(summary_from_records["category_sums"], category_sums)
        self.assertEqual(summary_from_records["token_totals_bytes"], summary_from_checkpoint["token_totals_bytes"])
        self.assertEqual(
            summary_from_records["modified_layers_cumulative_total_serialized_bytes"],
            sum(token_totals),
        )
        self.assertEqual(
            summary_from_records["disjoint_category_total_bytes"],
            sum(category_sums.values()),
        )
        self.assertEqual(
            summary_from_records["final_total_bytes"],
            summary_from_records["final_modified_layers_serialized_bytes"]
            + summary_from_records["unmodified_exact_token_totals_bytes"][-1],
        )
        self.assertTrue(summary_from_records["byte_accounting_consistent"])

    def test_method_storage_summary_rejects_mixed_scope_checkpoint_inputs(self) -> None:
        sequence_length = 4
        token_totals = [10, 20, 30, 40]
        category_sums = _empty_storage_category_sums()
        category_sums["encoded_key_bytes"] = 99
        category_sums["encoded_value_bytes"] = 2
        with self.assertRaisesRegex(Stage5ExecutionError, "modified-layer category total"):
            _method_storage_summary(
                method_name="rack_kv_compression_only",
                modified_token_totals=token_totals,
                category_sums_override=category_sums,
                total_layers=3,
                modified_layers=(0,),
                num_kv_heads=2,
                head_dim=4,
                sequence_length=sequence_length,
            )

    def test_method_storage_summary_certified_keeps_certificate_metadata_disjoint(self) -> None:
        sequence_length = 4
        modified_layers = (0,)
        num_kv_heads = 2
        head_dim = 4
        records = _synthetic_rack_storage_records(
            sequence_length=sequence_length,
            modified_layers=modified_layers,
            num_kv_heads=num_kv_heads,
            recent_window=2,
            block_size=2,
            head_dim=head_dim,
            value_dim=head_dim,
            precision=128,
        )
        summary = _method_storage_summary(
            method_name="rack_kv_certified",
            storage_records=records,
            total_layers=3,
            modified_layers=modified_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sequence_length=sequence_length,
            recent_window=2,
            block_size=2,
            precision=128,
        )
        self.assertEqual(summary["category_sums"]["certificate_metadata_bytes"], 0)
        self.assertEqual(
            summary["disjoint_category_total_bytes"],
            summary["modified_layers_cumulative_total_serialized_bytes"],
        )

    def test_storage_summary_repair_rebuilds_existing_checkpoints_without_model_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (output_dir / "provenance").mkdir(parents=True, exist_ok=True)
            prompt = SimpleNamespace(name="synthetic_prompt", source_sha256="abc123", token_count=5)
            full_settings = _stream_settings_payload(
                method_name="full_kv",
                prompt=prompt,
                repo_id="repo",
                repo_revision="rev",
                recent_window=2,
                block_size=2,
                tolerance=0.05,
                precision=128,
                seed=0,
                total_layers_to_run=4,
            )
            compression_settings = _stream_settings_payload(
                method_name="rack_kv_compression_only",
                prompt=prompt,
                repo_id="repo",
                repo_revision="rev",
                recent_window=2,
                block_size=2,
                tolerance=0.05,
                precision=128,
                seed=0,
                total_layers_to_run=4,
            )
            full_hidden_path = _stream_hidden_state_path(output_dir, prompt.name, "full_kv")
            compression_hidden_path = _stream_hidden_state_path(output_dir, prompt.name, "rack_kv_compression_only")
            _save_tensor_artifact(full_hidden_path, {"hidden_states": torch.zeros((5, 8), dtype=torch.bfloat16)})
            _save_tensor_artifact(compression_hidden_path, {"hidden_states": torch.zeros((5, 8), dtype=torch.bfloat16)})

            full_storage = _method_storage_summary(
                method_name="full_kv",
                total_layers=4,
                modified_layers=(0, 2),
                num_kv_heads=2,
                head_dim=4,
                sequence_length=5,
                recent_window=2,
                block_size=2,
                precision=128,
            )
            full_checkpoint = {
                "schema": STAGE5_STREAM_CHECKPOINT_SCHEMA,
                "status": "complete",
                "method_name": "full_kv",
                "settings": full_settings,
                "next_layer_index": 4,
                "hidden_state_path": str(full_hidden_path),
                "runtime_s": 1.0,
                "method_aggregate": {"storage": full_storage},
            }
            (output_dir / "checkpoints" / f"{prompt.name}__full_kv.json").write_text(
                json.dumps(full_checkpoint, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            records = _synthetic_rack_storage_records(
                sequence_length=5,
                modified_layers=(0, 2),
                num_kv_heads=2,
                recent_window=2,
                block_size=2,
                head_dim=4,
                value_dim=4,
                precision=128,
            )
            token_totals = [0 for _ in range(5)]
            category_sums = _empty_storage_category_sums()
            _accumulate_stage5_storage_records(
                method_name="rack_kv_compression_only",
                storage_records=records,
                storage_modified_token_totals=token_totals,
                storage_category_sums=category_sums,
                head_dim=4,
                value_dim=4,
                recent_window=2,
                block_size=2,
                precision=128,
            )
            compression_checkpoint = {
                "schema": STAGE5_STREAM_CHECKPOINT_SCHEMA,
                "status": "in_progress",
                "method_name": "rack_kv_compression_only",
                "settings": compression_settings,
                "next_layer_index": 4,
                "hidden_state_path": str(compression_hidden_path),
                "runtime_s": 2.0,
                "storage_modified_token_totals": token_totals,
                "storage_category_sums": category_sums,
                "certificate_records": [],
            }
            compression_checkpoint_path = output_dir / "checkpoints" / f"{prompt.name}__rack_kv_compression_only.json"
            compression_checkpoint_path.write_text(
                json.dumps(compression_checkpoint, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            repair_summary = _repair_storage_summary_from_checkpoints(output_dir=output_dir)
            self.assertEqual(repair_summary["model_forward_calls"], 0)
            self.assertEqual(repair_summary["remote_tensor_fetches"], 0)
            self.assertEqual(repair_summary["hidden_state_recomputations"], 0)
            self.assertEqual(repair_summary["mpfr_recomputations"], 0)
            self.assertTrue((output_dir / "accounting_repair_backup" / "checkpoints" / f"{prompt.name}__rack_kv_compression_only.json").exists())

            repaired_payload = json.loads(compression_checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(repaired_payload["storage_accounting_schema"], STAGE5_STORAGE_ACCOUNTING_SCHEMA)
            self.assertIn("reporting_only_repair", repaired_payload)
            self.assertEqual(
                repaired_payload["reporting_only_repair"]["repaired_storage_summary"]["disjoint_category_total_bytes"],
                repaired_payload["reporting_only_repair"]["repaired_storage_summary"]["modified_layers_cumulative_total_serialized_bytes"],
            )
            reloaded = _load_stream_checkpoint(
                output_dir,
                prompt.name,
                "rack_kv_compression_only",
                expected_settings=compression_settings,
            )
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded["status"], "in_progress")

    def test_finalize_from_checkpoints_builds_two_prompt_package_without_model_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            review_zip = output_dir / "review.zip"
            for relative in ("checkpoints", "metric_records", "certificate_records", "prompt_corpus", "provenance"):
                (output_dir / relative).mkdir(parents=True, exist_ok=True)
            (output_dir / "test_log.txt").write_text("Ran 1 tests in 0.001s\n\nOK\n", encoding="utf-8")
            (output_dir / "test_command.txt").write_text("python -m unittest tests.test_stage5_quality\n", encoding="utf-8")
            (output_dir / "test_return_code.txt").write_text("0\n", encoding="utf-8")
            (output_dir / "stage5_full_run_log.txt").write_text("synthetic finalize fixture\n", encoding="utf-8")

            required_prompts = ("natural_language_128", "passkey_retrieval_128")
            excluded_prompt = "structured_reasoning_128"
            methods = ("full_kv", "rack_kv_compression_only", "rack_kv_certified", "uniform_int8_kv")
            token_count = 40
            expected_scored = token_count - 32
            modified_layers = tuple(STAGE5_MODIFIED_LAYERS)

            def _write_prompt(prompt_name: str, *, include_passkey_metadata: bool) -> None:
                source_text = f"{prompt_name} synthetic prompt text.\n"
                source_sha = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
                source_path = output_dir / "prompt_corpus" / f"{prompt_name}.txt"
                token_ids_path = output_dir / "prompt_corpus" / f"{prompt_name}_token_ids.json"
                metadata_path = output_dir / "prompt_corpus" / f"{prompt_name}_metadata.json"
                source_path.write_text(source_text, encoding="utf-8")
                token_ids = list(range(token_count))
                token_ids_path.write_text(json.dumps(token_ids), encoding="utf-8")
                metadata = {
                    "schema": STAGE5_PROMPT_CORPUS_VERSION,
                    "name": prompt_name,
                    "source_sha256": source_sha,
                    "source_path": str(source_path),
                    "token_ids_path": str(token_ids_path),
                    "token_count": token_count,
                    "target_token_count": token_count,
                    "selection_rule": "synthetic_test",
                    "tokenizer_name": "synthetic",
                    "tokenizer_revision": "synthetic",
                }
                if include_passkey_metadata:
                    metadata.update(
                        {
                            "answer_text": "passkey-42",
                            "answer_token_ids": [7, 8],
                            "final_answer_start_token_index": 36,
                            "generation_seed": 20260723,
                        }
                    )
                metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

            for prompt_name in required_prompts:
                _write_prompt(prompt_name, include_passkey_metadata=(prompt_name == "passkey_retrieval_128"))
            _write_prompt(excluded_prompt, include_passkey_metadata=False)

            for prompt_name in required_prompts:
                prompt = SimpleNamespace(name=prompt_name, source_sha256="sha-" + prompt_name, token_count=token_count)
                full_storage = _method_storage_summary(
                    method_name="full_kv",
                    total_layers=32,
                    modified_layers=modified_layers,
                    num_kv_heads=2,
                    head_dim=4,
                    sequence_length=token_count,
                    recent_window=2,
                    block_size=2,
                    precision=128,
                )
                metric_records_by_method: dict[str, list[dict[str, object]]] = {}
                certificate_records: list[dict[str, object]] = [
                    {
                        "schema": "stage5_certificate_record_v2",
                        "prompt_name": prompt_name,
                        "layer_index": 0,
                        "token_index": 16,
                        "query_head_global": 0,
                        "kv_head_global": 0,
                        "eligible_blocks": 1,
                        "mpfr_invoked": True,
                        "fast_prefilter_rejected_blocks": 0,
                        "prefilter_rejected_decisions": 0,
                        "potential_skip_candidates_sent_to_mpfr": 1,
                        "mpfr_certified_skipped_blocks": 1,
                        "mpfr_rejected_candidates": 0,
                        "rigorous_interval_violation": 0,
                        "approximate_observed_violation": 0,
                        "false_safe_count": 0,
                        "numerical_fallback_used": 0,
                        "certificate_bound_text": "0.1",
                        "certificate_bound_upper_float": 0.1,
                        "rigorous_local_skipping_upper": 0.01,
                        "observed_local_skipping_error": 0.01,
                        "skipped_block_starts": [0],
                        "decoded_block_starts": [],
                    },
                    {
                        "schema": "stage5_certificate_record_v2",
                        "prompt_name": prompt_name,
                        "layer_index": 0,
                        "token_index": 16,
                        "query_head_global": 1,
                        "kv_head_global": 0,
                        "eligible_blocks": 1,
                        "mpfr_invoked": False,
                        "fast_prefilter_rejected_blocks": 1,
                        "prefilter_rejected_decisions": 1,
                        "potential_skip_candidates_sent_to_mpfr": 0,
                        "mpfr_certified_skipped_blocks": 0,
                        "mpfr_rejected_candidates": 0,
                        "rigorous_interval_violation": 0,
                        "approximate_observed_violation": 0,
                        "false_safe_count": 0,
                        "numerical_fallback_used": 0,
                        "skipped_block_starts": [],
                        "decoded_block_starts": [],
                    },
                ]
                for method_index, method_name in enumerate(methods):
                    metric_records = []
                    for local_index, token_index in enumerate(range(31, 31 + expected_scored)):
                        base_nll = 2.0 + 0.1 * local_index
                        if method_name == "full_kv":
                            metric_records.append(
                                _synthetic_metric_record(
                                    token_index=token_index,
                                    target_token_id=10 + local_index,
                                    nll=base_nll,
                                )
                            )
                        else:
                            delta = 0.01 * (method_index + 1)
                            metric_records.append(
                                _synthetic_metric_record(
                                    token_index=token_index,
                                    target_token_id=10 + local_index,
                                    nll=base_nll + delta,
                                    delta_nll_vs_full=delta,
                                    kl_divergence=0.001 * (method_index + 1),
                                    js_divergence=0.0005 * (method_index + 1),
                                    logit_l2_error=0.01 * (method_index + 1),
                                    relative_logit_l2_error=0.005 * (method_index + 1),
                                    max_abs_logit_component_error=0.02 * (method_index + 1),
                                    cosine_similarity=0.999 - 0.001 * method_index,
                                )
                            )
                    metric_records_by_method[method_name] = metric_records

                    settings = _stream_settings_payload(
                        method_name=method_name,
                        prompt=prompt,
                        repo_id="repo",
                        repo_revision="rev",
                        recent_window=2,
                        block_size=2,
                        tolerance=0.05,
                        precision=128,
                        seed=0,
                        total_layers_to_run=32,
                    )
                    hidden_path = _stream_hidden_state_path(output_dir, prompt_name, method_name)
                    _save_tensor_artifact(hidden_path, {"hidden_states": torch.zeros((token_count, 8), dtype=torch.bfloat16)})
                    payload = {
                        "schema": STAGE5_STREAM_CHECKPOINT_SCHEMA,
                        "status": "complete",
                        "method_name": method_name,
                        "settings": settings,
                        "next_layer_index": 32,
                        "hidden_state_path": str(hidden_path),
                        "runtime_s": 1.0 + method_index,
                        "metric_records": metric_records,
                    }
                    if method_name == "full_kv":
                        logits_path = output_dir / "checkpoints" / f"{prompt_name}__full_kv__logits.safetensors"
                        logits_path.write_bytes(b"synthetic")
                        payload["full_logits_path"] = str(logits_path)
                        payload["method_aggregate"] = {"storage": full_storage}
                        payload["certificate_records"] = []
                        payload["passkey_metrics"] = (
                            {
                                "teacher_forced_answer_log_probability": -0.9,
                                "teacher_forced_answer_log_probability_full_kv": -0.9,
                                "delta_answer_log_probability_vs_full": 0.0,
                                "first_answer_token_rank": 1,
                                "first_answer_token_is_top1": True,
                                "first_answer_token_in_top5": True,
                            }
                            if prompt_name == "passkey_retrieval_128"
                            else None
                        )
                    else:
                        if method_name in {"rack_kv_compression_only", "rack_kv_certified"}:
                            records = _synthetic_rack_storage_records(
                                sequence_length=token_count,
                                modified_layers=modified_layers,
                                num_kv_heads=2,
                                recent_window=2,
                                block_size=2,
                                head_dim=4,
                                value_dim=4,
                                precision=128,
                            )
                            token_totals = [0 for _ in range(token_count)]
                            category_sums = _empty_storage_category_sums()
                            _accumulate_stage5_storage_records(
                                method_name=method_name,
                                storage_records=records,
                                storage_modified_token_totals=token_totals,
                                storage_category_sums=category_sums,
                                head_dim=4,
                                value_dim=4,
                                recent_window=2,
                                block_size=2,
                                precision=128,
                            )
                        else:
                            token_totals = [200 * (index + 1) for index in range(token_count)]
                            category_total = sum(token_totals)
                            category_sums = _empty_storage_category_sums()
                            category_sums["encoded_key_bytes"] = 60000
                            category_sums["encoded_value_bytes"] = 60000
                            category_sums["quantization_scale_bytes"] = 20000
                            category_sums["recent_window_bytes"] = category_total - 140000
                        payload["storage_modified_token_totals"] = token_totals
                        payload["storage_category_sums"] = category_sums
                        payload["intermediate_layer_metrics"] = {
                            "final_hidden_state": {"max_l2_difference": 0.1},
                            "final_logits": {"max_l2_difference": 0.2},
                        }
                        payload["certificate_records"] = certificate_records if method_name == "rack_kv_certified" else []
                        payload["passkey_metrics"] = (
                            {
                                "teacher_forced_answer_log_probability": -1.0,
                                "delta_answer_log_probability_vs_full": -0.1,
                                "first_answer_token_rank": 2,
                                "first_answer_token_is_top1": False,
                                "first_answer_token_in_top5": True,
                            }
                            if prompt_name == "passkey_retrieval_128"
                            else None
                        )
                    _save_stream_checkpoint(output_dir, prompt_name, method_name, payload)

                (output_dir / "metric_records" / f"{prompt_name}.json").write_text(
                    json.dumps(
                        {
                            "schema": "stage5_metric_record_v1",
                            "prompt_name": prompt_name,
                            "records_by_method": metric_records_by_method,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                (output_dir / "certificate_records" / f"{prompt_name}.json").write_text(
                    json.dumps(
                        {
                            "schema": "stage5_certificate_record_v2",
                            "prompt_name": prompt_name,
                            "records": certificate_records,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )

            excluded_prompt = SimpleNamespace(name=excluded_prompt, source_sha256="sha-structured", token_count=token_count)
            excluded_settings = _stream_settings_payload(
                method_name="full_kv",
                prompt=excluded_prompt,
                repo_id="repo",
                repo_revision="rev",
                recent_window=2,
                block_size=2,
                tolerance=0.05,
                precision=128,
                seed=0,
                total_layers_to_run=32,
            )
            excluded_hidden = _stream_hidden_state_path(output_dir, excluded_prompt.name, "full_kv")
            _save_tensor_artifact(excluded_hidden, {"hidden_states": torch.zeros((token_count, 8), dtype=torch.bfloat16)})
            excluded_logits = output_dir / "checkpoints" / f"{excluded_prompt.name}__full_kv__logits.safetensors"
            excluded_logits.write_bytes(b"synthetic")
            excluded_storage = _method_storage_summary(
                method_name="full_kv",
                total_layers=32,
                modified_layers=modified_layers,
                num_kv_heads=2,
                head_dim=4,
                sequence_length=token_count,
                recent_window=2,
                block_size=2,
                precision=128,
            )
            _save_stream_checkpoint(
                output_dir,
                excluded_prompt.name,
                "full_kv",
                {
                    "schema": STAGE5_STREAM_CHECKPOINT_SCHEMA,
                    "status": "complete",
                    "method_name": "full_kv",
                    "settings": excluded_settings,
                    "next_layer_index": 32,
                    "hidden_state_path": str(excluded_hidden),
                    "full_logits_path": str(excluded_logits),
                    "runtime_s": 1.0,
                    "metric_records": [],
                    "method_aggregate": {"storage": excluded_storage},
                    "certificate_records": [],
                },
            )

            summary = _finalize_from_checkpoints(
                output_dir=output_dir,
                review_zip=review_zip,
                prompt_names=required_prompts,
                excluded_partial_prompts=(excluded_prompt.name,),
            )
            self.assertEqual(summary["status"], "complete")
            self.assertTrue(summary["all_required_streams_complete"])
            self.assertEqual(summary["model_forward_calls"], 0)
            self.assertEqual(summary["hidden_state_recomputations"], 0)
            self.assertEqual(summary["remote_tensor_fetches"], 0)
            self.assertEqual(summary["manifest_mismatch_count"], 0)

            results = json.loads((output_dir / STAGE5_RESULTS_JSON).read_text(encoding="utf-8"))
            self.assertEqual(results["prompt_names"], list(required_prompts))
            self.assertEqual(results["prompt_count"], 2)
            self.assertEqual(results["excluded_partial_prompts"], [excluded_prompt.name])
            self.assertEqual([prompt["name"] for prompt in results["prompts"]], list(required_prompts))
            self.assertEqual(results["aggregate_method_metrics"]["full_kv"]["case_count"], expected_scored * 2)

            report = (output_dir / STAGE5_REPORT_MD).read_text(encoding="utf-8")
            self.assertIn("Full 32-layer teacher-forced evaluation on two 128-token prompt categories", report)
            self.assertIn("excluded_partial_prompts", json.dumps(results))

            with zipfile.ZipFile(review_zip, "r") as handle:
                names = set(handle.namelist())
            self.assertIn("excluded_partial_prompt_provenance/checkpoints/structured_reasoning_128__full_kv.json", names)
            self.assertIn(STAGE5_RESULTS_JSON, names)
            self.assertIn(STAGE5_REPORT_MD, names)


if __name__ == "__main__":
    unittest.main()
