# RACK-KV 1.0 scientific code map

The authoritative implementation is `rack_kv/`. This freeze does not edit that
directory or the existing experiment scripts. `configs/rack_kv_v1.lock.json`
identifies every frozen source byte; `rack_kv_v1_source_snapshot.zip` preserves
the source, scripts and pre-existing tests. Git cannot resolve a commit in this
workspace. A new source freeze does not prove which source produced old tensors.

## Scientific components

| Source | Important entry points | Scientific role |
|---|---|---|
| `rack_kv/llama_trace.py` | `run_minimal_llama31_capture`, `run_multilayer_llama31_capture` | Pinned Llama CPU capture; sequential layer loading and stock/manual equivalence. |
| same | `save_attention_trace`, `load_attention_trace`, `replay_compact_trace_outputs`, `query_head_to_kv_head` | Compact post-RoPE Q/K and cache-update V; trace serialization, replay and GQA mapping. |
| `rack_kv/stage2b_prompt.py` | `_base_prompt_sections`, prompt materialization helpers | Seeded synthetic 256-token document; frozen token IDs and text in `configs/prompts`. |
| `rack_kv/codec.py` | `encode_block`, `_quantize_residuals` | First-entry FP16 K/V anchors; independent symmetric INT8 residuals with one FP16 scale each. Quantization uses the stored rounded scale. |
| same | `_rigorous_block_metadata` | Outward-safe FP32 radius about the stored first key anchor and maximum reconstructed value norm. Metadata is about reconstructed KV. |
| same | `CompressedBlock.serialize`, `deserialize`, `decode_block`, `decode_token` | Authoritative local representation and independent reconstruction. |
| same | `serialize_block_container`, `deserialize_block_container`, `SerializedBlockContainer.payload_for_block` | `RACKKVH1` header and byte-offset/length index; no cross-block dependencies. |
| `rack_kv/accounting.py` | `compressed_block_bytes`, `historical_memory_report` | Packed byte formulas. Model-wide formula estimates exclude experiment-wrapper overhead; do not substitute for actual payload length. |
| `rack_kv/ieee.py` | exact conversions and proven float upper conversions | Exact IEEE input interpretation and outward rounding into convenience fields. |
| `rack_kv/rigorous.py` | `Interval`, `rounded_*`, `dot_interval`, `norm_lower`, `norm_upper` | MPFR directed-rounding primitives; default precision 256 bits. |
| `rack_kv/certificate.py` | `_block_logit_cap_interval`, `_certificate_candidates` | Spherical logit cap and two rigorous output-error bounds. |
| same | `prepare_certification_inputs`, `progressive_certification_steps`, `certify_progressive_skipping`, prepared variants | Progressive retention with `largest_u_times_nu`; MPFR decides skips. FP64 mode is not certifying. |
| same | `rigorous_attention_output_interval`, `rigorous_output_error_upper_from_intervals`, `exact_reference_output_mpfr` | Validation against reconstructed attention with interval and reference arithmetic. |
| `rack_kv/stage2.py` | `validate_compact_trace`, `_build_prefix_result` | Trace schema/provenance validation, authoritative serialize/load, reconstruction, random access and detailed bytes. |
| same | `_build_query_case_base`, `_run_query_case_tolerance_bundle`, `run_stage2_smoke_experiment` | Original/compressed/kept output decomposition; progressive certificate evaluation across tolerances. |
| `rack_kv/stage2b.py` | `select_stage2b_query_positions`, `run_stage2b_pilot` | Single-layer pilot and configuration aggregation. |
| `rack_kv/stage3.py` | `run_stage3_reduced_layer_case` | Small multi-layer trace evaluation. |
| `rack_kv/stage4.py` | `build_stage4_case_context`, `run_baseline_case`, `aggregate_baseline_results` | Representative-layer scientific experiment and method/mode summaries. |
| same | `_serialize_full_kv`, `_serialize_uniform_int8_kv`, `_serialize_kivi_style` and decode partners | Exact BF16 control; row-wise INT8 control; source-inspired asymmetric affine 2-bit KIVI-style. |
| same | `_serialize_snapkv_style`, `_serialize_quest_style`, `_find_best_budget_match` | Current-query token retention and maximum-exact-logit page selection; native 25% or nearest serialized budget. These are style approximations. |
| same | `_serialize_rack_kv`, `_decode_rack_kv`, `_evaluate_rack_kv_case` | Wrapper accounting and MPFR local RACK evaluation; no full-model FP64 prefilter here. |
| `rack_kv/stage5.py` | `PersistentTensorRangeCache`, `MemoryGuard` | Exact hashed tensor range reuse, resource limits; not scientific compression. |
| same | `_build_stage5_rack_prefix_payload`, `_advance_incremental_rack_state`, `_materialize_incremental_rack_payload` | Incremental but equivalent per-KV-head compressed prefixes. |
| same | `_fast_prefilter_bound`, `_certified_attention_for_head` | FP64 rejects or sends to MPFR; never authorizes. Validation checks the selected skipped set. |
| same | `_shared_reconstructed_attention_output_numpy` | Shared compression-only/certified output path; zero-skip exact parity for identical arrays. |
| same | `_run_modified_layer_stream_exact_from_cache`, `_run_full_kv_prompt_stream`, `_run_nonfull_prompt_stream` | Independent method streams propagated through 32 layers; only selected layers modified. |
| same | `_metric_record_for_token`, `_aggregate_metric_records` | Teacher-forced final-logit metrics. |
| same | `_disjoint_storage_categories_from_record`, `_accumulate_stage5_storage_records`, `_method_storage_summary` | Separate layer-prefix, all-layer cumulative, and final-prefix byte scopes. |
| same | `_aggregate_certificate_records` | Logical skips, GQA eligibility and actual decode avoidance as distinct quantities. |
| same | `run_stage5_quality_smoke` | Full pipeline entry; its historical name does not imply layer-0-only scope. |

The complete AST inventory, including private helpers and line numbers, is
`docs/baseline_function_inventory.md`. Definitions there can be located directly
in the frozen source snapshot.

## Experiment entry points and ambiguity

- `scripts/run_minimal_llama31_capture.py`: original compact single-layer capture.
- `scripts/run_stage2_layer0_smoke.py`: 12-token smoke/configuration sweep.
- `scripts/run_stage2b_layer0_pilot.py`: resumable pilot; completed historical W16/B8 is distinct from its deferred larger sweep.
- `scripts/run_stage3_multilayer_pilot.py`: sequential capture of chosen layers using the frozen 256-token prompt. Default layers are only 0 and 31; the reproduction CLI supplies all five explicitly.
- `scripts/run_stage4_baselines_smoke.py`: both small and full representative benchmark. Its defaults are not the full paper selection. The new runner invokes the same `run_baseline_case` API for all 330 cases and eight method/mode combinations.
- `scripts/run_stage5_quality_smoke.py`: full execution, historical layer-0 packaging, reporting repair, or finalize-only depending on flags. The new full path supplies exactly two prompts and four methods.
- `paper/rack_kv_paper/`: evidence extraction and publication figures. These are presentation/reporting tools, not authoritative scientific implementations.
- `rack_kv_review/` and `zip_check/`: older duplicated source trees for archive inspection. They are not imported by the reproduction CLI and must not shadow `rack_kv/`.
- `.tmp/*/review_package/`: historical snapshots, not active code. `.tmp/stage5_quality_validation_layer0*` contains local safety/diagnostic runs; pseudo-logits must not be merged into full-model results.
- `.tmp/stage5_profile_probe`, short/memory validation directories: profiling or safety variants, not the primary paper population.
- `presentation/`, `templates/`, `memory/`: presentation assets or research notes; not numerical authorities. No such files were edited or deleted by this freeze.

## Tests

The original `tests/` suite covers codec/accounting, rigorous arithmetic,
capture/GQA, single-layer integration, resumable pilots, multi-layer cases,
baseline budgets, full-model tiny synthetic paths, reporting repair and
finalization. It does not load the real Llama weights for unit testing.
`tests/test_reproduce_v1.py` adds freeze checks, arbitrary byte-offset access,
outward metadata checks, nonzero rigorous skip fixtures, prefilter authority,
profiling immutability, known-rank geometry, aggregation and comparison scopes.

## New reproduction layer

`experiments/common.py` handles hashing, deterministic seeds and run provenance.
`freeze_v1.py` creates the one-time immutable control snapshot. `reproduce_v1.py`
orchestrates the frozen APIs and scripts; `results.py` exports/validates records;
`profiling.py` measures serialized blocks after decisions; `plots.py` reads the
complete exported sample; `compare_to_v1.py` compares matching populations.

No profiler output is an input to the v1 codec, prefilter, certificate or model.
