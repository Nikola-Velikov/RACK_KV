# RACK-KV 1.0 reproduction schema

Schema IDs are `rack_kv_v1_configuration_1`, `rack_kv_v1_freeze_1`,
`rack_kv_v1_run_1`, and `rack_kv_v1_summary_1`. Original method and serialization
versions remain unchanged. Manifest status must be `complete` before comparison.

Every output row is scoped to its containing run manifest. The manifest includes
the configuration, model/tokenizer revisions, RNG seeds, deterministic flags,
device, software versions, exact command, input evidence and source lock.
`source_snapshot/` preserves the orchestration and scientific code used in that
run. `manifest.json` excludes itself from its size/hash inventory.

`representative_cases.csv` preserves the frozen evaluator's field names and
semantics. Its key is `(case_key, method_name, mode)`; duplicates are errors.
The required population is layers x positions x local query indices x eight
method/mode pairs. The primary experiment has 330 query cases and 2,640 rows.
The explicitly configured short replay has two query cases and 16 rows.

## Additive storage fields

`anchor_bytes + residual_bytes + scale_bytes + metadata_bytes + index_bytes +
container_header_bytes + recent_window_bytes + certificate_exclusive_bytes`
equals `rack_kv_bytes` in every RACK storage row. `payload_bytes` is a convenience
subtotal `anchor_bytes + residual_bytes` and MUST NOT be added to those categories.
Geometric metadata is counted in block metadata once; no extra certificate-only
payload exists. All-method representative breakdown fields in the original
records also sum exactly to serialized bytes.

Do not sum representative query-head records to estimate physical cache usage:
different queries can reuse the same KV head. Whole-model storage instead uses
the existing GQA-aware storage-accounting schema. Final-prefix quantities and
cumulative-prefix quantities are explicitly separate fields.
For non-full methods, disjoint category sums describe cumulative prefixes over
the five modified layers; the all-layer cumulative total additionally contains
27 exact layers. For Full KV, categories and cumulative total both cover all 32
layers. The validator selects the authoritative denominator by method.

## Certificate diagnostics

- A candidate row is one historical block under one layer/query/head decision.
- `U_m` and actual unnormalized mass use the unshifted mathematical convention;
  log-domain fields preserve information when exponentials cannot fit FP64.
- The logit cap is recomputed with the existing MPFR interval function. Measured
  actual logits/mass and ratios use FP64 and never authorize any skip.
- `certificate_bound` and `actual_skipping_error`, when present, concern the
  joint selected skipped set specified by the query-case record. The candidate
  table repeats that set-level context only on its members, with an explicit
  `error_scope`. Do not average those repeated values as independent cases.
- `bound_to_error_ratio` uses the recorded `numerical_floor`. Zero-skip query
  records have zero bound/error but are excluded from tightness distributions.
- Fresh local evaluations observe the number of progressive returned MPFR
  states through a wrapper that returns the exact original result. This count
  includes the terminal state and is not the number of primitive MPFR operations.
  Old records have null for that unavailable count.
- There is no FP64 prefilter in the representative evaluator. Full-model
  certificate records contain the genuine prefilter/MPFR split.

## Geometry

Keys are `(layer_index, kv_head_global, block_start, block_size, representation)`.
Repeated prefixes/query heads share one geometry record. Representation is
`original` or `reconstructed`. The mean and first-token anchor are full vectors;
the centroid-centered SVD spectrum has min(block_size, key_dim) entries.
Effective rank is exp(entropy of normalized squared singular values); constant
blocks have rank and anisotropy zero by convention. Residual energies are the
squared singular-value tail, with corresponding normalized fractions.

## Known original-field caveat

In `stage4._evaluate_rack_kv_case`, `attention_output_l2_error` is the certified
total error, whereas some auxiliary fields (`relative_l2_error`, cosine and
maximum-component error) are calculated from full reconstructed output. This
freeze preserves those fields and their old behavior. Primary comparisons use
the original total L2 field. No scientific fix or claimed numerical improvement
has been made; using the auxiliary fields as kept-output errors would be wrong
on skipped cases.
