# RACK-KV V2 GQA Group-Level Certificate

## Purpose

Llama-3.1-8B maps 32 query heads onto 8 KV heads.  A logical skip accepted by
one query head does not by itself permit omission of the KV payload shared by
the other mapped heads.  This implementation therefore separates per-head
logical safety from potential physical eligibility.

## Unanimous physical eligibility

For a KV head `g`, let `H(g)` be the mapped query heads, derived from the
configured ratio `num_attention_heads / num_key_value_heads`.  A region `R`
is physically eligible exactly when every `h in H(g)` passes the unchanged
per-head RACK-KV theorem after adding the *same* `R` to its cumulative skipped
set:

\[
  \operatorname{Bound}_h(S_h \cup R) \leq \epsilon_h.
\]

The traversal commits `R` only after unanimous MPFR authorization, then adds it
to every `S_h`.  Thus no node receives a separate error allowance, and a
physical candidate never represents a union of differently selected regions.
Failure or numerical uncertainty retains the region.

## Group-output inequality

The existing theorem supplies rigorous per-head bounds
\(\lVert\Delta o_h\rVert_2 \leq b_h\).  Since concatenated head outputs
occupy disjoint coordinates,

\[
 \left\lVert\operatorname{concat}_{h\in H(g)}\Delta o_h\right\rVert_2
 \leq \sqrt{\sum_{h\in H(g)}b_h^2}=B_{\mathrm{concat},g}.
\]

Let `W_{O,g}` contain the output-projection columns corresponding to these
heads.  Then

\[
 \lVert\Delta y_g\rVert_2
 \leq \lVert W_{O,g}\rVert_2 B_{\mathrm{concat},g}
 \leq \lVert W_{O,g}\rVert_F B_{\mathrm{concat},g}=B_{\mathrm{group},g}.
\]

The first inequality is mathematical.  The implementation calculates the
Frobenius norm and all subsequent arithmetic with directed MPFR rounding, so
the reported quantity is an outward-safe upper bound.  This Frobenius choice
is deliberately conservative and is exposed through
`output_projection_norm_upper_bound` for future replacement by a tighter
certified operator-norm bound.

## Semantics

`physical_eligible_regions` and `physical_eligible_leaf_blocks` are potential
I/O omissions only.  Step 6 does not prevent payload decoding, change cache
storage, alter GQA attention execution, or establish latency improvement.
`potential_payload_bytes_avoided` is therefore not reported as bytes avoided.

## Current validation limitation

The frozen representative traces contain selected query heads 0, 1, and 4.
They do not include all four heads for any complete Llama GQA group.  The Step
6 API and synthetic safety tests are complete, but an empirical all-group
replay requires an already captured full-group trace or a future capture.  No
new trace is created in this step.
