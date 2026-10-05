# RACK-KV 2.0 Step 10: Rigorous Compression-Error Certificate

## Scope

Step 10 adds a formal bound for the error introduced by the frozen V1
first-token-anchor INT8 codec. It does not change the codec, the
anisotropic skip theorem, the skip decisions, or the serialized V1 format.
The validation uses the 330 frozen representative attention cases and
256-bit directed-rounding arithmetic; no model recapture or inference was
used.

The source tensors are bfloat16 trace values. They are promoted for
arithmetic, but are not treated as FP16 inputs.

## Certificate construction

At encode time, the implementation compares each original finite vector
with the vector produced by the actual serializer and decoder. For each
historical block it stores outward-safe upper bounds

\[
\kappa_m \geq \max_{i\in m}\|k_i-\hat{k}_i\|_2,
\qquad
\eta_m \geq \max_{i\in m}\|v_i-\hat{v}_i\|_2.
\]

Recent-window entries are exact and therefore receive zero error metadata.
The query-time certificate requires reconstructed K/V and these summaries,
not the original cache.

For a query \(q\), the key bound gives

\[
|s_i-\hat{s}_i|
\leq \delta_i
=\|q\|_2\kappa_i/\sqrt d.
\]

The implementation evaluates the softmax interval using shifted outward
MPFR exponentials. With \(Z_-\) and \(Z_+\) formed from the lower and upper
logit endpoints, it obtains

\[
l_i=\frac{e^{\hat{s}_i-\delta_i}}{Z_+},
\qquad
u_i=\frac{e^{\hat{s}_i+\delta_i}}{Z_-}.
\]

The compression certificate is

\[
E_{\mathrm{comp}}
=E_{\mathrm{value}}+E_{\mathrm{probability}},
\]

where the value term uses the smaller of the uniform \(\max_i\eta_i\) bound
and the interval-weighted \(\sum_i u_i\eta_i\) bound. The probability term
uses the smaller of \(\sum_iD_i\|\hat v_i\|_2\) and the generic probability
bound \(2\max_i\|\hat v_i\|_2\).

## Composition with skipping

The existing theorem remains independent:

\[
\|\hat{o}_{\mathrm{full}}-\hat{o}_{\mathrm{kept}}\|_2
\leq E_{\mathrm{skip}}.
\]

Therefore the local total guarantee follows by the triangle inequality:

\[
\|o_{\mathrm{exact}}-\hat{o}_{\mathrm{kept}}\|_2
\leq E_{\mathrm{comp}}+E_{\mathrm{skip}}.
\]

This composition is not the substantive novelty; the new component is the
construction of \(E_{\mathrm{comp}}\) from codec reconstruction bounds and
softmax perturbation intervals.

## Validation results

The corrected full replay contains 330 cases and 902 active rank-8 skip
decisions. The results are:

| Quantity | Mean | Median | P95 | Maximum |
|---|---:|---:|---:|---:|
| Actual compression error | 0.002152 | 0.000817 | 0.010309 | 0.028731 |
| Certified compression bound | 0.732208 | 0.638792 | 1.776045 | 4.230198 |
| Actual total error | 0.003254 | 0.002173 | 0.010342 | 0.028892 |
| Certified total bound | 0.756821 | 0.649639 | 1.802369 | 4.230198 |

There were **0 compression-certificate violations** and **0 total-certificate
violations**. The minimum observed compression safety margin was 0.009880;
the minimum total-certificate margin was 0.011873.

The mean compression-bound to actual-error ratio was approximately 869x;
the median was approximately 549x and the P95 was approximately 2,658x.
The probability-interval term dominated the looseness (mean 0.687553) over
the value term (mean 0.044655).

Per-token metadata was evaluated on a deterministic 20-case sample. On that
sample its mean bound was 0.028003, compared with 0.064573 for block metadata.
It reduced the bound, but costs 8 bytes per token rather than the mean 163.64
bytes per case for block summaries. The full block-level result therefore
remains the practical default; token metadata is retained as an analysis
option, not silently substituted into the main result.

The mean block error metadata cost was 163.64 bytes per case, while the
corresponding per-token cost was 1,298.91 bytes per case. Across the replay,
block error metadata added 54,000 bytes. Adding it to the V1 independently
decodable representation changes the measured 1.62513x ratio to 1.62041x;
this is the representation ratio including Step-10 block error metadata.
Per-token metadata would add 428,640 bytes and is therefore not the default.

## Files and reproducibility

The implementation is in
`rack_kv/compression_certificate.py`. The deterministic runner is
`experiments/run_step10_compression_certificate.py`, and chunk aggregation
is handled by `experiments/merge_step10_chunks.py`.

The machine-readable aggregate is in
`results/rack_kv_v2_step10_compression_certificate/`, including compression,
probability-interval, metadata, breakdown, total-certificate, and rigor
validation tables.

Step 10 establishes a complete **local** original-cache to reconstructed-
and-kept-cache guarantee for the evaluated attention cases. It does not by
itself establish a sequence-level model-quality guarantee, a latency result,
or a serving-system guarantee. The large probability-interval looseness is
the principal limitation and should be addressed before treating this as a
practical production certificate.
