# RACK-KV 2.0 Step 11: Tight Compression Probability Certificate

## Scope

Step 11 preserves the V1 first-token INT8 codec, the Step-10 error metadata,
the original skip theorem, and the 256-bit directed-rounding infrastructure.
It changes only the probability-side compression certificate. The frozen
validation contains 330 representative attention cases and does not require
model download or inference.

The Step-10 implementation remains available through
`certify_compression`; Step 11 is implemented separately as
`certify_compression_tight`.

## Tightening method

For reconstructed logits \(\hat{s}_i\) and rigorous intervals
\([L_i,U_i]\), Step 11 uses coordinate extrema rather than globally
independent numerator and denominator extrema:

\[
u_i = \frac{e^{U_i}}{e^{U_i}+\sum_{j\ne i}e^{L_j}},
\qquad
l_i = \frac{e^{L_i}}{e^{L_i}+\sum_{j\ne i}e^{U_j}}.
\]

All exponentials, sums, subtractions, and divisions use outward MPFR
rounding with a shared shift for numerical stability.

The certificate then uses \(\sum_i(p_i-\hat p_i)=0\). If
\(a_i=u_i-\hat p_i\) and \(b_i=\hat p_i-l_i\) are clipped at zero, the
common positive/negative mass is bounded by

\[
T\leq\min\left(\sum_i a_i,\sum_i b_i\right).
\]

The weighted perturbation bound is evaluated as two fractional-knapsack
relaxations, one for positive and one for negative mass. This remains safe
even if the two relaxed solutions overlap, because overlap only enlarges the
feasible set.

Finally, because \(\sum_i d_i=0\),

\[
\sum_i d_i\hat v_i=\sum_i d_i(\hat v_i-c)
\]

for any fixed center \(c\). The implementation evaluates zero, arithmetic
mean, and reconstructed full-attention output centers, then takes the
smallest independently valid bound. The value-error term is also tightened
with a box-plus-simplex greedy allocation over the \([l_i,u_i]\) intervals.

## Results

| Certificate | Mean | Median | P95 | Maximum |
|---|---:|---:|---:|---:|
| Step 10 | 0.732208 | 0.638792 | 1.769920 | 4.230198 |
| Tight extrema only | 0.524246 | 0.439568 | 1.307630 | 4.084129 |
| + mass conservation | 0.524234 | 0.439568 | 1.307630 | 4.084129 |
| + value centering | 0.509201 | 0.435149 | 1.269631 | 3.786304 |
| + simplex value term | **0.497756** | **0.418573** | **1.257854** | **3.751524** |

Relative to Step 10, the final mean bound reduction is **32.0%** and the P95
reduction is **28.9%**. The actual compression error remains mean
`2.152e-3` and P95 `9.953e-3`.

The final tight bound-to-actual ratio has mean **460.7x**, median **354.7x**,
and P95 **1,258.1x**. The composed total certificate has mean **0.522370**,
P95 **1.270118**, and median **0.433749**.

The reconstructed-output center wins most often: 247/330 cases (74.8%). The
zero center wins 73 cases and the arithmetic mean wins 10 cases. The mean
selected probability bound is 0.464546; the final value term is reported in
`value_term.csv`, and the probability term in `certificate_ablation.csv`.

There were **0 tight-interval violations**, **0 compression-certificate
violations**, and **0 total-certificate violations**. The minimum compression
margin was 0.008819 and the minimum total margin was 0.010258.

## Metadata and runtime

Step 11 adds no persistent metadata beyond Step 10. Block error metadata
remains 163.64 bytes per evaluated case on average; token metadata remains
1,298.91 bytes per case and is used only for the deterministic 20-case
ablation sample.

The Step 11 implementation measured a mean per-case certificate time of
16.34 seconds and P95 of 27.05 seconds in the frozen replay. In a matched
three-case direct smoke comparison, Step 10 took 11.63 seconds total and
Step 11 took 12.16 seconds total, an observed **1.045x** ratio. This is not
yet a production-performance result; larger cases and serving integration
still require separate measurement.

## Interpretation

The dominant looseness is reduced but not eliminated. Coordinate extrema
provide the largest improvement; value centering and the simplex value term
provide additional reductions. Mass conservation alone is nearly neutral on
this dataset because the weighted coordinate relaxation is already the
limiting term in most cases.

The result is a stronger rigorous local guarantee, but it is not yet
practically tight: the median final bound is still hundreds of times the
observed error. The system is therefore not ready for a final evaluation
freeze without either further certificate tightening or an explicit decision
to accept the conservative bound and its runtime cost.

## Outputs

Machine-readable results are in
`results/rack_kv_v2_step11_tight_certificate/`:

- `certificate_ablation.csv`
- `probability_bounds.csv`
- `centering_ablation.csv`
- `mass_conservation.csv`
- `value_term.csv`
- `block_vs_token.csv`
- `total_certificate.csv`
- `timing.csv`
- `rigor_validation.csv`
- `summary.json`

The replay is run in deterministic chunks with:

```powershell
python -m experiments.run_step11_tight_certificate --start-index 0 --limit 25
python -m experiments.merge_step11_chunks
```
