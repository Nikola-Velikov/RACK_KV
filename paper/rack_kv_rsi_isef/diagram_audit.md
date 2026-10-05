# Diagram Audit

All figures in the main paper were inspected after compilation from the rendered PDF pages. Every figure received a final `PASS`.

| Figure | Source file | Node overlap | Arrow-through-node | Arrow-label overlap | Clipping | Minimum readable text size | Corrections made | Final status |
|---|---|---|---|---|---|---|---|---|
| Figure 1 | `figures/fig_system_overview.tex` | PASS | PASS | PASS | PASS | ~9 pt | Removed opaque phase overlays, shortened phase labels, and rechecked all connector anchors so the certificate and attention nodes remain unobstructed. | PASS |
| Figure 2 | `figures/fig_block_format.tex` | PASS | PASS | PASS | PASS | ~9 pt | Kept the requested block, payload, and decoder as separate units and routed the access path around unrelated blocks instead of through them. | PASS |
| Figure 3 | `figures/fig_gqa_logical_vs_physical.tex` | PASS | PASS | PASS | PASS | ~9 pt | Reorganized the GQA diagram into query-head, logical-decision, group-eligibility, and physical-decision levels with one connector family per level. | PASS |
| Figure 4 | `figures/fig_certificate_geometry.tex` | PASS | PASS | PASS | PASS | ~9 pt | Kept vector labels outside the dense geometric region and used a separate callout box for the explanatory text. | PASS |
| Figure 5 | `figures/fig_error_decomposition.tex` | PASS | PASS | PASS | PASS | ~9 pt | Rebuilt the decomposition as a clean three-node chain with error labels above arrows and the total-error annotation outside the connector line. | PASS |
| Figure 6 | `figures/fig_error_vs_bytes.tex` | PASS | PASS (not applicable; plot) | PASS | PASS | ~9 pt | Enlarged the legend and verified that markers, labels, and axes remain readable at 100% zoom. | PASS |
| Figure 7 | `figures/fig_bound_vs_observed.tex` | PASS | PASS (not applicable; plot) | PASS | PASS | ~9 pt | Verified the diagonal comparison line, point markers, and axis text after the final page render. | PASS |
| Figure 8 | `figures/fig_per_layer_rack.tex` | PASS | PASS (not applicable; plot) | PASS | PASS | ~9 pt | Verified legend readability and ensured no clipping of axis labels or markers in the final PDF. | PASS |

## Summary

- Figures inspected: `8`
- Figures substantively redesigned in this revision: `5`
- Arrow-through-node defects remaining: `0`
- Arrow-label overlap defects remaining: `0`
- Clipped figures remaining: `0`
