# Theoretical Expansion Report

## Scope

This revision modifies the existing professional manuscript in place. It does not run model inference, change numerical results, alter the scientific implementation, or add unsupported claims.

## Main scientific expansions

The following main-paper subsections were substantially deepened or rewritten:

- `Background and Problem Formulation`
- `Related Work`
- `RACK-KV`
  - `Design Objectives`
  - `Cache Partitioning`
  - `Block-Local Sequence-Aware Compression`
  - `Independently Decodable Storage`
  - `Geometric Block Summaries`
  - `Query-Dependent Certified Skipping`
  - `Grouped-Query Attention Considerations`
- `Theoretical Analysis`
  - `Attention Decomposition`
  - `Blockwise Logit Upper Bound`
  - `Skipped-Mass Bound`
  - `Certified Output-Error Theorem`
  - `Proof and Interpretation`
  - `Compression and Skipping Error Decomposition`
  - `Storage and Computational Complexity`
  - `Conservativeness of the Certificate`
- `Results`
  - `Compression and Random-Access Correctness`
  - `Storage--Accuracy Tradeoff`
  - `Matched-Budget Baseline Comparison`
  - `Certificate Safety`
  - `Certificate Tightness and Skip Frequency`
  - `Layer-Dependent Behavior`
  - `Full-Model Quality`
- `Discussion`
- `Limitations`
- `Conclusion`

## Structural corrections

- Removed all lettered appendices from the main PDF.
- Ensured the main paper ends with:
  - `Discussion`
  - `Limitations`
  - `Conclusion`
  - `Acknowledgments`
  - `References`
- Moved proof-, reproducibility-, and extended-material content into a separate supplementary document:
  - `supplementary.tex`
  - output `RACK_KV_Supplementary.pdf`

## Quantitative depth added

- Displayed `equation`/`align` environments in the main-paper `sections/` tree:
  - backup before editorial revision: `21`
  - final professional v2 manuscript: `42`
  - net increase: `21`
- Main-paper theorem/proposition environments:
  - `2`

## Scientific presentation corrections

- Kept the full certified-skipping proof in the main paper instead of deferring it entirely to an appendix.
- Expanded the formal separation among:
  - original full-precision attention
  - fully reconstructed compressed attention
  - reconstructed attention after certified skipping
- Made the random-access contract explicit as a scientific design constraint rather than a software detail.
- Added a dedicated grouped-query-attention subsection explaining the distinction between logical query-head skips and physical KV-block omission.
- Added an explicit complexity discussion separating:
  - storage reduction
  - logical skipping
  - physical decode avoidance
  - runtime claims not yet established

## Constraints preserved

- No new experiment was run for this revision.
- No result value was fabricated or altered.
- No full-model quality claim was added without verified evidence.
- No latency or physical decode-avoidance claim was introduced.

## Remaining limitations retained prominently

- The theorem remains local to one reconstructed attention computation.
- Compression error is still empirical rather than rigorously certified.
- Logical skipping still does not imply physical GQA-group omission.
- The certificate remains conservative and skips rarely in the evaluated cases.
- Full 32-layer teacher-forced quality remains unverified in the main paper.
