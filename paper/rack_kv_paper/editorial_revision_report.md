# Editorial Revision Report

## Scope of This Revision

This revision rewrites the existing manuscript in place as a coherent research paper about one method rather than a development diary. No model inference was executed and no scientific result was fabricated or numerically improved. The revision uses the already verified local artifacts, including the representative-layer benchmark and the finalized two-prompt full 32-layer teacher-forced evaluation.

## Main Scientific Writing Changes

- Rewrote the abstract to center the scientific problem, the three-part RACK-KV design, the verified representative-layer result, and the verified two-prompt full-model result.
- Revised the introduction so the main narrative is the tension among compression, random access, and safe computational reduction, rather than a sequence of implementation stages.
- Updated the experimental setup to distinguish clearly between the representative-layer benchmark and the verified two-prompt 32-layer teacher-forced evaluation.
- Replaced the outdated “full-model evaluation unavailable” language with the verified current scope: two 128-token prompt categories, all 32 layers executed, modified layers 0/8/16/24/31, and independent method streams.
- Rewrote the discussion, limitations, and conclusion so they interpret the new full-model evidence honestly while keeping the limitations prominent.

## Results Presentation Changes

- Added a new main-paper full-model quality table summarizing mean NLL, delta NLL, perplexity, top-1 agreement, mean KL divergence, and whole-model final-prefix storage savings.
- Clarified that whole-model storage savings are modest in the 32-layer benchmark because only five of the 32 layers use the modified storage path in this first end-to-end experiment.
- Preserved the representative-layer comparison table, but renamed the methods to human-readable scientific labels such as “Full KV,” “RACK-KV,” and “Uniform INT8.”
- Kept the negative saving for the Full-KV serialization row and explained it explicitly as small benchmark-container overhead rather than a compression failure.

## Figure and Layout Changes

- Redesigned the main left-to-right RACK-KV pipeline figure so the phases are visually separated and the arrows terminate cleanly at node borders.
- Tightened the independent-random-access figure to fit the text block without clipping while keeping node text readable.
- Redesigned the grouped-query-attention figure to separate query heads, logical decisions, group-level eligibility, the shared KV block, and the physical decision path.
- Removed the only overfull-box warnings caused by the large TikZ diagrams.

## Remaining Limits

- The full-model evidence is still intentionally narrow: two short deterministic prompt categories under teacher forcing.
- The manuscript still does not claim physical decode avoidance, latency improvement, or broad language-model generalization.
- Exact provenance, hashes, and engineering-level reproducibility details remain outside the main narrative.
