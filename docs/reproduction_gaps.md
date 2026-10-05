# Reproduction gaps and interpretation limits

| Gap | Result affected | Evidence / likely source | Does the paper suffice? | Next action |
|---|---|---|---|---|
| No usable Git commit metadata | Exact repository ancestry | `.git` cannot be resolved by Git | No | Use the frozen source archive and file hashes; recover original Git history separately if available. |
| Fresh full-model replay is expensive | Independent reproduction of all eight 32-layer streams | Original full run required long CPU streams; accepted checkpoints and token records exist | No | Run the documented `--all --recapture` command on a dedicated machine; current import mode must be labeled historical evidence. |
| Historical representative records omit progressive MPFR iteration counts | Exact MPFR-work count | `stage2._run_query_case_tolerance_bundle`, `progressive_certification_steps` | No | Leave iteration count null. Candidate block count is not MPFR function-call count. Instrument only through an observer in a separately documented future reporting revision if required. |
| Historical certificates bound a selected set | Individual-block output-error attribution | `extra.skipped_block_starts`, certificate text and joint observed error | No | Preserve joint-set scope. Per-block logit/mass diagnostics are available; do not duplicate the joint error as an independently certified block error. |
| Auxiliary RACK metrics have mixed output scope | Relative L2/cosine/max-component interpretation on skipped cases | `stage4._evaluate_rack_kv_case` computes these from reconstructed full output, total L2 from kept output | No | Preserve old behavior; use total L2 for comparisons and document the auxiliary scope. No fix was applied. |
| Full-model streams do not persist all Q/K block tensors | Full-model geometry at every token/head | Full-model checkpoints retain outputs, metrics and certificates, not a complete Q/K history | No | Geometry dataset currently covers all representative trace blocks. A future capture observer would be needed for full-model per-block geometry; never infer it from representative traces. |
| Earlier layer-0 safety uses legacy arithmetic | Exact reproduction of historical pseudo-logit comparisons with current code | Historical layer-0 v2 review package | No | Preserve that archive; use its source snapshot for historical reproduction. Do not rerun it through current arithmetic or include it as full-model quality. |
| Earlier smoke/pilot settings differ from primary benchmark | Every historical development diagnostic | Stage2/2B scripts and artifacts | Partially | Existing entry points remain frozen; the master control experiment covers the current representative and two-prompt full-model findings, not every historical engineering probe. |
| Cross-platform bitwise determinism not established | Fresh capture/teacher-forced logits | Library builds, BLAS, CPU instructions, BF16 kernels | No | Pin dependencies and hardware; compare raw metrics with declared tolerance if the platform differs. Two local short replays test only same-environment repeatability. |
| Runtime provenance includes multiple processes | Inference speed claims | Historical logs/checkpoints vs new profiling timer | No | New runtime is orchestration/test/profiling time only. No inference acceleration claim. |
| Historical downloads allowed insecure TLS | Historical transport provenance | Capture report `tls_verification` | No | Preserve pinned revision/hashes; new runner verifies TLS by default. An explicit insecure flag is recorded if locally necessary. |

There is no proof of compression-error or end-to-end language-model-error bounds.
The theorem bounds reconstructed-cache skipping error. The full-model evidence
covers two deterministic 128-token prompts, with five modified layers in an
otherwise complete 32-layer path. GQA physical eligibility is distinct from
physical decoding actually avoided.

The geometry profiler measures centered SVD spectra. For blocks with eight
entries, centered rank is at most seven: residual energy after ranks 8 and 16
is zero by construction. This is not evidence of useful high-dimensional
compression at those ranks. The profiler does not implement an anisotropic
certificate or modify any skip decision.
