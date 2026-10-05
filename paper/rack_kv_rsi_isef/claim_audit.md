# Claim Audit

This file is generated from verified local artifacts and explicitly labels provisional or incomplete claims.

## C01 — VERIFIED
- Section: Random-Access Compression Format
- Claim: The implemented codec stores independently decodable historical blocks without cross-block dependency chains.
- Evidence: Source code + Stage 2/2B serializer validation
- Artifact path: `rack_kv/codec.py; .tmp/stage2b_single_config_freeze/stage2b_single_config_results.json`
- Scope: Implementation + 256-token layer-0 pilot
- Limitation: Independent decode is verified at the logical block level, not as a production GPU kernel.

## C02 — EMPIRICALLY OBSERVED
- Section: Certified Attention Skipping
- Claim: No rigorous certificate violation was observed in the representative-layer baseline benchmark.
- Evidence: Stage 4B representative-layer benchmark
- Artifact path: `.tmp/stage4_baselines_full/stage4_baseline_full_results.json`
- Scope: 330 representative-layer query cases
- Limitation: Zero observed violations do not prove universal safety.

## C03 — VERIFIED
- Section: Results
- Claim: At representative layers, RACK-KV achieved lower mean local attention-output error than matched-budget SnapKV-style and Quest-style baselines.
- Evidence: Stage 4B representative-layer benchmark
- Artifact path: `.tmp/stage4_baselines_full/stage4_baseline_full_results.json`
- Scope: 5 layers, 330 cases, matched serialized-byte comparison
- Limitation: The compared baselines are labeled SnapKV-style and Quest-style rather than exact official reproductions.

## C04 — VERIFIED
- Section: Results
- Claim: The current implementation does not yet demonstrate physical decode avoidance at the GQA-group level.
- Evidence: Corrected layer-0 safety-validation package
- Artifact path: `.tmp/stage5_quality_validation_layer0_review_v2.zip`
- Scope: Layer-0 safety validation only
- Limitation: This is a one-layer safety artifact with diagnostic pseudo-logits only.

## C05 — VERIFIED
- Section: Results
- Claim: A verified 32-layer teacher-forced full-model quality package is available locally for two prompt categories and all four required methods.
- Evidence: Verified Stage 5 final results package
- Artifact path: `.tmp/stage5_quality_final_review.zip`
- Scope: Full 32-layer teacher-forced evaluation on two 128-token prompt categories: natural-language continuation and passkey retrieval.
- Limitation: The verified full-model package covers only natural-language continuation and passkey retrieval.

## C06 — PROVISIONAL
- Section: Novelty Statement
- Claim: No fully verified work in the frozen corpus was found that combines sequence-aware or reference-based KV compression, independently decodable random-access compressed blocks, and a formal safe-skipping certificate in one inference-time system.
- Evidence: Frozen-corpus novelty assessment
- Artifact path: `HAOps synchronized literature artifact + exported bibliography`
- Scope: Corpus bounded to the frozen July 2026 library
- Limitation: This is not a proof of worldwide novelty.
