# Reproducing RACK-KV 1.0

The primary control consists of the 330-case representative-layer experiment
and the two-prompt, four-method full-model experiment. The frozen scientific
code is unchanged. Post-evaluation geometry and certificate-tightness profiles
are measurements only.

## Environment and setup

Use Python 3.11 or newer; the historical environment used Python 3.13.5 on
Windows 11, CPU PyTorch 2.13.0, Transformers 5.14.1 and gmpy2 2.2.1. Consult
`results/rack_kv_v1_baseline/manifest.json` for the actual current dependency
versions and hardware. `configs/rack_kv_v1_environment.txt` pins the observed
packages used in this freeze, including plotting and memory monitoring.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r configs/rack_kv_v1_environment.txt
```

The `+cpu` Torch build may require the official CPU-wheel index on another
machine. Python, operating system and hardware must also be matched for a
bitwise comparison; a pip version list alone is insufficient.

Run commands from the repository root. The configuration is JSON-formatted YAML
1.2, read with the standard library. The lock rejects changes to scientific
source, prompts or configuration. Do not regenerate the lock to conceal changes.
Future methods belong in a separately identified experiment/code snapshot.

## Commands

Short real-trace reproduction, without model inference:

```powershell
python -m experiments.reproduce_v1 --config configs/rack_kv_v1.yaml --quick --output results/v1_quick_a
python -m experiments.reproduce_v1 --config configs/rack_kv_v1.yaml --quick --output results/v1_quick_b
python -m experiments.compare_to_v1 results/v1_quick_b --baseline results/v1_quick_a
```

This explicitly selects layers 0 and 31, position 31, query head 0, all eight
method/mode combinations, and the unchanged W16/B8/epsilon0.05/256-bit settings.
It is not a substitute for the complete representative population.
After running the full suite once, `--test-scope invariants` selects only the
new invariant tests for the two repeatability runs. The deterministic harness
`python -m experiments.run_tests` executes the complete suite. On Windows it
suppresses only `EmptyWorkingSet` inside the test process, avoiding repeated OS
page eviction in tiny fixtures; garbage collection and scientific assertions
remain active. Production paths keep their existing memory behavior. Interrupted
or superseded test logs are retained under `test_history/`.

Reaggregate and profile existing accepted records without model inference:

```powershell
python -m experiments.reproduce_v1 --config configs/rack_kv_v1.yaml --from-existing --output results/rack_kv_v1_baseline
```

This hashes frozen inputs, checks serialized payload hashes, rebuilds summaries,
rechecks independent block decoding, profiles every representative candidate,
and reaggregates full-model token records. Its manifest identifies historical
inputs; it does not claim that the original model streams were rerun.

Complete fresh scientific reproduction, including trace generation:

```powershell
python -m experiments.reproduce_v1 --config configs/rack_kv_v1.yaml --all --recapture --output results/v1_fresh_full
```

Without `--recapture`, a verified existing representative trace can be reused;
all representative method outputs and full-model streams are computed in the
new output directory. Missing trace assets trigger capture. `--full-model`,
`--representative`, `--baselines`, `--certificate-safety`, and
`--serializer-tests` select components. Representative/baseline/certificate
flags evaluate the common paired population so scopes cannot drift silently.

No source editing is needed. Interrupted new runs can use the same command with
`--resume`; run identity and completed payload hashes must match. The original
checkpoints and review archives are never deleted. Do not point the new runner
at an old accepted result directory.

## Model assets and resource needs

Model: `NousResearch/Meta-Llama-3.1-8B`, revision
`1f47e50cdbe801ad8a5174156ec3a0655108fb9f`; tokenizer uses that revision.
Fresh capture may fetch tensor ranges and pinned tokenizer/config files.
The full-model runner uses the exact hashed tensor cache. Provide disk space
for BF16 model data, cache entries, trace files, checkpoints and outputs.
The CPU implementation streams layers and uses an 8 GiB process guard and
2 GiB available-memory floor. At least 16 GiB system RAM is a practical target;
the guards remain authoritative. No CUDA execution is required.

The historical five-layer capture took about 86 minutes on the original CPU.
Full-model computation took many hours across resumed runs; no single reliable
fresh total is available. Allow a dedicated long run. New short replay and
profiling runtimes are recorded in their summaries. Do not interpret test or
profiling runtime as inference latency.

TLS verification is enabled by default. `--allow-insecure-tls` is an explicit,
recorded transport override for historical environment problems, never a
silent fallback. Model licensing/access requirements still apply.

## Output and metric semantics

Every run has a manifest with seeds, deterministic flags, pinned revisions,
library/hardware information, exact command, source lock and output hashes.
Each CSV belongs to that run manifest. A missing result is blank/null with a
status, never an invented zero. JSON export rejects NaN/Inf.

| File | Population and meaning |
|---|---|
| `representative_cases.csv` | One query case/method/mode, preserving the original metric definitions. |
| `certificate_cases.csv` | One query case, joint skipped-set error/bound, violations and fallbacks. |
| `candidate_tightness.csv` | Every representative query-head/block decision, block cap and actual mass. |
| `geometry_blocks.csv` | Unique layer/KV-head/start/length blocks, original and reconstructed key geometry. |
| `storage_breakdown.csv` | Disjoint authoritative bytes for each RACK query case; `payload_bytes` is a labeled subtotal of anchors+residuals, never added again. |
| `random_access.json` | Every representative serialized block decoded directly by offset and compared with complete reconstruction. |
| `full_model_results.csv` | Per-prompt and pooled token metrics; perplexity is exp(pooled mean NLL). |
| `full_model_storage.json` | Final-prefix and cumulative-prefix totals remain distinct, summed across all layers. |
| `full_model_certificate_records.json` | Existing rigorous decisions, including prefilter and MPFR fields. |
| `summary.json` | Machine-readable representative/full-model summaries and execution status. |
| `test_report.txt` | This run's deterministic tests; exact command and exit code have companion files. |
| `plots/` | Nine vector PDFs plus PNGs and a machine-readable population/filter description. |

Representative mean bytes are the mean per-query, per-KV-head prefix storage,
not a physical cache total summed over query heads. Means of compression ratios
are not ratios of mean bytes. Full KV may show a small negative saving because
serialized controls include headers while the raw comparison denominator does
not. Full-model final cache sums all layers and both prompts; cumulative bytes
sum every prefix and must never be called final cache bytes.

Geometry uses centroid-centered singular values and energy-normalized entropy
rank. Rank residual energies are squared Frobenius energy. Original and
reconstructed blocks are both retained. Main plots use all unique reconstructed
blocks; candidate plots use all query decisions, including repeated blocks
under different queries. FP64 mass/logit diagnostics never authorize skips.
Log-domain mass ratios are retained; overflow in convenience exponentials is
null rather than Inf. Joint certificate ratios use an explicit 1e-30 floor.

## Comparison and interpretation

```powershell
python -m experiments.compare_to_v1 results/new_experiment
```

The comparison rejects differing query populations or baseline plans, reports
new-minus-control storage/error/certificate/quality deltas, and flags invariant
regressions. Runtime differences include only the recorded runtime scope.
Future interventions must keep evaluation populations fixed and identify source
changes explicitly. The v1 lock is intentionally not an editable v2 switch.

See `docs/baseline_code_map.md`, `docs/baseline_function_inventory.md`, and
`docs/reproduction_gaps.md` for full code coverage and historical limitations.
