# RACK-KV

RACK-KV is a research implementation for random-access KV-cache compression and certified attention skipping.

## Included

- `rack_kv/`: compression, random-access storage, anisotropic certification, hierarchy, GQA eligibility, and physical omission.
- `experiments/`: reproducible capture, evaluation, validation, and aggregation entry points.
- `tests/`: unit and focused integration tests.
- `configs/`: pinned V1 configuration and representative prompt metadata.
- `docs/`: architecture, theorem, reproducibility, and limitation documentation.
- `paper/` and `presentation/`: source files only; generated PDFs and archives are intentionally excluded.

Generated results, model traces, caches, private files, temporary files, logs, PDFs, and ZIP archives are not part of this repository. Reproduce measurements locally using the documented commands and your own model access.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

See [SETUP.md](SETUP.md) and [REPRODUCING_RACK_KV_V1.md](REPRODUCING_RACK_KV_V1.md) for environment and reproduction details.

## Tests

```powershell
python -m pytest -q
```

## Reproduction

The frozen configuration is [configs/rack_kv_v1.yaml](configs/rack_kv_v1.yaml). Use the experiment modules under `experiments/` and the commands documented in `docs/rack_kv_v2_reproducibility.md`.

RACK-KV 1.0 remains the control implementation. RACK-KV 2.0 mechanisms are retained as separate experimental code paths and do not replace the V1 codec or theorem.
