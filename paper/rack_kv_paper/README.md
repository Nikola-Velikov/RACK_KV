# RACK-KV Paper Source

This directory contains the professionally revised LaTeX manuscript for the RACK-KV project.

- `main.tex`: main-paper entry point
- `supplementary.tex`: separate supplementary-document entry point
- `sections/`: main-paper sections
- `appendices/`: supplementary proof, numerical contract, codec specification, extended results, and reproducibility material
- `figures/`: TikZ and PGFPlots figures
- `tables/`: hand-curated main and appendix tables plus generated support tables
- `data/`: CSV and JSON extracted from verified local artifacts
- `scripts/extract_verified_data.py`: validates local evidence, refreshes the CSV/JSON inputs, and updates `missing_evidence.json`
- `scripts/package_source.py`: creates `RACK_KV_Paper_Professional_v2_Source.zip`

## Build

Preferred command on Windows PowerShell:

```powershell
.\build.ps1
```

Fallback sequence when a TeX toolchain is already installed:

```powershell
python scripts\extract_verified_data.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
Copy-Item main.pdf RACK_KV_Paper_Professional_v2.pdf
Copy-Item supplementary.pdf RACK_KV_Supplementary.pdf
python scripts\package_source.py
```

## Evidence refresh

```powershell
python scripts\extract_verified_data.py
```

The extraction step never invokes model inference. It only reads verified local artifacts such as the representative-layer benchmark, the corrected layer-0 safety-validation package, and the local full-model status record.

## Output files

- `RACK_KV_Paper_Professional_v2.pdf`
- `RACK_KV_Supplementary.pdf`
- `RACK_KV_Paper_Professional_v2_Source.zip`
- `theoretical_expansion_report.md`
- `diagram_audit.md`
- `final_page_visual_audit.md`

## Updating future full-model results

The current draft deliberately withholds full 32-layer quality claims until a verified final package exists. When such a package becomes available:

1. place the completed artifact in the expected local Stage 5 output location;
2. update `scripts/extract_verified_data.py` only if the schema changed;
3. rerun the extraction step;
4. rebuild the manuscript.

## Known limitations

- The paper source does not include model weights or full tensor caches.
- The layer-0 safety-validation artifact is used only for memory, bookkeeping, and all-head validation, not as full-model quality evidence.
- The build reads verified local artifacts only; it never launches model inference.
