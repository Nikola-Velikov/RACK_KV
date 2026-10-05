#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/extract_verified_data.py

build_tex() {
  local entry="$1"
  local stem="${entry%.tex}"
  if [ -x ./tectonic_dist/tectonic.exe ]; then
    ./tectonic_dist/tectonic.exe --keep-logs --keep-intermediates "$entry"
  elif command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode -halt-on-error "$entry"
  elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode -halt-on-error "$entry"
    bibtex "$stem"
    pdflatex -interaction=nonstopmode -halt-on-error "$entry"
    pdflatex -interaction=nonstopmode -halt-on-error "$entry"
  else
    echo "No LaTeX toolchain found. Install latexmk or pdflatex+bibtex before building." >&2
    exit 1
  fi
}

build_tex main.tex
build_tex supplementary.tex

cp main.pdf RACK_KV_Paper_Professional_v2.pdf
cp supplementary.pdf RACK_KV_Supplementary.pdf
python scripts/package_source.py
