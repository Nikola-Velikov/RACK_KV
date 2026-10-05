$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
python scripts\extract_verified_data.py

function Invoke-TexBuild {
  param([string]$EntryPoint)
  if (Test-Path .\tectonic_dist\tectonic.exe) {
    .\tectonic_dist\tectonic.exe --keep-logs --keep-intermediates $EntryPoint
  } elseif (Get-Command latexmk -ErrorAction SilentlyContinue) {
    latexmk -pdf -interaction=nonstopmode -halt-on-error $EntryPoint
  } elseif (Get-Command pdflatex -ErrorAction SilentlyContinue -and (Get-Command bibtex -ErrorAction SilentlyContinue)) {
    pdflatex -interaction=nonstopmode -halt-on-error $EntryPoint
    bibtex ([System.IO.Path]::GetFileNameWithoutExtension($EntryPoint))
    pdflatex -interaction=nonstopmode -halt-on-error $EntryPoint
    pdflatex -interaction=nonstopmode -halt-on-error $EntryPoint
  } else {
    throw "No LaTeX toolchain found. Install latexmk or pdflatex+bibtex before building."
  }
}

Invoke-TexBuild main.tex
Invoke-TexBuild supplementary.tex

Copy-Item -LiteralPath main.pdf -Destination RACK_KV_Paper_Professional_v2.pdf -Force
Copy-Item -LiteralPath supplementary.pdf -Destination RACK_KV_Supplementary.pdf -Force
python scripts\package_source.py
