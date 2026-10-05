"""Merge Step-11 deterministic chunk outputs, excluding the one-case smoke run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.common import ROOT, write_csv, write_json


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stats(rows, field):
    values = [float(row[field]) for row in rows]
    return {"mean": float(np.mean(values)), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/rack_kv_v2_step11_tight_certificate")
    args = parser.parse_args()
    smoke_chunks = {"chunk_0_1", "chunk_0_3"}
    chunks = sorted(p for p in args.root.glob("chunk_*") if p.is_dir() and p.name not in smoke_chunks)
    if not chunks:
        raise SystemExit("no Step-11 chunks found")
    names = ["certificate_ablation.csv", "probability_bounds.csv", "centering_ablation.csv", "mass_conservation.csv", "value_term.csv", "block_vs_token.csv", "total_certificate.csv", "timing.csv", "rigor_validation.csv"]
    merged = {}
    for name in names:
        rows = []
        for chunk in chunks:
            path = chunk / name
            if path.exists():
                rows.extend(read_rows(path))
        merged[name] = rows
        write_csv(args.root / name, rows)
    ablation = merged["certificate_ablation.csv"]
    total = merged["total_certificate.csv"]
    rigor = merged["rigor_validation.csv"]
    truth = {"true", "1", "yes"}
    summary = {
        "scope": "Frozen 330-case Step-11 tight compression certificate; no model inference.",
        "chunks": [p.name for p in chunks],
        "cases": len(ablation),
        "compression": {
            "actual": stats(ablation, "actual_compression_error"),
            "step10": stats(ablation, "e_comp_step10"),
            "coordinate": stats(ablation, "e_comp_coordinate"),
            "mass": stats(ablation, "e_comp_mass"),
            "centered": stats(ablation, "e_comp_centered"),
            "tight": stats(ablation, "e_comp_tight"),
            "violations": sum(str(row["tight_violation"]).lower() in truth for row in ablation),
        },
        "total": {
            "actual": stats(total, "actual_total_error"),
            "tight": stats(total, "total_bound_tight"),
            "violations": sum(str(row["total_violation"]).lower() in truth for row in total),
        },
        "interval_violations": sum(int(row["interval_violations"]) for row in ablation),
        "centers": {name: sum(row["selected_center"] == name for row in ablation) for name in ("zero", "mean", "ohat")},
        "minimum_compression_margin": min(float(row["compression_margin"]) for row in rigor),
        "minimum_total_margin": min(float(row["total_margin"]) for row in rigor),
    }
    write_json(args.root / "summary.json", summary)
    (args.root / "test_report.txt").write_text(f"Merged {len(chunks)} deterministic chunks; {len(ablation)} cases; 256-bit directed rounding.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
