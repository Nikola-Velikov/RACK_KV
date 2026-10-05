"""Merge deterministic Step-10 chunk outputs."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from experiments.common import ROOT, write_csv, write_json


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/rack_kv_v2_step10_compression_certificate")
    args = parser.parse_args()
    # The one-case smoke output is intentionally excluded from the complete
    # validation aggregate; production chunks are the 25-case (or final
    # remainder) ranges created by the runner.
    chunks = sorted(
        p for p in args.root.glob("chunk_*")
        if p.is_dir() and p.name != "chunk_0_1"
    )
    if not chunks:
        raise SystemExit("no Step-10 chunks found")
    names = ["compression_cases.csv", "error_metadata.csv", "probability_intervals.csv", "bound_breakdown.csv", "block_vs_token_metadata.csv", "total_certificate.csv", "rigor_validation.csv", "metadata_accounting.csv"]
    for name in names:
        rows = []
        for chunk in chunks:
            path = chunk / name
            if path.exists(): rows.extend(read_rows(path))
        write_csv(args.root / name, rows)
    total = read_rows(args.root / "total_certificate.csv")
    comp = read_rows(args.root / "compression_cases.csv")
    def nums(field, data): return [float(r[field]) for r in data]
    truth = {"true", "1", "yes"}
    summary = {"scope": "Frozen 330-case trace validation; no model inference.", "chunks": [p.name for p in chunks], "cases": len(total), "precision": 256, "codec": "V1 first-token INT8", "block_certificate": {"violations": sum(str(r["block_violation"]).lower() in truth for r in comp), "mean_bound": float(np.mean(nums("block_cert", comp))), "p95_bound": float(np.percentile(nums("block_cert", comp), 95))}, "total_certificate": {"violations": sum(str(r["total_violation"]).lower() in truth for r in total), "mean_bound": float(np.mean(nums("total_bound", total))), "p95_bound": float(np.percentile(nums("total_bound", total), 95))}}
    write_json(args.root / "summary.json", summary)
    (args.root / "test_report.txt").write_text(f"Merged {len(chunks)} deterministic chunks; {len(total)} cases; 256-bit directed rounding.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
