"""Aggregate durable chunk outputs from the V2 flat validation replay."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.directory.glob("v2_validation_cases_*.csv"))
    rows = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    by_mode = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
    summary = {"chunk_files": len(files), "rows": len(rows), "modes": {}}
    for mode, group in sorted(by_mode.items()):
        skips = [int(row["skipped_blocks"]) for row in group]
        tokens = [int(row["skipped_tokens"]) for row in group]
        errors = [float(row["attention_l2_error"]) for row in group]
        bounds = [float(row["certificate_bound"]) for row in group]
        margins = [bound - error for bound, error in zip(bounds, errors)]
        summary["modes"][mode] = {
            "cases": len(group),
            "candidate_decisions": len(group),
            "executed_skips": sum(skips),
            "skipped_tokens": sum(tokens),
            "cases_with_skip": sum(value > 0 for value in skips),
            "multi_token_cases": sum(value > 1 for value in tokens),
            "mean_skipping_error": sum(errors) / len(errors),
            "max_skipping_error": max(errors),
            "mean_certificate_bound": sum(bounds) / len(bounds),
            "max_certificate_bound": max(bounds),
            "p95_skipping_error": sorted(errors)[max(0, math.ceil(0.95 * len(errors)) - 1)],
            "p95_certificate_bound": sorted(bounds)[max(0, math.ceil(0.95 * len(bounds)) - 1)],
            "minimum_safety_margin": min(margins),
            "theorem_violation_count": sum(margin < -1e-12 for margin in margins),
            "mpfr_fallbacks": sum(row["mpfr_fallback"].lower() == "true" for row in group),
        }
    (args.directory / "flat_aggregate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.directory / "active_flat_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["mode"], row["case_key"])))


if __name__ == "__main__":
    main()
