"""Verify run artifacts and compare two same-seed replays without inference."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import numpy as np

from .common import ROOT, read_json, sha256, write_json


def finite_tree(value, path="root"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite JSON number: {path}")
    if isinstance(value, dict):
        for k, v in value.items():
            finite_tree(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            finite_tree(v, f"{path}[{i}]")


def audit_run(folder):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError(f"Run incomplete: {folder}")
    for entry in manifest["files"]:
        path = folder / entry["path"]
        if path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Manifest mismatch: {path}")
        if path.suffix == ".json":
            finite_tree(read_json(path), str(path))
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                for i, row in enumerate(csv.DictReader(handle)):
                    for key, value in row.items():
                        if value and value.strip().lower() in ("nan", "inf", "-inf", "infinity", "-infinity"):
                            raise ValueError(f"Nonfinite CSV at {path}:{i}/{key}")
    logs = [folder / "test_report.txt", *folder.glob("test_history/*/test_report.txt")]
    counts, history_status = [], []
    for log in logs:
        matches = re.findall(r"Ran (\d+) tests? in", log.read_text(encoding="utf-8"))
        counts.extend(map(int, matches))
        return_code = (log.parent / "test_return_code.txt").read_text().strip()
        history_status.append({"log": str(log), "return_code": return_code,
                               "completed_test_count": int(matches[-1]) if matches else None})
        if log == folder / "test_report.txt" and return_code != "0":
            raise ValueError(f"Failed test evidence {log}")
    return {"directory": str(folder), "manifest_file_count": len(manifest["files"]),
            "manifest_mismatches": 0, "nonfinite_numeric_records": 0, "test_counts": counts, "test_history": history_status}


def compare_replays(first, second):
    fields = ["representative_cases.csv", "certificate_cases.csv", "storage_breakdown.csv",
              "candidate_tightness.csv", "geometry_blocks.csv", "random_access.json"]
    matches = {name: sha256(Path(first) / name) == sha256(Path(second) / name) for name in fields}
    if not all(matches.values()):
        raise ValueError(f"Same-seed deterministic mismatch: {matches}")
    return matches


def historical_discrepancies(baseline, quick):
    historical = read_json(Path(baseline) / "representative_records.json")
    new = read_json(Path(quick) / "representative_records.json")
    by_key = {(r["case_key"], r["method_name"], r["mode"]): r for r in historical}
    fields = ["attention_output_l2_error", "total_serialized_bytes", "certified_skipped_blocks",
              "certificate_upper_bound", "observed_skipping_error"]
    deltas = []
    for r in new:
        old = by_key[(r["case_key"], r["method_name"], r["mode"])]
        deltas.append({"case": r["case_key"], "method": r["method_name"], "mode": r["mode"],
                       **{k: r[k] - old[k] if r.get(k) is not None and old.get(k) is not None else None for k in fields}})
    return deltas


def profile_findings(folder):
    folder = Path(folder)
    with (folder / "geometry_blocks.csv").open(encoding="utf-8", newline="") as h:
        geometry = [r for r in csv.DictReader(h) if r["representation"] == "reconstructed"]
    with (folder / "candidate_tightness.csv").open(encoding="utf-8", newline="") as h:
        candidates = list(csv.DictReader(h))
    with (folder / "certificate_cases.csv").open(encoding="utf-8", newline="") as h:
        certificates = [r for r in csv.DictReader(h) if int(r["certified_skips"]) > 0]

    def stats(values):
        a = np.asarray(values, dtype=float)
        return {"count": len(a), "mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.quantile(a, .95)), "max": float(a.max())} if len(a) else {"count": 0}

    return {"reconstructed_unique_blocks": len(geometry),
            "effective_rank": stats([r["effective_rank_energy_entropy"] for r in geometry]),
            "anisotropy": stats([r["anisotropy_sigma1_over_mean"] for r in geometry]),
            "rank_2_residual_fraction": stats([r["residual_energy_fraction_rank_2"] for r in geometry]),
            "rank_4_residual_fraction": stats([r["residual_energy_fraction_rank_4"] for r in geometry]),
            "logit_slack": stats([r["logit_bound_slack"] for r in candidates]),
            "log_mass_ratio": stats([r["log_mass_bound_ratio"] for r in candidates]),
            "certified_joint_bound_error_ratio": stats([r["bound_to_error_ratio"] for r in certificates]),
            "interpretation": "Centered rank <= block_size-1; directional structure alone does not establish a tighter certificate."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=Path("results/rack_kv_v1_baseline"))
    p.add_argument("--first", type=Path, default=Path("results/v1_quick_a"))
    p.add_argument("--second", type=Path, default=Path("results/v1_quick_b"))
    args = p.parse_args()
    report = {"runs": [audit_run(d) for d in (args.baseline, args.first, args.second)],
              "same_seed_exact_matches": compare_replays(args.first, args.second),
              "replayed_subset_minus_historical": historical_discrepancies(args.baseline, args.first),
              "geometry_and_tightness": profile_findings(args.baseline)}
    write_json(ROOT / "docs/v1_execution_audit.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
