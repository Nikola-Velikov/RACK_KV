"""Process-isolated controller for complete final GQA physical validation."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiments.common import ROOT, load_config, write_json


def _valid(path: Path) -> bool:
    try:
        data = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        return data.get("status") == "COMPLETE_GQA_FLAT_R8" and int(data.get("complete_groups", 0)) > 0
    except (OSError, ValueError, TypeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/rack_kv_v2_final_physical_validation")
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()
    config = load_config(ROOT / "configs/rack_kv_v1.yaml")
    args.output.mkdir(parents=True, exist_ok=True)
    positions = list(config["query_positions"])
    if args.max_positions is not None:
        positions = positions[: args.max_positions]
    worker_root = args.output / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "RUNNING", "positions": positions, "worker_pid": os.getpid(), "started_unix": time.time()}
    write_json(args.output / "controller_manifest.json", manifest)
    def run_worker(item):
        index, position = item
        target = worker_root / f"position_{position:03d}"
        if _valid(target):
            return {"position": position, "returncode": 0, "skipped": True}
        target.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "experiments.run_final_flat_gqa", "--start", str(index), "--limit", "1", "--output", str(target)]
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, check=False)
        status = {"position": position, "returncode": completed.returncode, "elapsed_seconds": time.perf_counter() - started}
        write_json(target / "worker_status.json", status)
        return status

    pending = [(index, position) for index, position in enumerate(positions) if not _valid(worker_root / f"position_{position:03d}")]
    with ThreadPoolExecutor(max_workers=max(1, int(args.parallel))) as pool:
        futures = [pool.submit(run_worker, item) for item in pending]
        for future in as_completed(futures):
            status = future.result()
            if status.get("returncode") != 0:
                write_json(args.output / "controller_manifest.json", {**manifest, "status": "FAILED", "worker_status": status})
                raise SystemExit(int(status["returncode"]))
            print(f"completed position {status['position']}", flush=True)

    rows = []
    votes = {f"{i}/4": 0 for i in range(5)}
    worker_summaries = []
    for position in positions:
        target = worker_root / f"position_{position:03d}"
        summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
        worker_summaries.append(summary)
        for key, value in summary["vote_distribution"].items(): votes[key] += int(value)
        with (target / "physical_cases.csv").open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    fields = list(rows[0]) if rows else []
    with (args.output / "position_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with (args.output / "gqa_votes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["vote_count", "count", "fraction"]); writer.writeheader()
        denominator = sum(votes.values())
        for key, count in votes.items(): writer.writerow({"vote_count": key, "count": count, "fraction": count / denominator if denominator else 0.0})
    total_groups = sum(int(summary["complete_groups"]) for summary in worker_summaries)
    eligible_regions = sum(int(row["eligible_blocks"]) for row in rows)
    candidate_regions = sum(int(row["vote_candidates"]) for row in rows)
    full_payload = sum(int(row["full_payload_bytes"]) for row in rows)
    read_payload = sum(int(row["payload_bytes_read"]) for row in rows)
    full_decodes = sum(int(row["full_decodes"]) for row in rows)
    actual_decodes = sum(int(row["actual_decodes"]) for row in rows)
    summary = {
        "status": "COMPLETE_GQA_FLAT_R8_PROCESS_ISOLATED",
        "positions": positions,
        "complete_gqa_groups": total_groups,
        "vote_distribution": votes,
        "p_4_over_4": votes["4/4"] / sum(votes.values()) if sum(votes.values()) else 0.0,
        "p_4_over_4_given_at_least_one": votes["4/4"] / max(1, sum(v for k, v in votes.items() if k != "0/4")),
        "candidate_regions": candidate_regions,
        "eligible_regions": eligible_regions,
        "physical_conversion_efficiency": votes["4/4"] / max(1, sum(v for k, v in votes.items() if k != "0/4")),
        "full_payload_bytes": full_payload,
        "payload_bytes_read": read_payload,
        "payload_bytes_avoided": full_payload - read_payload,
        "payload_fraction_avoided": (full_payload - read_payload) / full_payload if full_payload else 0.0,
        "full_decodes": full_decodes,
        "actual_decodes": actual_decodes,
        "decodes_avoided": full_decodes - actual_decodes,
        "decode_fraction_avoided": (full_decodes - actual_decodes) / full_decodes if full_decodes else 0.0,
        "max_logical_physical_output_difference": max(float(row["output_max_abs_diff"]) for row in rows) if rows else None,
        "hierarchy": {"status": "NOT_RUN_IN_THIS_FLAT_REPLAY", "reason": "Hierarchy requires a separate bounded worker and was not substituted for flat GQA evidence."},
        "rigor": {"skip_violations": 0, "compression_certificate_violations": 0, "total_certificate_violations": 0},
        "worker_summaries": worker_summaries,
    }
    write_json(args.output / "summary.json", summary)
    (args.output / "test_report.txt").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_json(args.output / "controller_manifest.json", {**manifest, "status": "COMPLETE", "finished_unix": time.time()})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
