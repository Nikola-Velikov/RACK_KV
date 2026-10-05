"""Compare like-for-like experiment populations; never silently mix quick/full scopes."""
import argparse
import json
from pathlib import Path
from .common import read_json


def compare(baseline, new):
    if baseline["identity"]["selection"] != new["identity"]["selection"]:
        raise ValueError("Incomparable query populations: selection differs")
    if baseline["identity"]["method_mode_plan"] != new["identity"]["method_mode_plan"]:
        raise ValueError("Incomparable baseline method/mode plans")
    deltas, warnings = {}, []
    fields = ("mean_serialized_bytes", "mean_compression_ratio", "mean_L2", "weighted_skip_fraction",
              "mean_certificate_bound_to_error_ratio_certified_cases")
    for method, old in baseline.get("representative", {}).items():
        if method not in new.get("representative", {}):
            warnings.append(f"Missing method {method}")
            continue
        current = new["representative"][method]
        if old["case_count"] != current["case_count"]:
            raise ValueError("Case count differs")
        deltas[method] = {key: current[key] - old[key] if current.get(key) is not None and old.get(key) is not None else None for key in fields}
        for key in ("false_safe_count", "rigorous_interval_violations", "numerical_fallbacks"):
            if current.get(key, 0) > old.get(key, 0):
                warnings.append(f"Invariant/regression warning: {method}/{key}")
    for method, old in baseline.get("full_model", {}).get("metrics", {}).items():
        current = new.get("full_model", {}).get("metrics", {}).get(method)
        if current is None:
            warnings.append(f"Full-model result unavailable: {method}")
        else:
            deltas["full_model:" + method] = {k: current[k] - old[k] for k in
                ("mean_nll", "perplexity", "mean_top1_agreement", "mean_kl_divergence", "mean_logit_l2_error")}
    if new.get("profiling", {}).get("random_access", {}).get("mismatches", 0):
        warnings.append("Independent decoding invariant regressed")
    if not new.get("scientific_source_unchanged", False):
        warnings.append("Scientific source differs from frozen v1; identify the intended intervention")
    return {"delta_new_minus_v1": deltas, "warnings": warnings,
            "runtime_delta_seconds": new.get("runtime_seconds", 0) - baseline.get("runtime_seconds", 0),
            "runtime_caveat": "Orchestration runtime includes tests/profiling; not inference latency. Compare only identical modes/hardware."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("new_result_directory", type=Path)
    p.add_argument("--baseline", type=Path, default=Path("results/rack_kv_v1_baseline"))
    args = p.parse_args()
    print(json.dumps(compare(read_json(args.baseline / "summary.json"), read_json(args.new_result_directory / "summary.json")), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
