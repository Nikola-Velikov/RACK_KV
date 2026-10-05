"""Single entry point; all scientific operations delegate to frozen v1 functions."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import shutil

from .common import ROOT, environment, inventory, load_config, read_json, seed_execution, sha256, verify_frozen_input, verify_review_archive, write_csv, write_json


def run_tests(output, scope="full"):
    command = [sys.executable, "-m", "experiments.run_tests"]
    if scope == "invariants":
        command += ["--invariants"]
    if (output / "test_report.txt").exists():
        old_hash = sha256(output / "test_report.txt")[:16]
        history = output / "test_history" / old_hash
        history.mkdir(parents=True, exist_ok=True)
        for name in ("test_report.txt", "test_command.txt", "test_return_code.txt"):
            if (output / name).exists():
                shutil.copy2(output / name, history / name)
    (output / "test_command.txt").write_text(subprocess.list2cmdline(command), encoding="utf-8")
    with (output / "test_report.txt").open("w", encoding="utf-8") as handle:
        run = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    (output / "test_return_code.txt").write_text(str(run.returncode), encoding="utf-8")
    if run.returncode:
        raise RuntimeError("Deterministic test suite failed; see test_report.txt")


def invoke_script(script, arguments, config_path, output, name):
    command = [sys.executable, "-m", "experiments.reproduce_v1", "--config", str(config_path),
               "--worker-script", script, "--", *map(str, arguments)]
    print("Executing " + subprocess.list2cmdline(command), flush=True)
    with (output / f"{name}_log.txt").open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{script} failed ({result.returncode}); see {name}_log.txt")


def ensure_capture(config, output, args):
    capture = ROOT / config["existing_capture"]
    if args.recapture or not capture.exists():
        capture = output / "capture_run/capture"
        flags = ["--output-dir", capture.parent, "--repo-id", config["model"], "--repo-revision", config["revision"],
                 "--capture-layers", ",".join(map(str, config["layers"])),
                 "--selected-query-heads", ",".join(map(str, config["selected_query_heads"])),
                 "--prompt-metadata-path", config["prompt_metadata"], "--prompt-source-path", config["prompt_source"],
                 "--recent-window", config["recent_window"], "--block-size", config["block_size"],
                 "--precision", config["mpfr_precision"], "--seed", config["seed"],
                 "--max-rss-gb", config["max_rss_gb"], "--min-free-memory-gb", config["min_free_gb"]]
        if args.allow_insecure_tls:
            flags.append("--allow-insecure-tls")
        invoke_script("scripts/run_stage3_multilayer_pilot.py", flags, args.config, output, "capture")
    else:
        for layer in config["layers"]:
            verify_frozen_input(capture / f"llama31_layer{layer}_trace.safetensors")
    return capture


def evaluate(config, selection, capture, output):
    from rack_kv.stage2 import validate_compact_trace
    from rack_kv.stage4 import build_stage4_case_context, run_baseline_case, stage4_case_result_to_dict
    from rack_kv import stage2
    from unittest.mock import patch
    rows = []
    for layer in selection["layers"]:
        trace = validate_compact_trace(capture / f"llama31_layer{layer}_trace.safetensors", allow_nonzero_layer=True)
        if trace.checkpoint_revision != config["revision"] or list(trace.selected_query_heads) != config["selected_query_heads"]:
            raise ValueError("Trace/model/head configuration mismatch")
        for position in selection["query_positions"]:
            for ql in selection["query_local_indices"]:
                context = None
                target = None
                for method, mode in config["method_mode_plan"]:
                    key = f"layer{layer}_pos{position}_ql{ql}__{method}__{mode}"
                    checkpoint = output / "cases" / f"{key}.json"
                    relative = f"payloads/{key}.bin"
                    if checkpoint.exists():
                        row = read_json(checkpoint)
                        if sha256(output / relative) != row["payload_sha256"]:
                            raise ValueError(f"Resume payload mismatch: {key}")
                    else:
                        if context is None:
                            context = build_stage4_case_context(trace=trace, record_index=trace.query_positions.index(position),
                                query_local_index=ql, recent_window=config["recent_window"], precision=config["mpfr_precision"])
                        original_progressive = stage2.progressive_certification_steps
                        observed_iterations = []

                        def observe(*a, **kw):
                            steps = original_progressive(*a, **kw)
                            observed_iterations.append(len(steps))
                            return steps

                        with patch.object(stage2, "progressive_certification_steps", side_effect=observe):
                            result, payload = run_baseline_case(context=context, method_name=method, mode=mode, trace=trace,
                                recent_window=config["recent_window"], block_size=config["block_size"], tolerance=config["epsilon"],
                                precision=config["mpfr_precision"], payload_relative_path=relative, rack_target_bytes=target)
                        (output / "payloads").mkdir(exist_ok=True)
                        (output / relative).write_bytes(payload)
                        row = stage4_case_result_to_dict(result)
                        if method == "rack_kv":
                            row["reproduction_observer"] = {"progressive_function_calls": len(observed_iterations),
                                                           "progressive_returned_steps": sum(observed_iterations)}
                        write_json(checkpoint, row)
                    if method == "rack_kv":
                        target = row["total_serialized_bytes"]
                    rows.append(row)
                print(f"Evaluated layer={layer} position={position} query_head={trace.selected_query_heads[ql]} ({len(rows)} method rows)", flush=True)
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/rack_kv_v1.yaml"))
    p.add_argument("--output", type=Path)
    p.add_argument("--test-scope", choices=["full", "invariants"], default="full")
    p.add_argument("--skip-tests", action="store_true", help="Resume only: reuse an existing successful test log in this run directory.")
    for flag in ("serializer-tests", "representative", "certificate-safety", "full-model", "baselines", "all", "quick", "from-existing", "recapture", "resume", "allow-insecure-tls"):
        p.add_argument("--" + flag, action="store_true")
    p.add_argument("--worker-script", choices=["scripts/run_stage3_multilayer_pilot.py", "scripts/run_stage5_quality_smoke.py"])
    p.add_argument("worker_args", nargs=argparse.REMAINDER)
    return p


def main():
    args = parser().parse_args()
    os.chdir(ROOT)
    config = load_config(args.config)
    if os.environ.get("PYTHONHASHSEED") != str(config["seed"]):
        env = dict(os.environ, PYTHONHASHSEED=str(config["seed"]), OMP_NUM_THREADS=str(config["torch_threads"]),
                   MKL_NUM_THREADS=str(config["torch_threads"]), CUBLAS_WORKSPACE_CONFIG=":4096:8")
        return subprocess.call([sys.executable, "-m", "experiments.reproduce_v1", *sys.argv[1:]], env=env)
    seed_execution(config)
    if args.worker_script:
        sys.argv = [args.worker_script, *[a for a in args.worker_args if a != "--"]]
        runpy.run_path(args.worker_script, run_name="__main__")
        return 0
    if args.quick and (args.from_existing or args.full_model or args.all or args.recapture):
        raise ValueError("--quick is a real-trace local replay subset; cannot combine with full/model/import/capture modes")
    chosen = any((args.serializer_tests, args.representative, args.certificate_safety, args.baselines, args.full_model, args.all, args.from_existing))
    if not chosen:
        args.quick = True
    output = args.output or Path(config["output_root"]) / ("rack_kv_v1_baseline" if args.from_existing else time.strftime("v1_%Y%m%d_%H%M%S"))
    output = output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError(f"Output exists: {output}; use --resume only for the same frozen experiment")
    if output.exists() and any(output.iterdir()) and not (output / "run_identity.json").exists():
        raise ValueError("Refusing to write into a directory without a reproduction run identity")
    output.mkdir(parents=True, exist_ok=True)
    selection = config["quick"] if args.quick else {k: config[k] for k in ("layers", "query_positions", "query_local_indices")}
    identity = {"configuration_hash": sha256(args.config), "selection": selection,
                "from_existing": args.from_existing, "full_model_requested": args.all or args.full_model or args.from_existing,
                "method_mode_plan": config["method_mode_plan"]}
    if (output / "run_identity.json").exists() and read_json(output / "run_identity.json") != identity:
        raise ValueError("Resume experiment identity mismatch")
    write_json(output / "run_identity.json", identity)
    manifest = environment(config)
    manifest.update({"schema": "rack_kv_v1_run_1", "identity": identity, "status": "running",
                     "tls_verification": not args.allow_insecure_tls,
                     "data_status": "historical_reaggregation" if args.from_existing else "new_computation",
                     "real_model_computation_requested": bool((args.all or args.full_model or args.recapture) and not args.from_existing)})
    write_json(output / "manifest.json", manifest)
    for folder in ("experiments", "rack_kv", "scripts", "tests"):
        for file in (ROOT / folder).glob("*.py"):
            target = output / "source_snapshot" / folder / file.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
    for file in (ROOT / "configs").rglob("*"):
        if file.is_file():
            target = output / "source_snapshot" / file.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
    started = time.perf_counter()
    summary = {"schema": "rack_kv_v1_summary_1", "identity": identity,
               "scientific_source_unchanged": True, "full_model": {"status": "not_requested"}}
    try:
        if args.skip_tests:
            if not args.resume or not (output / "test_return_code.txt").exists() or (output / "test_return_code.txt").read_text().strip() != "0":
                raise ValueError("--skip-tests requires --resume and existing successful test evidence")
            manifest["tests_reused_from_same_run"] = True
        elif args.serializer_tests or args.all or args.quick or args.from_existing:
            print("Running deterministic test suite; log: " + str(output / "test_report.txt"), flush=True)
            run_tests(output, args.test_scope)
        local = args.all or args.representative or args.baselines or args.certificate_safety or args.quick or args.from_existing
        if local:
            if args.from_existing:
                evidence = ROOT / config["existing_representative"]
                manifest["representative_evidence"] = verify_frozen_input(evidence)
                manifest["review_archives"] = [verify_review_archive(ROOT / ".tmp" / name) for name in
                    ("stage4_baselines_full_review.zip", "stage5_quality_final_review.zip")]
                source = read_json(evidence)
                rows = source["case_results"]
                expected = len(config["layers"]) * len(config["query_positions"]) * len(config["query_local_indices"]) * len(config["method_mode_plan"])
                if len(rows) != expected or source["actual_result_count"] != source["expected_result_count"]:
                    raise ValueError("Historical representative experiment is incomplete")
                capture, payload_root = ROOT / config["existing_capture"], evidence.parent
                for layer in config["layers"]:
                    verify_frozen_input(capture / f"llama31_layer{layer}_trace.safetensors")
                summary["representative_status"] = "historical_results_reaggregated_and_reprofiled"
            else:
                capture = ensure_capture(config, output, args)
                rows = evaluate(config, selection, capture, output)
                payload_root = output
                summary["representative_status"] = "recomputed_from_existing_traces" if not args.recapture else "recomputed_from_new_capture"
            from .results import summarize
            from .profiling import profile_results
            summary["representative"] = summarize(rows)
            write_csv(output / "representative_cases.csv", rows)
            write_json(output / "representative_records.json", rows)
            summary["profiling"] = profile_results(rows, payload_root, capture, config, output)
            from .plots import make_plots
            make_plots(output)
        if args.all or args.full_model or args.from_existing:
            from .results import export_full_model
            full_path = ROOT / config["existing_full_model"]
            if not args.from_existing:
                if not local:
                    capture = ensure_capture(config, output, args)
                flags = ["--output-dir", output / "full_model", "--review-zip", output / "full_model_review.zip",
                         "--capture-dir", capture, "--tensor-cache-dir", config["tensor_cache"],
                         "--repo-id", config["model"], "--repo-revision", config["revision"],
                         "--prompt-names", ",".join(config["full_model"]["prompts"]),
                         "--method-names", ",".join(config["full_model"]["methods"]),
                         "--prompt-token-count", config["full_model"]["token_count"],
                         "--max-rss-gb", config["max_rss_gb"], "--min-free-gb", config["min_free_gb"]]
                if args.allow_insecure_tls:
                    flags.append("--allow-insecure-tls")
                invoke_script("scripts/run_stage5_quality_smoke.py", flags, args.config, output, "full_model")
                invoke_script("scripts/run_stage5_quality_smoke.py", ["--output-dir", output / "full_model",
                    "--review-zip", output / "full_model_review.zip", "--finalize-from-checkpoints",
                    "--prompt-names", ",".join(config["full_model"]["prompts"])], args.config, output, "finalize")
                full_path = output / "full_model/stage5_quality_final_results.json"
            summary["full_model"] = export_full_model(full_path, output, config, historical=args.from_existing)
        else:
            write_csv(output / "full_model_results.csv", [], fields=["prompt", "method", "status"])
        load_config(args.config)
        summary["runtime_seconds"] = time.perf_counter() - started
        summary["runtime_scope"] = "current_orchestration_tests_and_measurements_only"
        write_json(output / "summary.json", summary)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        manifest["files"] = inventory(output)
        write_json(output / "manifest.json", manifest)
    print(str(output / "summary.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
