from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import shutil
import sys
import zipfile

import gmpy2
import numpy as np
import safetensors
import torch
import transformers

from rack_kv.stage2 import result_to_dict, run_stage2_smoke_experiment


DEFAULT_TRACE_PATH = Path(".tmp/llama31_capture/llama31_layer0_trace.safetensors")
DEFAULT_OUTPUT_DIR = Path(".tmp/stage2_layer0_smoke")
DEFAULT_RESULTS_FILENAME = "stage2_smoke_results.json"
DEFAULT_REPORT_FILENAME = "stage2_smoke_report.md"
DEFAULT_DEPENDENCY_FILENAME = "stage2_dependency_versions.json"
DEFAULT_COMMAND_FILENAME = "command.txt"
DEFAULT_TEST_COMMAND_FILENAME = "test_command.txt"
DEFAULT_TEST_RETURN_CODE_FILENAME = "test_return_code.txt"
DEFAULT_TEST_LOG_FILENAME = "stage2_layer0_smoke_test_log.txt"
DEFAULT_MANIFEST_FILENAME = "review_package_manifest.json"
DEFAULT_REVIEW_ZIP = "rack_kv_stage2_layer0_smoke_review.zip"
REVIEW_SOURCE_FILES = (
    "rack_kv/__init__.py",
    "rack_kv/accounting.py",
    "rack_kv/codec.py",
    "rack_kv/certificate.py",
    "rack_kv/ieee.py",
    "rack_kv/llama_trace.py",
    "rack_kv/rigorous.py",
    "rack_kv/stage2.py",
    "rack_kv/types.py",
    "tests/test_codec_and_accounting.py",
    "tests/test_llama_trace.py",
    "tests/test_rigorous_certificate.py",
    "tests/test_stage2_integration.py",
    "scripts/run_stage2_layer0_smoke.py",
)


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(piece.strip()) for piece in raw.split(",") if piece.strip())


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _dependency_versions() -> dict[str, str]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "gmpy2_version": gmpy2.version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "safetensors_version": safetensors.__version__,
        "device": "cpu",
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _try_git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except Exception:
        return None
    revision = result.stdout.strip()
    return revision or None


def _summarize_previous_results(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    configs = payload.get("configurations", [])
    if not configs:
        return None
    return {
        "max_reported_compression_error": max(
            (cfg.get("max_compression_error", cfg.get("max_true_compression_error", 0.0)) for cfg in configs),
            default=0.0,
        ),
        "max_reported_total_error": max(
            (cfg.get("max_total_observed_error", cfg.get("max_reference_total_error", 0.0)) for cfg in configs),
            default=0.0,
        ),
        "max_reported_observed_skipping_error": max(
            (cfg.get("max_observed_reconstructed_skipping_error", 0.0) for cfg in configs),
            default=0.0,
        ),
        "max_model_reference_gap": max((cfg.get("max_model_reference_gap", 0.0) for cfg in configs), default=0.0),
        "max_true_compression_error": max(
            (cfg.get("max_true_compression_error", cfg.get("max_compression_error", 0.0)) for cfg in configs),
            default=0.0,
        ),
        "max_rigorous_skip_error_upper": max((cfg.get("max_rigorous_skip_error_upper", 0.0) for cfg in configs), default=0.0),
        "max_reference_total_error": max(
            (cfg.get("max_reference_total_error", cfg.get("max_total_observed_error", 0.0)) for cfg in configs),
            default=0.0,
        ),
        "max_captured_model_total_gap": max((cfg.get("max_captured_model_total_gap", 0.0) for cfg in configs), default=0.0),
    }


def _source_snapshot_sha256(source_files: tuple[str, ...]) -> str:
    entries = []
    for relative_path in sorted(source_files):
        path = Path(relative_path)
        entries.append(f"{relative_path}\t{_sha256_file(path)}")
    payload = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_tests_capture(
    *,
    test_command_path: Path,
    test_return_code_path: Path,
    test_log_path: Path,
) -> dict[str, str | int]:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
    command_text = subprocess.list2cmdline(command)
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    test_command_path.write_text(command_text, encoding="utf-8")
    test_return_code_path.write_text(str(result.returncode), encoding="utf-8")
    test_log_path.write_text(result.stdout, encoding="utf-8")
    reopened = test_log_path.read_text(encoding="utf-8")
    if reopened != result.stdout:
        raise RuntimeError("UTF-8 test log verification failed after writing the captured unittest output.")
    if result.returncode != 0:
        raise RuntimeError(f"Full test suite failed with return code {result.returncode}.")
    return {
        "test_command": command_text,
        "test_return_code": result.returncode,
        "test_log_path": str(test_log_path),
    }


def _render_report(results: dict) -> str:
    trace = results["trace"]
    comparison = results.get("definition_change_comparison")
    snapshot_sha256 = results.get("source_snapshot_sha256")
    global_prefix = results["global_prefix_ratio_summary"]
    lines = [
        "# RACK-KV Stage 2 Layer-0 Smoke Experiment",
        "",
        "This is the first layer-0 integration experiment.",
        "The trace contains only 12 tokens.",
        "This is a smoke test, not a meaningful long-context benchmark.",
        "No external baseline comparison has been performed.",
        "No multi-layer or generation integration has been performed.",
        "The skipping certificate applies relative to reconstructed compressed KV.",
        "Compression error remains empirical.",
        "No novelty conclusion follows from this experiment alone.",
        "",
        "## Trace",
        "",
        f"- Trace file: `{trace['trace_path']}`",
        f"- Trace SHA-256: `{trace['trace_sha256']}`",
        f"- Checkpoint revision: `{trace['checkpoint_revision']}`",
        f"- Selected query heads: `{trace['selected_query_heads']}`",
        f"- Selected KV heads: `{trace['selected_kv_heads']}`",
        f"- Query-to-KV mapping: `{trace['query_to_kv_heads']}`",
        f"- Visible prefix lengths: `{trace['visible_lengths']}`",
        f"- Saved attention scaling: `{trace['scaling']}`",
        f"- Canonical 1/sqrt(head_dim): `{trace['canonical_one_over_sqrt_head_dim']}`",
        f"- Scaling difference: `{trace['scaling_minus_canonical']}`",
        f"- Source snapshot SHA-256: `{snapshot_sha256}`",
        "",
        "## Metric Definitions",
        "",
        "- `model_reference_gap = ||captured_model_output - original_reference_output||_2`",
        "- `compression_error = ||original_reference_output - reconstructed_full_output||_2`",
        "- `observed_skipping_error = ||reconstructed_full_output - reconstructed_kept_output||_2`",
        "- `reference_total_error = ||original_reference_output - reconstructed_kept_output||_2`",
        "- `captured_model_total_gap = ||captured_model_output - reconstructed_kept_output||_2`",
        "",
        "## Configurations",
        "",
        "| W | B | tol | agg ratio | final ratio | max model gap | max compression err | max skip err | max rigorous skip upper | max ref total err | max model total gap | skip frac | fallbacks | rigorous viols |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for config in results["configurations"]:
        lines.append(
            "| {recent_window} | {block_size} | {tolerance:.6f} | {aggregate_prefix_compression_ratio:.6f} | "
            "{final_prefix_compression_ratio:.6f} | {max_model_reference_gap:.6f} | {max_true_compression_error:.6f} | "
            "{max_observed_reconstructed_skipping_error:.6f} | {max_rigorous_skip_error_upper:.6f} | "
            "{max_reference_total_error:.6f} | {max_captured_model_total_gap:.6f} | {skipped_block_fraction:.6f} | "
            "{fallback_count} | {rigorous_interval_violation_count} |".format(
                **config
            )
        )
    lines.extend(
        [
            "",
            "## Memory Terminology",
            "",
            "- `aggregate_prefix_*` metrics sum all evaluated visible-prefix snapshots across the selected KV heads for this workload.",
            "- `final_prefix_*` metrics describe the final 12-token visible cache footprint across the selected KV heads only.",
            "- Aggregate workload bytes must not be interpreted as the memory footprint of one KV cache.",
            "- Some very short prefixes expand because block headers, metadata, and container indices dominate the payload.",
            "",
            "## Prefix Ratio Summary Across All Evaluated Prefixes",
            "",
            f"- Global minimum prefix ratio: `{global_prefix['minimum']}`",
            f"- Global mean prefix ratio: `{global_prefix['mean']}`",
            f"- Global median prefix ratio: `{global_prefix['median']}`",
            f"- Global maximum prefix ratio: `{global_prefix['maximum']}`",
            "",
            "## Per-Configuration Prefix Ratio Table",
            "",
            "| W | B | min prefix ratio | mean prefix ratio | median prefix ratio | max prefix ratio | final prefix ratio |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            "",
        ]
    )
    for config in results["configurations"]:
        lines.append(
            "| {recent_window} | {block_size} | {min_prefix_compression_ratio:.16f} | "
            "{mean_prefix_compression_ratio:.16f} | {median_prefix_compression_ratio:.16f} | "
            "{max_prefix_compression_ratio:.16f} | {final_prefix_compression_ratio:.16f} |".format(
                **config
            )
        )
    lines.append("")
    if comparison is not None:
        lines.extend(
            [
                "## Definition Change From Previous Report",
                "",
                f"- Previous reported `max_compression_error`: `{comparison['previous_max_reported_compression_error']}`",
                f"- Corrected `max_true_compression_error`: `{comparison['corrected_max_true_compression_error']}`",
                f"- Corrected `max_model_reference_gap`: `{comparison['corrected_max_model_reference_gap']}`",
                f"- Previous accepted `max_rigorous_skip_error_upper`: `{comparison['previous_max_rigorous_skip_error_upper']}`",
                f"- Previous accepted `max_reference_total_error`: `{comparison['previous_max_reference_total_error']}`",
                f"- Previous accepted `max_captured_model_total_gap`: `{comparison['previous_max_captured_model_total_gap']}`",
                "- The previous `compression_error` mixed true codec reconstruction error with the gap between the captured bfloat16 model head output and a high-precision replay on the original saved KV tensors.",
                "- The corrected `compression_error` now measures only `||original_reference_output - reconstructed_full_output||_2`.",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _package_review(
    *,
    output_dir: Path,
    review_zip_path: Path,
    trace_path: Path,
    results_path: Path,
    report_path: Path,
    dependency_path: Path,
    command_path: Path,
    test_command_path: Path,
    test_return_code_path: Path,
    manifest_output_path: Path,
    test_log_path: Path,
) -> dict:
    package_root = output_dir / "review_package"
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    for relative_path in REVIEW_SOURCE_FILES:
        source = Path(relative_path)
        destination = package_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_files.append(relative_path)

    trace_sidecars = [
        trace_path,
        trace_path.with_name("capture_report.json"),
        trace_path.with_name("dependency_versions.json"),
    ]
    for source_path in (*trace_sidecars, results_path, report_path, dependency_path, command_path, test_command_path, test_return_code_path):
        if not source_path.exists():
            continue
        destination = package_root / source_path.name
        shutil.copy2(source_path, destination)
        copied_files.append(destination.relative_to(package_root).as_posix())

    if not test_log_path.exists():
        raise FileNotFoundError(f"Expected UTF-8 test log does not exist: {test_log_path}")
    destination = package_root / test_log_path.name
    shutil.copy2(test_log_path, destination)
    copied_files.append(destination.relative_to(package_root).as_posix())

    source_snapshot_sha256 = _source_snapshot_sha256(REVIEW_SOURCE_FILES)
    manifest = {
        "trace_sha256": _sha256_file(trace_path),
        "source_revision": _try_git_revision(),
        "source_snapshot_sha256": source_snapshot_sha256,
        "files": [],
    }
    for path in sorted(package_root.rglob("*")):
        if path.is_file():
            manifest["files"].append(
                {
                    "relative_path": path.relative_to(package_root).as_posix(),
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    manifest_file = package_root / DEFAULT_MANIFEST_FILENAME
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest_output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    copied_files.append(DEFAULT_MANIFEST_FILENAME)

    with zipfile.ZipFile(review_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in package_root.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(package_root))

    return {
        "review_zip_path": str(review_zip_path),
        "copied_files": copied_files,
        "manifest_path": str(manifest_output_path),
        "trace_sha256": manifest["trace_sha256"],
        "source_revision": manifest["source_revision"],
        "source_snapshot_sha256": source_snapshot_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the first Stage 2 layer-0 RACK-KV smoke experiment.")
    parser.add_argument("--trace-path", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recent-windows", type=str, default="1,2,4")
    parser.add_argument("--block-sizes", type=str, default="2,4")
    parser.add_argument("--tolerances", type=str, default="0.0,0.01,0.05,0.1")
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-log-path", type=Path, default=None)
    parser.add_argument("--review-zip-path", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_zip_path = args.review_zip_path or (args.output_dir / DEFAULT_REVIEW_ZIP)
    results_path = args.output_dir / DEFAULT_RESULTS_FILENAME
    report_path = args.output_dir / DEFAULT_REPORT_FILENAME
    dependency_path = args.output_dir / DEFAULT_DEPENDENCY_FILENAME
    command_path = args.output_dir / DEFAULT_COMMAND_FILENAME
    test_command_path = args.output_dir / DEFAULT_TEST_COMMAND_FILENAME
    test_return_code_path = args.output_dir / DEFAULT_TEST_RETURN_CODE_FILENAME
    manifest_path = args.output_dir / DEFAULT_MANIFEST_FILENAME
    test_log_path = args.test_log_path or (args.output_dir / DEFAULT_TEST_LOG_FILENAME)
    previous_summary = _summarize_previous_results(results_path)

    recent_windows = _parse_csv_ints(args.recent_windows)
    block_sizes = _parse_csv_ints(args.block_sizes)
    tolerances = _parse_csv_floats(args.tolerances)
    test_capture = _run_tests_capture(
        test_command_path=test_command_path,
        test_return_code_path=test_return_code_path,
        test_log_path=test_log_path,
    )

    result = run_stage2_smoke_experiment(
        trace_path=args.trace_path,
        recent_windows=recent_windows,
        block_sizes=block_sizes,
        tolerances=tolerances,
        precision=args.precision,
        random_seed=args.seed,
    )
    result_dict = result_to_dict(result)
    if previous_summary is not None:
        result_dict["definition_change_comparison"] = {
            "previous_max_reported_compression_error": previous_summary["max_reported_compression_error"],
            "previous_max_reported_total_error": previous_summary["max_reported_total_error"],
            "previous_max_reported_observed_skipping_error": previous_summary["max_reported_observed_skipping_error"],
            "previous_max_rigorous_skip_error_upper": previous_summary["max_rigorous_skip_error_upper"],
            "previous_max_reference_total_error": previous_summary["max_reference_total_error"],
            "previous_max_captured_model_total_gap": previous_summary["max_captured_model_total_gap"],
            "corrected_max_true_compression_error": max(
                (cfg["max_true_compression_error"] for cfg in result_dict["configurations"]),
                default=0.0,
            ),
            "corrected_max_model_reference_gap": max(
                (cfg["max_model_reference_gap"] for cfg in result_dict["configurations"]),
                default=0.0,
            ),
        }
    result_dict["source_snapshot_sha256"] = _source_snapshot_sha256(REVIEW_SOURCE_FILES)
    result_dict["test_capture"] = test_capture
    _write_json(results_path, result_dict)
    report_path.write_text(_render_report(result_dict), encoding="utf-8")
    _write_json(dependency_path, _dependency_versions())
    command_path.write_text(" ".join(sys.argv), encoding="utf-8")

    manifest = _package_review(
        output_dir=args.output_dir,
        review_zip_path=review_zip_path,
        trace_path=args.trace_path,
        results_path=results_path,
        report_path=report_path,
        dependency_path=dependency_path,
        command_path=command_path,
        test_command_path=test_command_path,
        test_return_code_path=test_return_code_path,
        manifest_output_path=manifest_path,
        test_log_path=test_log_path,
    )
    print(json.dumps({"results_path": str(results_path), "report_path": str(report_path), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
