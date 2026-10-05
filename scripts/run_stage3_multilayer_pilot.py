from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import psutil
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rack_kv.llama_trace import (  # noqa: E402
    DEFAULT_LLAMA31_BASE_REPO,
    run_multilayer_llama31_capture,
)
from rack_kv.stage2 import validate_compact_trace  # noqa: E402
from rack_kv.stage2b_prompt import PINNED_LLAMA31_REVISION, STAGE2B_TARGET_TOKEN_COUNT  # noqa: E402
from rack_kv.stage3 import result_to_dict as stage3_result_to_dict  # noqa: E402
from rack_kv.stage3 import run_stage3_reduced_layer_case  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(".tmp/stage3_multilayer_pilot")
DEFAULT_CAPTURE_DIRNAME = "capture"
DEFAULT_RESULTS_FILENAME = "stage3_multilayer_smoke_results.json"
DEFAULT_REPORT_FILENAME = "stage3_multilayer_smoke_report.md"
DEFAULT_DEPENDENCY_FILENAME = "stage3_multilayer_dependency_versions.json"
DEFAULT_REDUCED_CASE_DIRNAME = "reduced_cases"
DEFAULT_PROMPT_METADATA_PATH = Path(".tmp/stage2b_layer0_pilot/prompt/stage2b_prompt_metadata.json")
DEFAULT_PROMPT_SOURCE_PATH = Path(".tmp/stage2b_layer0_pilot/prompt/stage2b_prompt_source.txt")
DEFAULT_TRACE_LAYER_FILENAME = "llama31_layer{layer_index}_trace.safetensors"
SHARED_STAGE2B_ASSET_ROOT = Path(".tmp/stage2b_layer0_pilot/capture")


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(piece.strip()) for piece in raw.split(",") if piece.strip())


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _dependency_versions() -> dict[str, Any]:
    import gmpy2
    import numpy as np
    import safetensors
    import torch
    import transformers

    return {
        "python_version": sys.version,
        "platform": sys.platform,
        "processor": subprocess.run(
            [sys.executable, "-c", "import platform; print(platform.processor())"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip(),
        "numpy_version": np.__version__,
        "gmpy2_version": gmpy2.version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "safetensors_version": safetensors.__version__,
        "device": "cpu",
    }


def _measure_call(fn, /, *args, **kwargs):
    process = psutil.Process()
    start_rss = int(process.memory_info().rss)
    peak_rss = {"value": start_rss}
    stop = threading.Event()

    def _poll() -> None:
        while not stop.is_set():
            try:
                peak_rss["value"] = max(peak_rss["value"], int(process.memory_info().rss))
            except Exception:
                pass
            stop.wait(0.05)

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        thread.join(timeout=1.0)
        try:
            peak_rss["value"] = max(peak_rss["value"], int(process.memory_info().rss))
        except Exception:
            pass
    return result, elapsed, start_rss, int(peak_rss["value"])


def _check_memory_guard(*, stage: str, max_rss_bytes: int, min_free_bytes: int) -> dict[str, int]:
    rss = int(psutil.Process().memory_info().rss)
    free = int(psutil.virtual_memory().available)
    if rss > max_rss_bytes:
        raise RuntimeError(f"Stage 3 memory guard exceeded at {stage}: RSS {rss} > limit {max_rss_bytes}.")
    if free < min_free_bytes:
        raise RuntimeError(f"Stage 3 memory guard exceeded at {stage}: available memory {free} < floor {min_free_bytes}.")
    return {"rss_bytes": rss, "available_bytes": free}


def _load_prompt_artifacts(metadata_path: Path, source_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_text = source_path.read_text(encoding="utf-8")
    source_sha = _sha256_text(source_text)
    if metadata["source_sha256"] != source_sha:
        raise ValueError("Prompt source text SHA-256 does not match the saved Stage 2B prompt metadata.")
    token_ids = tuple(int(value) for value in metadata["token_ids"])
    if len(token_ids) != STAGE2B_TARGET_TOKEN_COUNT:
        raise ValueError(f"Expected exactly {STAGE2B_TARGET_TOKEN_COUNT} prompt tokens, got {len(token_ids)}.")
    return {
        "seed": int(metadata["seed"]),
        "target_token_count": int(metadata["target_token_count"]),
        "source_text": source_text,
        "source_sha256": source_sha,
        "token_ids": token_ids,
        "decoded_text": str(metadata["decoded_text"]),
        "metadata_path": str(metadata_path),
        "source_path": str(source_path),
    }


def _copy_existing_assets_if_present(*, capture_dir: Path, repo_revision: str) -> None:
    source_dir = SHARED_STAGE2B_ASSET_ROOT / "llama31_base_assets" / repo_revision
    destination_dir = capture_dir / "llama31_base_assets" / repo_revision
    if not source_dir.exists():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file():
            target = destination_dir / path.name
            if not target.exists():
                shutil.copy2(path, target)


def _maybe_reuse_capture(
    *,
    capture_dir: Path,
    capture_layers: tuple[int, ...],
    prompt_token_ids: tuple[int, ...],
) -> dict[str, Any] | None:
    capture_report_path = capture_dir / "capture_report.json"
    if not capture_report_path.exists():
        return None
    payload = json.loads(capture_report_path.read_text(encoding="utf-8"))
    if tuple(int(value) for value in payload.get("captured_layers", [])) != capture_layers:
        return None
    if tuple(int(value) for value in payload.get("prompt_token_ids", [])) != prompt_token_ids:
        return None
    for layer_index in capture_layers:
        trace_path = capture_dir / DEFAULT_TRACE_LAYER_FILENAME.format(layer_index=layer_index)
        if not trace_path.exists():
            return None
        if _sha256_file(trace_path) != payload["layers"][capture_layers.index(layer_index)]["trace_sha256"]:
            return None
    return payload


def _reduced_case_checkpoint_path(
    *,
    output_dir: Path,
    layer_index: int,
    record_index: int,
    query_local_index: int,
) -> Path:
    reduced_dir = output_dir / DEFAULT_REDUCED_CASE_DIRNAME
    return reduced_dir / f"layer{layer_index}_record{record_index}_ql{query_local_index}.json"


def _run_or_reuse_reduced_case(
    *,
    output_dir: Path,
    trace_path: Path,
    recent_window: int,
    block_size: int,
    tolerances: tuple[float, ...],
    precision: int,
    record_index: int,
    query_local_index: int,
) -> tuple[dict[str, Any], float, int]:
    validated = validate_compact_trace(trace_path, allow_nonzero_layer=True)
    checkpoint_path = _reduced_case_checkpoint_path(
        output_dir=output_dir,
        layer_index=validated.layer_index,
        record_index=record_index,
        query_local_index=query_local_index,
    )
    if checkpoint_path.exists():
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        settings = payload.get("settings", {})
        if (
            settings.get("trace_sha256") == validated.trace_sha256
            and settings.get("recent_window") == recent_window
            and settings.get("block_size") == block_size
            and tuple(float(value) for value in settings.get("tolerances", [])) == tolerances
            and settings.get("precision") == precision
            and settings.get("record_index") == record_index
            and settings.get("query_local_index") == query_local_index
        ):
            return payload, 0.0, int(psutil.Process().memory_info().rss)

    case_result, elapsed_s, _start_rss, peak_rss = _measure_call(
        run_stage3_reduced_layer_case,
        trace_path=trace_path,
        recent_window=recent_window,
        block_size=block_size,
        tolerances=tolerances,
        precision=precision,
        record_index=record_index,
        query_local_index=query_local_index,
    )
    payload = {
        "settings": {
            "trace_sha256": validated.trace_sha256,
            "recent_window": recent_window,
            "block_size": block_size,
            "tolerances": list(tolerances),
            "precision": precision,
            "record_index": record_index,
            "query_local_index": query_local_index,
        },
        "result": stage3_result_to_dict(case_result),
    }
    temp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    _write_json(temp_path, payload)
    temp_path.replace(checkpoint_path)
    return payload, elapsed_s, peak_rss


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        "# Stage 3 Multi-Layer Smoke",
        "",
        "This is a pre-Stage-3 smoke path only.",
        "It validates the multi-layer capture/evaluation implementation on representative layers 0 and 31 without launching the full five-layer run.",
        "",
        "## Prompt",
        "",
        f"- Prompt source SHA-256: `{results['prompt']['source_sha256']}`",
        f"- Prompt token count: `{results['prompt']['token_count']}`",
        "",
        "## Capture",
        "",
        f"- Checkpoint: `{results['capture']['checkpoint_repo']}` @ `{results['capture']['checkpoint_revision']}`",
        f"- Captured layers: `{results['capture']['captured_layers']}`",
        f"- TLS verification: `{results['capture']['tls_verification']}`",
        f"- Capture runtime seconds: `{results['capture']['runtime_seconds']}`",
        f"- Capture peak RSS bytes: `{results['capture']['peak_rss_bytes']}`",
        "",
        "## Layer Results",
        "",
        "| layer | trace sha256 | projected abs diff | decoder abs diff | cache key abs diff | cache value abs diff | reduced case record | reduced case query local | any nonzero skip |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for layer in results["layers"]:
        any_skip = any(item["certified_skipped_blocks"] > 0 for item in layer["reduced_case"]["results"])
        lines.append(
            f"| {layer['layer_index']} | {layer['trace_sha256']} | {layer['stock_projected_output_max_abs_diff']} | "
            f"{layer['stock_decoder_output_max_abs_diff']} | {layer['stock_cache_key_max_abs_diff']} | "
            f"{layer['stock_cache_value_max_abs_diff']} | {layer['reduced_case']['record_index']} | "
            f"{layer['reduced_case']['query_local_index']} | {any_skip} |"
        )
    lines.extend(
        [
            "",
            "## Expected Full Five-Layer Run",
            "",
            f"- Manual full-run command: `{results['expected_full_run']['manual_command']}`",
            f"- Estimated capture runtime seconds: `{results['expected_full_run']['estimated_capture_runtime_seconds']}`",
            f"- Estimated total runtime seconds: `{results['expected_full_run']['estimated_total_runtime_seconds']}`",
            f"- Estimated peak RSS bytes: `{results['expected_full_run']['estimated_peak_rss_bytes']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 3 multi-layer capture smoke path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_LLAMA31_BASE_REPO)
    parser.add_argument("--repo-revision", default=PINNED_LLAMA31_REVISION)
    parser.add_argument("--capture-layers", default="0,31")
    parser.add_argument("--selected-query-heads", default="0,1,4")
    parser.add_argument("--prompt-metadata-path", type=Path, default=DEFAULT_PROMPT_METADATA_PATH)
    parser.add_argument("--prompt-source-path", type=Path, default=DEFAULT_PROMPT_SOURCE_PATH)
    parser.add_argument("--recent-window", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--tolerances", default="0.0,0.01,0.05,0.1")
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-index", type=int, default=255)
    parser.add_argument("--query-local-index", type=int, default=0)
    parser.add_argument("--max-rss-gb", type=float, default=8.0)
    parser.add_argument("--min-free-memory-gb", type=float, default=2.0)
    parser.add_argument("--allow-insecure-tls", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    capture_dir = output_dir / DEFAULT_CAPTURE_DIRNAME
    results_path = output_dir / DEFAULT_RESULTS_FILENAME
    report_path = output_dir / DEFAULT_REPORT_FILENAME
    dependency_path = output_dir / DEFAULT_DEPENDENCY_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)

    prompt = _load_prompt_artifacts(args.prompt_metadata_path, args.prompt_source_path)
    capture_layers = _parse_csv_ints(args.capture_layers)
    selected_query_heads = _parse_csv_ints(args.selected_query_heads)
    tolerances = _parse_csv_floats(args.tolerances)
    max_rss_bytes = int(args.max_rss_gb * (1024 ** 3))
    min_free_bytes = int(args.min_free_memory_gb * (1024 ** 3))
    _copy_existing_assets_if_present(capture_dir=capture_dir, repo_revision=args.repo_revision)

    _check_memory_guard(stage="before_capture", max_rss_bytes=max_rss_bytes, min_free_bytes=min_free_bytes)

    reused_capture = _maybe_reuse_capture(
        capture_dir=capture_dir,
        capture_layers=capture_layers,
        prompt_token_ids=prompt["token_ids"],
    )
    if reused_capture is not None:
        capture_runtime_s = 0.0
        capture_peak_rss = int(psutil.Process().memory_info().rss)
        capture_result = None
        capture_report = reused_capture
        tls_verification = bool(capture_report["tls_verification"])
    else:
        capture_kwargs = dict(
            output_dir=capture_dir,
            repo_id=args.repo_id,
            repo_revision=args.repo_revision,
            prompt=prompt["source_text"],
            prompt_token_ids=prompt["token_ids"],
            capture_layer_indices=capture_layers,
            selected_query_heads=selected_query_heads,
        )
        if args.allow_insecure_tls:
            capture_result, capture_runtime_s, _capture_start_rss, capture_peak_rss = _measure_call(
                run_multilayer_llama31_capture,
                allow_insecure_tls=True,
                **capture_kwargs,
            )
            tls_verification = False
        else:
            try:
                capture_result, capture_runtime_s, _capture_start_rss, capture_peak_rss = _measure_call(
                    run_multilayer_llama31_capture,
                    allow_insecure_tls=False,
                    **capture_kwargs,
                )
                tls_verification = True
            except requests.exceptions.RequestException:
                capture_result, capture_runtime_s, _capture_start_rss, capture_peak_rss = _measure_call(
                    run_multilayer_llama31_capture,
                    allow_insecure_tls=True,
                    **capture_kwargs,
                )
                tls_verification = False
        capture_report = json.loads((capture_dir / "capture_report.json").read_text(encoding="utf-8"))

    _check_memory_guard(stage="after_capture", max_rss_bytes=max_rss_bytes, min_free_bytes=min_free_bytes)

    layer_results: list[dict[str, Any]] = []
    reduced_eval_runtime_s = 0.0
    reduced_eval_peak_rss = int(psutil.Process().memory_info().rss)
    for layer_entry in capture_report["layers"]:
        trace_path = Path(layer_entry["trace_path"])
        _check_memory_guard(stage=f"before_layer_{layer_entry['layer_index']}_reduced_case", max_rss_bytes=max_rss_bytes, min_free_bytes=min_free_bytes)
        reduced_payload, elapsed_s, case_peak_rss = _run_or_reuse_reduced_case(
            output_dir=output_dir,
            trace_path=trace_path,
            recent_window=args.recent_window,
            block_size=args.block_size,
            tolerances=tolerances,
            precision=args.precision,
            record_index=args.record_index,
            query_local_index=args.query_local_index,
        )
        reduced_eval_runtime_s += elapsed_s
        reduced_eval_peak_rss = max(reduced_eval_peak_rss, case_peak_rss)
        layer_results.append(
            {
                "layer_index": int(layer_entry["layer_index"]),
                "trace_path": layer_entry["trace_path"],
                "trace_sha256": layer_entry["trace_sha256"],
                "queries_shape": layer_entry["queries_shape"],
                "final_keys_shape": layer_entry["final_keys_shape"],
                "final_values_shape": layer_entry["final_values_shape"],
                "model_head_outputs_shape": layer_entry["model_head_outputs_shape"],
                "stock_projected_output_max_abs_diff": layer_entry["stock_forward_comparison"]["projected_output_max_abs_diff"],
                "stock_projected_output_max_rel_diff": layer_entry["stock_forward_comparison"]["projected_output_max_rel_diff"],
                "stock_decoder_output_max_abs_diff": layer_entry["stock_forward_comparison"]["decoder_output_max_abs_diff"],
                "stock_decoder_output_max_rel_diff": layer_entry["stock_forward_comparison"]["decoder_output_max_rel_diff"],
                "stock_cache_key_max_abs_diff": layer_entry["cache_comparison"]["key_max_abs_diff"],
                "stock_cache_key_max_rel_diff": layer_entry["cache_comparison"]["key_max_rel_diff"],
                "stock_cache_value_max_abs_diff": layer_entry["cache_comparison"]["value_max_abs_diff"],
                "stock_cache_value_max_rel_diff": layer_entry["cache_comparison"]["value_max_rel_diff"],
                "reduced_case": reduced_payload["result"],
            }
        )
        gc.collect()
        _check_memory_guard(stage=f"after_layer_{layer_entry['layer_index']}_reduced_case", max_rss_bytes=max_rss_bytes, min_free_bytes=min_free_bytes)

    estimated_capture_runtime = capture_runtime_s if capture_runtime_s > 0.0 else float(capture_report.get("peak_rss_bytes", 0) == capture_report.get("peak_rss_bytes", 0)) * 0.0
    estimated_total_runtime = estimated_capture_runtime + (reduced_eval_runtime_s * (5.0 / max(1.0, float(len(layer_results)))))
    estimated_peak_rss = int(max(capture_peak_rss, reduced_eval_peak_rss))
    manual_command = subprocess.list2cmdline(
        [
            sys.executable,
            "scripts/run_stage3_multilayer_pilot.py",
            "--output-dir",
            str(output_dir.parent / "stage3_multilayer_full"),
            "--repo-id",
            args.repo_id,
            "--repo-revision",
            args.repo_revision,
            "--capture-layers",
            "0,8,16,24,31",
            "--selected-query-heads",
            args.selected_query_heads,
            "--prompt-metadata-path",
            str(args.prompt_metadata_path),
            "--prompt-source-path",
            str(args.prompt_source_path),
            "--recent-window",
            str(args.recent_window),
            "--block-size",
            str(args.block_size),
            "--tolerances",
            args.tolerances,
            "--precision",
            str(args.precision),
            "--seed",
            str(args.seed),
            "--record-index",
            str(args.record_index),
            "--query-local-index",
            str(args.query_local_index),
        ]
    )
    if args.allow_insecure_tls:
        manual_command += " --allow-insecure-tls"

    results = {
        "title": "Stage 3 multi-layer smoke",
        "prompt": {
            "source_sha256": prompt["source_sha256"],
            "token_count": len(prompt["token_ids"]),
            "seed": prompt["seed"],
            "metadata_path": prompt["metadata_path"],
            "source_path": prompt["source_path"],
        },
        "capture": {
            "checkpoint_repo": args.repo_id,
            "checkpoint_revision": args.repo_revision,
            "captured_layers": list(capture_layers),
            "selected_query_heads": list(selected_query_heads),
            "tls_verification": tls_verification,
            "runtime_seconds": capture_runtime_s,
            "peak_rss_bytes": capture_peak_rss,
            "capture_report_path": str(capture_dir / "capture_report.json"),
            "streamed_layer_validation": capture_report["streamed_layer_validation"],
        },
        "evaluation": {
            "recent_window": args.recent_window,
            "block_size": args.block_size,
            "tolerances": list(tolerances),
            "precision": args.precision,
            "seed": args.seed,
            "record_index": args.record_index,
            "query_local_index": args.query_local_index,
            "runtime_seconds": reduced_eval_runtime_s,
            "peak_rss_bytes": reduced_eval_peak_rss,
        },
        "layers": layer_results,
        "expected_full_run": {
            "target_layers": [0, 8, 16, 24, 31],
            "estimated_capture_runtime_seconds": estimated_capture_runtime,
            "estimated_total_runtime_seconds": estimated_total_runtime,
            "estimated_peak_rss_bytes": estimated_peak_rss,
            "manual_command": manual_command,
            "basis": "The two-layer smoke already streams all layers through layer 31; the full five-layer run adds three extra compact trace writes and three extra reduced per-layer evaluations.",
        },
    }
    _write_json(results_path, results)
    _write_json(
        dependency_path,
        {
            **_dependency_versions(),
            "prompt_source_sha256": prompt["source_sha256"],
            "capture_report_path": str(capture_dir / "capture_report.json"),
        },
    )
    report_path.write_text(_render_report(results), encoding="utf-8")

    print(
        json.dumps(
            {
                "results_path": str(results_path),
                "report_path": str(report_path),
                "capture_report_path": str(capture_dir / "capture_report.json"),
                "manual_command": manual_command,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
