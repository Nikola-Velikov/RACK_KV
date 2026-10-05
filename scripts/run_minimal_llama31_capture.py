from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rack_kv.llama_trace import run_minimal_llama31_capture


def _parse_optional_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8")
    return args.prompt


def _parse_optional_prompt_token_ids(args: argparse.Namespace) -> tuple[int, ...] | None:
    if args.prompt_token_ids_json is None:
        return None
    payload = json.loads(args.prompt_token_ids_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("token_ids")
    if not isinstance(payload, list) or not payload:
        raise ValueError("prompt_token_ids_json must contain a non-empty JSON list of token IDs or an object with token_ids.")
    return tuple(int(value) for value in payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal layer-0 Llama-3.1-8B KV trace capture.")
    parser.add_argument("--output-dir", default=str(Path(".tmp") / "llama31_capture"))
    parser.add_argument("--repo-id", default="NousResearch/Meta-Llama-3.1-8B")
    parser.add_argument("--repo-revision", default=None)
    parser.add_argument("--prompt", default="RACK-KV trace prompt for offline attention replay.")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--prompt-token-ids-json", type=Path, default=None)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument(
        "--selected-query-heads",
        default=None,
        help="Comma-separated global query-head indices to capture, e.g. 0,1,4. If omitted, a config-derived default is used.",
    )
    parser.add_argument("--allow-insecure-tls", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    selected_query_heads = None
    if args.selected_query_heads is not None:
        selected_query_heads = tuple(int(part.strip()) for part in args.selected_query_heads.split(",") if part.strip())
    result = run_minimal_llama31_capture(
        output_dir=output_dir,
        repo_id=args.repo_id,
        repo_revision=args.repo_revision,
        prompt=_parse_optional_prompt(args),
        prompt_token_ids=_parse_optional_prompt_token_ids(args),
        layer_index=args.layer_index,
        selected_query_heads=selected_query_heads,
        allow_insecure_tls=args.allow_insecure_tls,
    )
    print(
        json.dumps(
            {
                "trace_path": str(result.trace_path),
                "capture_report_path": str(result.capture_report_path),
                "dependency_report_path": str(result.dependency_report_path),
                "repo_id": result.checkpoint.repo_id,
                "repo_revision": result.checkpoint.repo_revision,
                "num_hidden_layers": result.checkpoint.num_hidden_layers,
                "num_attention_heads": result.checkpoint.num_attention_heads,
                "num_key_value_heads": result.checkpoint.num_key_value_heads,
                "head_dim": result.checkpoint.head_dim,
                "model_dtype": result.checkpoint.model_dtype,
                "rope_theta": result.checkpoint.rope_theta,
                "rope_scaling": result.checkpoint.rope_scaling,
                "max_position_embeddings": result.checkpoint.max_position_embeddings,
                "token_ids": list(result.token_ids),
                "layer_index": result.layer_index,
                "selected_query_heads": list(result.selected_query_heads),
                "selected_kv_heads": list(result.selected_kv_heads),
                "query_to_kv_heads": list(result.query_to_kv_heads),
                "queries_shape": list(result.queries_shape),
                "final_keys_shape": list(result.final_keys_shape),
                "final_values_shape": list(result.final_values_shape),
                "model_head_outputs_shape": list(result.model_head_outputs_shape),
                "source_dtype": result.source_dtype,
                "storage_dtype": result.storage_dtype,
                "tls_verification": result.tls_verification,
                "trace_schema": result.trace_schema,
                "max_roundtrip_abs_diff": result.max_roundtrip_abs_diff,
                "max_compact_replay_abs_diff": result.max_compact_replay_abs_diff,
                "compact_replay_tolerance": result.compact_replay_tolerance,
                "stock_projected_output_max_abs_diff": result.stock_projected_output_max_abs_diff,
                "stock_projected_output_max_rel_diff": result.stock_projected_output_max_rel_diff,
                "stock_decoder_output_max_abs_diff": result.stock_decoder_output_max_abs_diff,
                "stock_decoder_output_max_rel_diff": result.stock_decoder_output_max_rel_diff,
                "stock_cache_key_max_abs_diff": result.stock_cache_key_max_abs_diff,
                "stock_cache_key_max_rel_diff": result.stock_cache_key_max_rel_diff,
                "stock_cache_value_max_abs_diff": result.stock_cache_value_max_abs_diff,
                "stock_cache_value_max_rel_diff": result.stock_cache_value_max_rel_diff,
                "peak_rss_bytes": result.peak_rss_bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
