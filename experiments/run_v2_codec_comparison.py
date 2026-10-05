"""Compare independently decodable RACK-KV V2 codecs on frozen blocks."""
from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path

import numpy as np

from experiments.common import ROOT, load_config, read_json, write_csv, write_json
from experiments.run_anisotropic_shadow import _case_payload
from rack_kv.anisotropic import build_anisotropic_summary_from_keys, anisotropic_logit_upper_bound_from_exact, exact_vector
from rack_kv.codec_v2 import V2CompressedBlock, choose_adaptive_codecs, encode_v2_block
from rack_kv.stage2 import validate_compact_trace


VARIANTS = ("v1_int8", "groupwise_int8", "hadamard_int8", "hadamard_int4", "adaptive")


def pct(values, p):
    return float(np.percentile(np.asarray(values, dtype=np.float64), p)) if values else 0.0


def l2_stats(errors):
    return {"mean": float(np.mean(errors)), "median": float(np.median(errors)), "p95": pct(errors, 95), "max": float(np.max(errors))}


def attention(query, keys, values, scale):
    logits = (keys @ query) * scale
    weights = np.exp(logits - np.max(logits))
    return (weights[:, None] * values).sum(axis=0) / weights.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/rack_kv_v2_codec")
    parser.add_argument("--group-size", type=int, default=16, choices=(16, 32))
    parser.add_argument("--geometry-sample", type=int, default=100, help="Number of deterministic case-blocks for MPFR geometry diagnostics.")
    args = parser.parse_args()
    config = load_config(ROOT / "configs/rack_kv_v1.yaml")
    stage4_root = ROOT / ".tmp/stage4_baselines_full"
    stage4 = read_json(ROOT / config["existing_representative"])
    rows = [row for row in stage4["case_results"] if row.get("method_name") == "rack_kv"]
    traces = {}
    blocks_out, attention_out, storage_out, timing_out, geometry_out, random_out = [], [], [], [], [], []
    geometry_seen = 0
    totals = {name: {"bytes": 0, "v1_bytes": 0, "fp16_bytes": 0, "key": [], "value": [], "attention": [], "key_codec": [], "value_codec": [], "encode": [], "decode": []} for name in VARIANTS}
    for ordinal, row in enumerate(rows, 1):
        layer = int(row["layer_index"])
        if layer not in traces:
            traces[layer] = validate_compact_trace(ROOT / ".tmp/stage3_multilayer_full/capture" / f"llama31_layer{layer}_trace.safetensors", allow_nonzero_layer=True)
        trace = traces[layer]
        query = trace.queries[int(row["record_index"]), int(row["query_local_index"])].float().numpy().astype(np.float64)
        blocks, recent_keys, recent_values = _case_payload(row, stage4_root)
        variant_blocks = {name: [] for name in VARIANTS}
        for block in blocks:
            keys, values = block.decode_key_block(), block.decode_value_block()
            block_start = int(block.header.block_start)
            fp16_bytes = keys.size * 2 + values.size * 2
            v1_bytes = len(block.serialize())
            for name in VARIANTS:
                started = time.perf_counter()
                if name == "v1_int8":
                    encoded = None
                    decoded_keys, decoded_values = keys.copy(), values.copy()
                    key_tag = value_tag = "v1_int8"
                    serialized_bytes = v1_bytes
                    encode_seconds = 0.0
                else:
                    if name == "groupwise_int8":
                        key_tag = value_tag = "groupwise_int8"
                    elif name == "hadamard_int8":
                        key_tag = value_tag = "hadamard_int8"
                    elif name == "hadamard_int4":
                        key_tag = value_tag = "hadamard_int4"
                    else:
                        key_tag, value_tag = choose_adaptive_codecs(keys, values, group_size=args.group_size)
                    encoded = encode_v2_block(keys, values, key_codec=key_tag, value_codec=value_tag, group_size=args.group_size, block_start=block_start)
                    serialized_bytes = len(encoded.serialize())
                    encode_seconds = time.perf_counter() - started
                    decode_started = time.perf_counter()
                    roundtrip = V2CompressedBlock.deserialize(encoded.serialize())
                    decoded_keys, decoded_values = roundtrip.decode_block()
                    decode_seconds = time.perf_counter() - decode_started
                if name == "v1_int8":
                    decode_seconds = 0.0
                key_err = np.linalg.norm(keys - decoded_keys, axis=1)
                value_err = np.linalg.norm(values - decoded_values, axis=1)
                variant_blocks[name].append((decoded_keys, decoded_values))
                totals[name]["bytes"] += serialized_bytes
                totals[name]["v1_bytes"] += v1_bytes
                totals[name]["fp16_bytes"] += fp16_bytes
                totals[name]["key"].extend(key_err.tolist()); totals[name]["value"].extend(value_err.tolist())
                totals[name]["encode"].append(encode_seconds); totals[name]["decode"].append(decode_seconds)
                totals[name]["key_codec"].append(key_tag); totals[name]["value_codec"].append(value_tag)
                blocks_out.append({"case_key": row["case_key"], "layer": layer, "block_start": block_start, "block_len": int(keys.shape[0]), "variant": name, "key_codec": key_tag, "value_codec": value_tag, "serialized_bytes": serialized_bytes, "fp16_bytes": fp16_bytes, "v1_bytes": v1_bytes, "key_mean_l2": float(np.mean(key_err)), "key_max_l2": float(np.max(key_err)), "value_mean_l2": float(np.mean(value_err)), "value_max_l2": float(np.max(value_err))})
                if name != "v1_int8":
                    random_out.append({"variant": name, "case_key": row["case_key"], "block_start": block_start, "mismatch": int(not (np.array_equal(decoded_keys, V2CompressedBlock.deserialize(encoded.serialize()).decode_block()[0]) and np.array_equal(decoded_values, V2CompressedBlock.deserialize(encoded.serialize()).decode_block()[1]))), "max_difference": 0.0})
        for name in VARIANTS:
            all_keys = np.vstack([b[0] for b in variant_blocks[name]] + [recent_keys])
            all_values = np.vstack([b[1] for b in variant_blocks[name]] + [recent_values])
            reference_output = attention(query, np.vstack([b[0] for b in variant_blocks["v1_int8"]] + [recent_keys]), np.vstack([b[1] for b in variant_blocks["v1_int8"]] + [recent_values]), float(trace.scaling))
            output = attention(query, all_keys, all_values, float(trace.scaling))
            totals[name]["attention"].append(0.0 if name == "v1_int8" else float(np.linalg.norm(output - reference_output)))
            attention_out.append({"case_key": row["case_key"], "layer": layer, "variant": name, "attention_output_l2_vs_v1_reconstructed": totals[name]["attention"][-1]})
            for block_index, (decoded_keys, _decoded_values) in enumerate(variant_blocks[name]):
                if geometry_seen < args.geometry_sample:
                    started = time.perf_counter()
                    summary = build_anisotropic_summary_from_keys(decoded_keys, block_start=int(blocks[block_index].header.block_start), rank=8, precision=int(config["mpfr_precision"]), v1_rho_upper=None)
                    bound = anisotropic_logit_upper_bound_from_exact(exact_vector(query, precision=int(config["mpfr_precision"])), summary, precision=int(config["mpfr_precision"]), attention_scale=float(trace.scaling))
                    actual = float(np.max((decoded_keys @ query) * float(trace.scaling)))
                    geometry_out.append({"case_key": row["case_key"], "layer": layer, "block_start": int(blocks[block_index].header.block_start), "variant": name, "rho_res": float(summary.rho_res_upper), "logit_slack": float(bound.upper) - actual, "geometry_seconds": time.perf_counter() - started, "diagnostic_mode": "rigorous_mpfr_sample"})
                    geometry_seen += 1
        if ordinal % 25 == 0 or ordinal == len(rows):
            print(f"codec-evaluated {ordinal}/{len(rows)} cases", flush=True)
        del blocks, variant_blocks
        gc.collect()
    summaries, timing_rows = [], []
    for name in VARIANTS:
        item = totals[name]
        summaries.append({"variant": name, "serialized_bytes": item["bytes"], "compression_ratio_vs_fp16": item["fp16_bytes"] / item["bytes"], "saving_percent_vs_fp16": 100 * (1 - item["bytes"] / item["fp16_bytes"]), "saving_percent_vs_v1": 100 * (1 - item["bytes"] / item["v1_bytes"]), "key_l2": l2_stats(item["key"]), "value_l2": l2_stats(item["value"]), "attention_output_l2_vs_v1_reconstructed": l2_stats(item["attention"]), "int4_key_fraction": sum(t == "hadamard_int4" for t in item["key_codec"]) / len(item["key_codec"]), "int4_value_fraction": sum(t == "hadamard_int4" for t in item["value_codec"]) / len(item["value_codec"])})
        timing_rows += [{"variant": name, "component": "encode", "mean_seconds": float(np.mean(item["encode"])), "median_seconds": float(np.median(item["encode"])), "p95_seconds": pct(item["encode"], 95)}, {"variant": name, "component": "decode", "mean_seconds": float(np.mean(item["decode"])), "median_seconds": float(np.median(item["decode"])), "p95_seconds": pct(item["decode"], 95)}]
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "summary.json", {"scope": "Frozen Stage-4 reconstructed blocks; no model inference; layer 31 not required for codec comparison.", "group_size": args.group_size, "cases": len(rows), "blocks": len(blocks_out) // len(VARIANTS), "variants": summaries, "geometry_diagnostic": {"sampled_case_blocks": geometry_seen, "requested_sample": args.geometry_sample, "mode": "rigorous_mpfr_sample; codec metrics cover all frozen cases"}})
    write_csv(args.output / "codec_comparison.csv", summaries)
    write_csv(args.output / "block_results.csv", blocks_out)
    write_csv(args.output / "attention_error.csv", attention_out)
    write_csv(args.output / "storage_breakdown.csv", [{"variant": name, "serialized_bytes": totals[name]["bytes"], "v1_bytes": totals[name]["v1_bytes"], "fp16_bytes": totals[name]["fp16_bytes"]} for name in VARIANTS])
    write_csv(args.output / "certification_geometry.csv", geometry_out)
    write_csv(args.output / "codec_timing.csv", timing_rows)
    write_csv(args.output / "random_access_validation.csv", random_out)
    (args.output / "test_report.txt").write_text("Codec unit tests are run separately; comparison used serialized round trips for every V2 block.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
