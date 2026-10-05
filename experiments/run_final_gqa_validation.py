"""Complete-GQA validation on the internally consistent all-head trace."""

from __future__ import annotations

import csv
import gc
import json
import time
from pathlib import Path

import numpy as np

from experiments.common import ROOT, load_config, write_json
# The environment contains the installed hub package under a renamed
# directory; reuse the local-only capture shim before importing transformers-
# dependent RACK-KV modules.
from experiments import capture_local_layer31 as _local_capture  # noqa: F401
from rack_kv.codec import encode_block
from rack_kv.gqa import certify_gqa_hierarchical_group, gqa_groups
from rack_kv.hierarchy import build_hierarchy
from rack_kv.physical import KVPayloadStore, execute_logical_gqa, execute_physical_gqa
from rack_kv.stage2 import validate_compact_trace


TRACE_ROOT = ROOT / ".tmp" / "rack_kv_v2_final_trace_v2"
OUT = ROOT / "results" / "rack_kv_v2_final_gqa_validation"


def main() -> None:
    config = load_config(ROOT / "configs" / "rack_kv_v1.yaml")
    OUT.mkdir(parents=True, exist_ok=True)
    groups = gqa_groups(num_attention_heads=int(config["query_heads"]), num_key_value_heads=int(config["kv_heads"]))
    vote_counts = {f"{i}/{len(groups[0])}": 0 for i in range(len(groups[0]) + 1)}
    rows = []
    io_rows = []
    total_groups = 0
    total_eligible_leaves = 0
    total_eligible_tokens = 0
    total_nodes = total_leaf_checks = total_mpfr = 0
    max_output_diff = 0.0
    layers = tuple(config["layers"])
    positions = tuple(config["query_positions"])
    started = time.perf_counter()
    for layer in layers:
        trace = validate_compact_trace(TRACE_ROOT / f"llama31_layer{layer}_trace.safetensors", allow_nonzero_layer=True)
        for position in positions:
            record = int(position)
            visible = int(trace.visible_lengths[record])
            history_end = max(0, visible - int(config["recent_window"]))
            for kv_head, mapped_heads in enumerate(groups):
                key = trace.final_keys[kv_head, :visible].float().numpy().astype(np.float64)
                value = trace.final_values[kv_head, :visible].float().numpy().astype(np.float64)
                recent_keys = key[history_end:]
                recent_values = value[history_end:]
                historical = key[:history_end]
                historical_values = value[:history_end]
                blocks = [encode_block(historical[start:start + int(config["block_size"])], historical_values[start:start + int(config["block_size"])], block_start=start, precision=int(config["mpfr_precision"])) for start in range(0, history_end, int(config["block_size"]))]
                if not blocks:
                    continue
                index = build_hierarchy(blocks, fanout=4, rank=8, precision=int(config["mpfr_precision"]))
                leaf_payloads = [block.decode_block() for block in blocks]
                leaf_keys = [payload[0] for payload in leaf_payloads]
                leaf_values = [payload[1] for payload in leaf_payloads]
                queries = {head: trace.queries[record, trace.selected_query_heads.index(head)].float().numpy().astype(np.float64) for head in mapped_heads}
                recent_by_head = {head: recent_keys for head in mapped_heads}
                recent_values_by_head = {head: recent_values for head in mapped_heads}
                t0 = time.perf_counter()
                result = certify_gqa_hierarchical_group(
                    kv_head=kv_head, query_by_head=queries, recent_keys_by_head=recent_by_head,
                    recent_values_by_head=recent_values_by_head, index=index, leaf_keys=leaf_keys,
                    leaf_values=leaf_values, epsilon_head=float(config["epsilon"]),
                    precision=int(config["mpfr_precision"]), attention_scale=float(trace.scaling),
                    num_attention_heads=int(config["query_heads"]), num_key_value_heads=int(config["kv_heads"]),
                    head_dim=int(config["head_dim"]),
                )
                cert_time = time.perf_counter() - t0
                total_groups += 1
                for decision in result.region_decisions:
                    vote_counts[f"{decision.gqa_skip_vote_count}/{len(mapped_heads)}"] += 1
                eligible = tuple(result.physical_eligible_leaf_blocks)
                total_eligible_leaves += len(eligible)
                total_eligible_tokens += sum(blocks[index.nodes[node_id].leaf_index].block_len for node_id in eligible)
                total_nodes += result.nodes_considered
                total_leaf_checks += sum(1 for d in result.region_decisions if d.tree_level == 0)
                total_mpfr += len(result.region_decisions) * len(mapped_heads)
                payload_path = OUT / "payload" / f"layer{layer}_pos{position}_kv{kv_head}.rackv2p"
                store = KVPayloadStore.create(payload_path, blocks)
                logical = execute_logical_gqa(store=store, required_leaf_blocks=result.required_leaf_blocks, queries_by_head=queries, recent_keys_by_head=recent_by_head, recent_values_by_head=recent_values_by_head, key_dim=int(config["head_dim"]), value_dim=int(config["head_dim"]), attention_scale=float(trace.scaling))
                store = KVPayloadStore.open(payload_path)
                physical = execute_physical_gqa(store=store, queries_by_head=queries, recent_keys_by_head=recent_by_head, recent_values_by_head=recent_values_by_head, required_leaf_blocks=result.required_leaf_blocks, physically_eligible_blocks=eligible, complete_gqa_group=True, all_heads_represented=True, mpfr_authorized=not result.numerical_fallback_used, numerical_fallback=result.numerical_fallback_used, key_dim=int(config["head_dim"]), value_dim=int(config["head_dim"]), attention_scale=float(trace.scaling))
                diff = max(float(np.max(np.abs(logical[head] - physical.outputs_by_head[head]))) for head in mapped_heads)
                max_output_diff = max(max_output_diff, diff)
                rows.append({"layer": layer, "position": position, "kv_head": kv_head, "mapped_heads": ",".join(map(str, mapped_heads)), "candidate_regions": len(result.region_decisions), "physical_eligible_leaf_blocks": len(eligible), "physical_eligible_tokens": sum(blocks[index.nodes[node_id].leaf_index].block_len for node_id in eligible), "nodes_considered": result.nodes_considered, "leaf_checks": sum(1 for d in result.region_decisions if d.tree_level == 0), "certificate_seconds": cert_time, "payload_full_bytes": store.full_payload_bytes, "payload_read_bytes": store.stats.payload_bytes_read, "payload_avoided_bytes": store.full_payload_bytes - store.stats.payload_bytes_read, "full_decodes": store.full_block_count, "actual_decodes": store.stats.blocks_decoded, "decode_avoided": store.full_block_count - store.stats.blocks_decoded, "output_max_abs_diff": diff, "fallback": result.numerical_fallback_used})
                gc.collect()
            if len(rows) % 40 == 0:
                print(f"validated {len(rows)} physical GQA cases", flush=True)
    fields = list(rows[0]) if rows else []
    with (OUT / "gqa_physical_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with (OUT / "gqa_vote_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["vote_count", "count", "fraction"]); writer.writeheader()
        denominator = sum(vote_counts.values())
        for key, count in vote_counts.items(): writer.writerow({"vote_count": key, "count": count, "fraction": count / denominator if denominator else 0.0})
    summary = {
        "status": "COMPLETE_GQA_FINAL_TRACE",
        "trace_root": str(TRACE_ROOT), "layers": list(layers), "positions": list(positions),
        "complete_gqa_groups": total_groups, "groups": [list(group) for group in groups],
        "vote_distribution": vote_counts, "p_4_over_4": vote_counts.get("4/4", 0) / sum(vote_counts.values()) if vote_counts else 0.0,
        "physical_eligible_leaf_blocks": total_eligible_leaves, "physical_eligible_tokens": total_eligible_tokens,
        "nodes_considered": total_nodes, "leaf_checks": total_leaf_checks, "mpfr_calls_estimate": total_mpfr,
        "max_logical_physical_output_difference": max_output_diff,
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_note": "This is complete-GQA validation over re-encoded V1 blocks from the internally consistent final trace; no historical selected-head data were mixed into the decision set.",
    }
    write_json(OUT / "summary.json", summary)
    (OUT / "test_report.txt").write_text(f"complete GQA groups: {total_groups}\nmax logical/physical output difference: {max_output_diff}\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
