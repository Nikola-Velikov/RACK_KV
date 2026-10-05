"""Bounded-memory flat rank-8 GQA unanimity and physical-read replay."""
from __future__ import annotations

import argparse, csv, gc, json, time
from pathlib import Path
import numpy as np

from experiments.common import ROOT, load_config, write_json
from experiments import capture_local_layer31 as _shim  # noqa: F401
from rack_kv.codec import encode_block
from rack_kv.anisotropic import execute_anisotropic_flat, reconstructed_attention_output
from rack_kv.gqa import gqa_groups
from rack_kv.physical import KVPayloadStore, execute_physical_gqa
from rack_kv.stage2 import validate_compact_trace

TRACE_ROOT = ROOT / ".tmp/rack_kv_v2_final_trace_v2"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--start",type=int,default=0); p.add_argument("--limit",type=int,default=None); p.add_argument("--output",type=Path,default=ROOT/"results/rack_kv_v2_final_flat_gqa")
    a=p.parse_args(); c=load_config(ROOT/"configs/rack_kv_v1.yaml"); groups=gqa_groups(num_attention_heads=c["query_heads"],num_key_value_heads=c["kv_heads"])
    positions=list(c["query_positions"]); positions=positions[a.start:] if a.limit is None else positions[a.start:a.start+a.limit]
    rows=[]; votes={f"{i}/4":0 for i in range(5)}; maxdiff=0.; total=0; started=time.perf_counter()
    for layer in c["layers"]:
        tr=validate_compact_trace(TRACE_ROOT/f"llama31_layer{layer}_trace.safetensors",allow_nonzero_layer=True)
        for pos in positions:
            visible=int(tr.visible_lengths[pos]); hist_end=max(0,visible-c["recent_window"])
            for kv, heads in enumerate(groups):
                k=tr.final_keys[kv,:visible].float().numpy().astype(np.float64); v=tr.final_values[kv,:visible].float().numpy().astype(np.float64)
                hk,hv=k[:hist_end],v[:hist_end]; recent_k,recent_v=k[hist_end:],v[hist_end:]
                blocks=[encode_block(hk[s:s+c["block_size"]],hv[s:s+c["block_size"]],block_start=s,precision=c["mpfr_precision"]) for s in range(0,hist_end,c["block_size"])]
                if not blocks: continue
                outcomes={}
                for h in heads:
                    q=tr.queries[pos,tr.selected_query_heads.index(h)].float().numpy().astype(np.float64)
                    outcomes[h]=execute_anisotropic_flat(query=q,recent_keys=recent_k,recent_values=recent_v,historical_blocks=blocks,rank=8,tolerance=c["epsilon"],precision=c["mpfr_precision"],attention_scale=float(tr.scaling))
                starts=sorted({b.header.block_start for o in outcomes.values() for b in blocks if b.header.block_start in o.skipped_block_starts})
                eligible=[s for s in starts if all(s in outcomes[h].skipped_block_starts for h in heads)]
                for s in starts: votes[f"{sum(s in outcomes[h].skipped_block_starts for h in heads)}/4"]+=1
                block_by_start={b.header.block_start:i for i,b in enumerate(blocks)}; eligible_ids=tuple(block_by_start[s] for s in eligible); required=tuple(i for i in range(len(blocks)) if i not in eligible_ids)
                path=a.output/"payload"/f"layer{layer}_pos{pos}_kv{kv}.rackv2p"; path.parent.mkdir(parents=True,exist_ok=True); store=KVPayloadStore.create(path,blocks); store=KVPayloadStore.open(path)
                qmap={h:tr.queries[pos,tr.selected_query_heads.index(h)].float().numpy().astype(np.float64) for h in heads}; rk={h:recent_k for h in heads}; rv={h:recent_v for h in heads}
                phys=execute_physical_gqa(store=store,queries_by_head=qmap,recent_keys_by_head=rk,recent_values_by_head=rv,required_leaf_blocks=required,physically_eligible_blocks=eligible_ids,complete_gqa_group=True,all_heads_represented=True,mpfr_authorized=True,key_dim=c["head_dim"],value_dim=c["head_dim"],attention_scale=float(tr.scaling))
                diff=0.
                loaded_keys=np.vstack([recent_k]+[blocks[i].decode_key_block() for i in required]); loaded_values=np.vstack([recent_v]+[blocks[i].decode_value_block() for i in required])
                for h in heads: diff=max(diff,float(np.linalg.norm(reconstructed_attention_output(qmap[h],loaded_keys,loaded_values,attention_scale=float(tr.scaling))-phys.outputs_by_head[h])))
                maxdiff=max(maxdiff,diff); total+=1
                rows.append({"layer":layer,"position":pos,"kv_head":kv,"candidate_blocks":len(blocks),"vote_candidates":len(starts),"eligible_blocks":len(eligible_ids),"eligible_tokens":sum(blocks[i].block_len for i in eligible_ids),"full_payload_bytes":store.full_payload_bytes,"payload_bytes_read":store.stats.payload_bytes_read,"payload_bytes_avoided":store.full_payload_bytes-store.stats.payload_bytes_read,"full_decodes":store.full_block_count,"actual_decodes":store.stats.blocks_decoded,"decodes_avoided":store.full_block_count-store.stats.blocks_decoded,"output_max_abs_diff":diff})
                del outcomes,blocks,store; gc.collect()
            if total and total%40==0: print(f"validated {total} flat GQA cases",flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/"physical_cases.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (a.output/"vote_distribution.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["vote_count","count","fraction"]);w.writeheader();den=sum(votes.values());[w.writerow({"vote_count":k,"count":v,"fraction":v/den if den else 0}) for k,v in votes.items()]
    summary={"status":"COMPLETE_GQA_FLAT_R8","layers":list(c["layers"]),"positions":positions,"complete_groups":total,"vote_distribution":votes,"p_4_over_4":votes["4/4"]/sum(votes.values()) if sum(votes.values()) else 0.,"max_output_difference":maxdiff,"elapsed_seconds":time.perf_counter()-started,"note":"Flat rank-8 unanimity replay on the internally consistent all-head trace; hierarchy remains a separate validation."}
    write_json(a.output/"summary.json",summary); (a.output/"test_report.txt").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
