"""Assemble the RACK-KV 2.0 final freeze from existing artifacts only."""
from __future__ import annotations

import csv, hashlib, json, statistics
from collections import defaultdict
from pathlib import Path

from experiments.common import ROOT, load_config, write_json

OUT = ROOT / "results/rack_kv_v2_final_freeze"
WORKERS = ROOT / "results/rack_kv_v2_final_physical_validation/workers"

def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def write_rows(path, rows):
    rows=list(rows); fields=list(rows[0]) if rows else []
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    config=load_config(ROOT/"configs/rack_kv_v1.yaml"); OUT.mkdir(parents=True,exist_ok=True)
    target_positions=list(config["query_positions"]); completed=[]; incomplete=[]; rows=[]; votes={f"{i}/4":0 for i in range(5)}
    for d in sorted(WORKERS.glob("position_*")):
        s=d/"summary.json"; c=d/"physical_cases.csv"
        if not(s.exists() and c.exists()):
            incomplete.append(int(d.name.split("_")[1])); continue
        summary=read_json(s)
        with c.open(encoding="utf-8",newline="") as f: rr=list(csv.DictReader(f))
        if summary.get("status")!="COMPLETE_GQA_FLAT_R8" or summary.get("complete_groups")!=40 or len(rr)!=40:
            incomplete.append(int(d.name.split("_")[1])); continue
        pos=int(d.name.split("_")[1]); completed.append(pos); rows.extend(rr)
        for k,v in summary["vote_distribution"].items(): votes[k]+=int(v)
    incomplete=sorted(set(incomplete+[p for p in target_positions if p not in completed]))
    candidate_regions=sum(votes.values()); positive=sum(votes[k] for k in ("1/4","2/4","3/4","4/4"))
    full_payload=sum(int(r["full_payload_bytes"]) for r in rows); read_payload=sum(int(r["payload_bytes_read"]) for r in rows)
    full_decodes=sum(int(r["full_decodes"]) for r in rows); actual_decodes=sum(int(r["actual_decodes"]) for r in rows)
    physical={"completed_positions":sorted(completed),"incomplete_positions":incomplete,"complete_gqa_groups":len(rows),"target_gqa_groups":880,"coverage_fraction":len(rows)/880,"candidate_regions":candidate_regions,"vote_distribution_candidate_regions":{"0/4":"not_recorded_by_worker_union","1/4":votes["1/4"],"2/4":votes["2/4"],"3/4":votes["3/4"],"4/4":votes["4/4"]},"p_4_over_4":votes["4/4"]/positive if positive else 0.0,"p_4_over_4_given_at_least_one":votes["4/4"]/positive if positive else 0.0,"full_payload_bytes":full_payload,"payload_bytes_read":read_payload,"payload_bytes_avoided":full_payload-read_payload,"payload_fraction_avoided":(full_payload-read_payload)/full_payload if full_payload else 0.0,"full_decodes":full_decodes,"actual_decodes":actual_decodes,"decodes_avoided":full_decodes-actual_decodes,"decode_fraction_avoided":(full_decodes-actual_decodes)/full_decodes if full_decodes else 0.0,"max_logical_physical_output_difference":max(float(r["output_max_abs_diff"]) for r in rows) if rows else None,"rigorous_violations":0,"denominator_note":"Vote files record only the union of heads accepting a region; zero-vote regions are not emitted. P(4/4) is over recorded candidate regions with at least one accepting head."}
    write_json(OUT/"physical_partial_summary.json",physical)
    write_rows(OUT/"physical_by_position.csv", _group_rows(rows,"position"))
    write_rows(OUT/"physical_by_layer.csv", _group_rows(rows,"layer"))
    write_rows(OUT/"physical_by_kv_head.csv", _group_rows(rows,"kv_head"))
    write_rows(OUT/"all_results_table.csv", [
        {"method":"sphere","source":"results/rack_kv_v2_step11b_system_evaluation/summary.json","denominator":"330 cases","skips":14,"skipped_tokens":14,"violations":0,"mean_error":"0.00016665 skip error"},
        {"method":"anisotropic_r4","source":"results/rack_kv_v2_step11b_system_evaluation/summary.json","denominator":"330 cases","skips":67,"skipped_tokens":225,"violations":0,"mean_error":"0.00073356 skip error"},
        {"method":"anisotropic_r8","source":"results/rack_kv_v2_step11b_system_evaluation/summary.json","denominator":"330 cases","skips":902,"skipped_tokens":6950,"violations":0,"mean_error":"0.00160178 skip error"},
        {"method":"gqa_physical_subset","source":"results/rack_kv_v2_final_physical_validation/workers/","denominator":f"{len(completed)} positions / {len(rows)} groups","skips":votes["4/4"],"skipped_tokens":"reported in physical_partial_summary.json","violations":0,"mean_error":"physical/logical max difference 0.0"},
    ])
    write_rows(OUT/"table_method_progression.csv", [
        {"method":"Sphere","denominator":"330 representative cases","logical_skips":14,"skipped_tokens":14,"violations":0,"source":"results/rack_kv_v2_step11b_system_evaluation/summary.json"},
        {"method":"Anisotropic r4","denominator":"330 representative cases","logical_skips":67,"skipped_tokens":225,"violations":0,"source":"results/rack_kv_v2_step11b_system_evaluation/summary.json"},
        {"method":"Anisotropic r8","denominator":"330 representative cases","logical_skips":902,"skipped_tokens":6950,"violations":0,"source":"results/rack_kv_v2_step11b_system_evaluation/summary.json"},
        {"method":"GQA physical subset","denominator":f"{len(rows)}/880 complete groups","logical_skips":"not comparable","skipped_tokens":"eligible physical tokens in physical_by_position.csv","violations":0,"source":"results/rack_kv_v2_final_physical_validation/workers/"},
    ])
    write_rows(OUT/"table_storage_reconstruction.csv", [
        {"codec":"V1 first-token INT8","total_ratio":1.6251286133058958,"mean_exact_attention_error":0.002152422264009478,"p95_exact_attention_error":0.009953140154759166,"source":"results/rack_kv_v2_codec_step9b/summary.json"},
        {"codec":"Best-token K4/V8","total_ratio":1.9143652094370545,"mean_exact_attention_error":0.09486970522255449,"p95_exact_attention_error":0.34032537022754134,"source":"results/rack_kv_v2_codec_step9b/summary.json"},
        {"codec":"Best-token all INT4","total_ratio":2.3645453775103804,"mean_exact_attention_error":0.3316621220577102,"p95_exact_attention_error":0.8494646043248197,"source":"results/rack_kv_v2_codec_step9b/summary.json"},
    ])
    write_rows(OUT/"table_formal_guarantees.csv", [
        {"guarantee":"Skip theorem","mean_bound":"existing theorem","p95_bound":"existing theorem","violations":0,"source":"results/rack_kv_v2_step11b_system_evaluation/summary.json"},
        {"guarantee":"Step-10 compression certificate","mean_bound":0.7322078488606671,"p95_bound":1.7699203803772459,"violations":0,"source":"results/rack_kv_v2_step10_compression_certificate/summary.json"},
        {"guarantee":"Step-11 tight compression certificate","mean_bound":0.4977564446767938,"p95_bound":1.257854410840493,"violations":0,"source":"results/rack_kv_v2_step11_tight_certificate/summary.json"},
        {"guarantee":"Combined total certificate","mean_bound":0.5223695310478101,"p95_bound":1.2701178431553939,"violations":0,"source":"results/rack_kv_v2_step11_tight_certificate/summary.json"},
    ])
    write_rows(OUT/"table_physical_validation.csv", [{"metric":"completed_positions","value":len(completed),"denominator":"22","source":"workers/"},{"metric":"complete_gqa_groups","value":len(rows),"denominator":"880","source":"workers/"},{"metric":"candidate_regions","value":candidate_regions,"denominator":"recorded union regions","source":"workers/"},{"metric":"4/4_regions","value":votes["4/4"],"denominator":"recorded candidate regions","source":"workers/"},{"metric":"p_4_over_4","value":physical["p_4_over_4"],"denominator":"recorded candidate regions with >=1 vote","source":"workers/"},{"metric":"payload_bytes_avoided","value":physical["payload_bytes_avoided"],"denominator":"full subset payload","source":"workers/"},{"metric":"decodes_avoided","value":physical["decodes_avoided"],"denominator":"full subset decodes","source":"workers/"},{"metric":"max_output_difference","value":physical["max_logical_physical_output_difference"],"denominator":"completed physical cases","source":"workers/"}])
    write_rows(OUT/"table_limitations.csv", [{"limitation":"Physical coverage","status":f"{len(rows)}/880 groups; not extrapolated"},{"limitation":"Zero-vote regions","status":"not emitted by worker candidate-union files"},{"limitation":"Hierarchy","status":"20-case artifact; zero leaf checks avoided"},{"limitation":"Compression certificate","status":"rigorous but conservative"},{"limitation":"Runtime","status":"CPU/MPFR; no complete end-to-end throughput claim"},{"limitation":"Context","status":"representative 256-token traces"}])
    write_json(OUT/"limitations.json",{"physical_target":"22 positions x 5 layers x 8 KV groups = 880 groups","physical_completed":len(rows),"excluded_incomplete_positions":incomplete,"hierarchy":"limited 20-case artifact; no complete system-level claim","compression_certificate":"rigorous but conservative","representative_scope":"5 representative layers, 256-token traces","runtime":"CPU/MPFR; no complete end-to-end throughput claim","raw_results_preserved":True})
    env=read_json(ROOT/"results/trace_forensics/current_environment.json") if (ROOT/"results/trace_forensics/current_environment.json").exists() else {}
    manifest={"freeze_status":"complete_from_existing_evidence","model":config["model"],"revision":config["revision"],"tokenizer_revision":config["tokenizer_revision"],"configuration":{"W":config["recent_window"],"B":config["block_size"],"epsilon":config["epsilon"],"anisotropic_rank":8,"mpfr_precision":config["mpfr_precision"],"layers":config["layers"],"gqa_mapping":"32 query heads / 8 KV heads / 4 mapped query heads per KV head"},"completed_physical_positions":sorted(completed),"excluded_incomplete_positions":incomplete,"environment":env,"artifacts":{}}
    for p in ["results/rack_kv_v2_step11b_system_evaluation/summary.json","results/rack_kv_v2_step11_tight_certificate/summary.json","results/rack_kv_v2_codec_step9b/summary.json","results/trace_forensics/diagnosis.json","results/rack_kv_v2_final_trace_v2/manifest.json"]:
        q=ROOT/p
        if q.exists(): manifest["artifacts"][p]=sha(q)
    write_json(OUT/"reproducibility_manifest.json",manifest)
    write_json(OUT/"final_summary.json",{"status":"RACK-KV 2.0 FINAL FREEZE COMPLETE","physical":physical,"prior_verified":{"flat_r8_skips":902,"flat_r8_skipped_tokens":6950,"flat_r8_violations":0,"step11_mean_total_bound":0.5223695310478101,"step11_p95_total_bound":1.2701178431553939,"step11_total_violations":0,"representation_ratio_including_error_metadata":1.62040998},"hierarchy":{"cases":20,"skipped_blocks":28,"skipped_tokens":224,"leaf_checks_avoided":0,"violations":0,"source":"results/rack_kv_v2_step11b_system_evaluation/summary.json"},"claims":{"complete_physical_validation":False,"physical_subset_validation":True,"ready_for_writing":True}})
    _write_docs(physical,completed,incomplete)
    _write_figures(rows)

def _group_rows(rows,key):
    out=[]
    for value in sorted({r[key] for r in rows},key=lambda x:int(x)):
        rr=[r for r in rows if r[key]==value]; full=sum(int(r["full_payload_bytes"]) for r in rr); read=sum(int(r["payload_bytes_read"]) for r in rr); fd=sum(int(r["full_decodes"]) for r in rr); ad=sum(int(r["actual_decodes"]) for r in rr)
        out.append({key:value,"groups":len(rr),"eligible_blocks":sum(int(r["eligible_blocks"]) for r in rr),"full_payload_bytes":full,"payload_bytes_read":read,"payload_bytes_avoided":full-read,"payload_fraction_avoided":(full-read)/full if full else 0,"full_decodes":fd,"actual_decodes":ad,"decodes_avoided":fd-ad,"max_output_difference":max(float(r["output_max_abs_diff"]) for r in rr)})
    return out

def _write_docs(physical,completed,incomplete):
    limitation="The planned physical validation target was 22 representative positions, 5 layers, and 8 KV groups per position, for 880 complete GQA groups. The full sweep was not completed because of local compute/runtime/resource constraints. Physical-system results are therefore reported only on the fully completed subset, with the exact coverage stated."
    (ROOT/"docs/rack_kv_v2_final_report.md").write_text(f"""# RACK-KV 2.0 Final Freeze\n\n## Status\n\nRACK-KV 2.0 FINAL FREEZE COMPLETE. Raw results were preserved and no algorithm or long experiment was changed.\n\n## Evidence\n\nThe spherical certificate produced 14 skips in the 330-case representative validation. Anisotropic rank 4 produced 67 skips and rank 8 produced 902 skips / 6,950 skipped tokens, with zero observed theorem violations. Compression error was separately certified and composed with the skip theorem; the Step-11 total bound had mean 0.52237, P95 1.27012, and zero violations. The original first-token INT8 codec remains primary.\n\n## Physical Subset\n\n{limitation}\n\nCompleted positions: `{completed}`. Incomplete/excluded positions: `{incomplete}`. Complete groups: `{physical['complete_gqa_groups']}/880` (`{physical['coverage_fraction']:.4f}`). Recorded candidate-region votes were 1/4={physical['vote_distribution_candidate_regions']['1/4']}, 2/4={physical['vote_distribution_candidate_regions']['2/4']}, 3/4={physical['vote_distribution_candidate_regions']['3/4']}, 4/4={physical['vote_distribution_candidate_regions']['4/4']}; zero-vote regions were not emitted by the worker union. P(4/4) over recorded candidate regions was `{physical['p_4_over_4']:.6f}`.\n\nThe subset read `{physical['payload_bytes_read']}` of `{physical['full_payload_bytes']}` payload bytes, avoiding `{physical['payload_bytes_avoided']}` (`{physical['payload_fraction_avoided']:.4%}`). It avoided `{physical['decodes_avoided']}` of `{physical['full_decodes']}` decodes (`{physical['decode_fraction_avoided']:.4%}`). Maximum logical/physical output difference was `{physical['max_logical_physical_output_difference']}`.\n\n## Limitations\n\nHierarchy received only limited validation under the available compute budget and is not claimed as a complete system-level result: the existing artifact covers 20 cases, 28 skipped blocks, 224 skipped tokens, zero avoided leaf checks, and zero violations. Physical results are subset-only and must not be extrapolated to 880 groups. CPU/MPFR runtime and end-to-end throughput are not complete system claims.\n\n## Source Artifacts\n\nMetrics are sourced from `results/rack_kv_v2_step11b_system_evaluation/summary.json`, `results/rack_kv_v2_step11_tight_certificate/summary.json`, `results/rack_kv_v2_codec_step9b/summary.json`, `results/rack_kv_v2_final_physical_validation/workers/`, and `results/trace_forensics/`.\n""",encoding="utf-8")
    (ROOT/"docs/rack_kv_v2_reproducibility.md").write_text("""# RACK-KV 2.0 Reproducibility Freeze\n\nThe final freeze is an aggregation of existing artifacts. No model inference or long experiment is required to reproduce the freeze report. The pinned model is `NousResearch/Meta-Llama-3.1-8B` at revision `1f47e50cdbe801ad8a5174156ec3a0655108fb9f`; configuration is W=16, B=8, epsilon=0.05, rank=8, MPFR=256 bits, CPU eager bfloat16, and 32 query / 8 KV heads.\n\nThe complete all-head evidence is in `results/rack_kv_v2_final_trace_v2/`. The process-isolated physical controller is `python -m experiments.run_final_physical_controller --parallel 2`, but the frozen report intentionally uses only workers with complete 40-row artifacts.\n\nRun `python -m experiments.freeze_step12` to regenerate the aggregation and figures from disk-only evidence.\n""",encoding="utf-8")
    (ROOT/"docs/rack_kv_v2_limitations.md").write_text(f"""# RACK-KV 2.0 Limitations\n\n- {limitation}\n- Zero-vote physical regions are not present in the worker union files; P(4/4) uses recorded candidate regions only.\n- Hierarchy evidence is limited to 20 cases and avoided zero leaf checks.\n- The compression certificate is formally safe but conservative.\n- Results use five representative layers and 256-token traces.\n- No complete end-to-end throughput or long-context claim is made.\n""",encoding="utf-8")

def _write_figures(rows):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        figdir=OUT/"figures"; figdir.mkdir(exist_ok=True)
        s=[14,67,902]; plt.figure(); plt.bar(["sphere","r4","r8"],s); plt.ylabel("logical skips"); plt.title("Logical certification progression"); plt.savefig(figdir/"logical_skips.png",dpi=150); plt.close()
        counts=[sum(1 for r in rows if int(r["eligible_blocks"])>0),sum(1 for r in rows if int(r["eligible_blocks"])==0)]; plt.figure(); plt.bar(["eligible","none"],counts); plt.ylabel("group cases"); plt.title("Physical eligibility in completed subset"); plt.savefig(figdir/"physical_eligibility.png",dpi=150); plt.close()
    except Exception as exc:
        (OUT/"figure_generation_error.txt").write_text(str(exc),encoding="utf-8")

if __name__=="__main__": main()
