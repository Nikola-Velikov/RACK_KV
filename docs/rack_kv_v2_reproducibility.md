# RACK-KV 2.0 Reproducibility Freeze

The final freeze is an aggregation of existing artifacts. No model inference or long experiment is required to reproduce the freeze report. The pinned model is `NousResearch/Meta-Llama-3.1-8B` at revision `1f47e50cdbe801ad8a5174156ec3a0655108fb9f`; configuration is W=16, B=8, epsilon=0.05, rank=8, MPFR=256 bits, CPU eager bfloat16, and 32 query / 8 KV heads.

The complete all-head evidence is in `results/rack_kv_v2_final_trace_v2/`. The process-isolated physical controller is `python -m experiments.run_final_physical_controller --parallel 2`, but the frozen report intentionally uses only workers with complete 40-row artifacts.

Run `python -m experiments.freeze_step12` to regenerate the aggregation and figures from disk-only evidence.
