# RACK-KV V2 Hierarchy Implementation

## Scope

This change adds V2 logical skipping without modifying any V1 source listed in
`configs/rack_kv_v1.lock.json`.  It retains the V1 codec, first-token anchor,
reconstructed keys and values, MPFR precision, and output-error theorem.  The
only new input to that theorem is an anisotropic MPFR upper mass bound.

## Execution paths

`rack_kv.anisotropic.execute_anisotropic_flat` implements
`rack_v2_aniso_flat_r4` and `rack_v2_aniso_flat_r8`.  It builds rank-specific
directional summaries, calls the existing theorem through the rigorous shadow
certificate, and then constructs attention from the recent window plus only
the retained reconstructed leaf blocks.  Float64 diagnostics have no
authorization path.

`rack_kv.hierarchy.execute_anisotropic_hierarchical` implements
`rack_v2_aniso_hier_r8`.  With `hierarchy_enabled=False`, it delegates exactly
to flat V2.  V1 sphere mode remains in the frozen V1 code and is not changed.

## Tree design and safety

`build_hierarchy` creates leaves from the existing consecutive compressed
blocks and groups them into fanout-sized consecutive regions until one root
remains.  Nodes retain only IDs, token ranges, child IDs, directional metadata,
and a rigorous value-norm bound.  Leaf compressed payloads remain external to
the index.

Traversal uses a deterministic coarse-first priority order: larger ranges,
then earlier token range, then node ID.  A candidate region is authorized only
after `_global_bound` recomputes the unchanged V1 theorem for the entire
already-skipped set plus that region.  Thus the implementation has one
cumulative `epsilon` budget per query head, rather than one budget per node.
When a parent is certified, none of its descendants are evaluated.  When it
is rejected, traversal descends.  A numerical error conservatively retains the
node.

For an eight-token leaf centered at its first token, there are at most seven
nonzero residual rows; its residual matrix consequently has rank at most seven.
This does not apply to internal nodes, whose residual radius is recomputed
rigorously for the requested rank.

## Files

- `rack_kv/anisotropic.py`: generic reconstructed-region summaries and real
  flat logical attention execution.
- `rack_kv/hierarchy.py`: metadata-only hierarchy, cumulative certification,
  and retained-leaf attention execution.
- `tests/test_v2_hierarchy.py`: fast structural, geometric, authorization,
  output, rank, and V1-lock invariants.

## Fast validation performed

```powershell
python -m py_compile rack_kv\anisotropic.py rack_kv\hierarchy.py
python -m unittest tests.test_anisotropic_certificate tests.test_v2_hierarchy
```

The focused suite passed: 16 tests.  No model, trace capture, or large
representative benchmark was run.

## Remaining validation

The next combined scientific run should replay the frozen representative
payloads once and compare `rack_v1_sphere`, flat rank 4, flat rank 8, and
hierarchical rank 8 under the same 6,750 decisions.  It must also measure the
actual retained-attention error, tree work avoided, and all MPFR fallbacks.
Internal-node metadata is currently held as float64 basis/interval data plus
MPFR residual and value-norm bounds; physical GQA payload avoidance is not
implemented.

Run that deferred validation, and only that validation, with:

```powershell
python -m experiments.run_v2_combined_validation --config configs/rack_kv_v1.yaml --output results/rack_kv_v2_combined_validation --flat-ranks 0 4 8 --fanout 4
```

The command replays frozen traces and serialized representative payloads.  It
does not instantiate Llama, recapture a trace, or download tensors.
