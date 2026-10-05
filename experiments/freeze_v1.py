"""One-time snapshot creation. Existing locks are never overwritten."""
from pathlib import Path
import ast
import shutil
import zipfile

from .common import ROOT, LOCK, read_json, sha256, write_json


def main():
    if LOCK.exists():
        raise SystemExit("Freeze already exists; refusing to overwrite it.")
    prompts = ROOT / "configs/prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    for source, target in [("stage2b_prompt_metadata.json", "representative_metadata.json"),
                           ("stage2b_prompt_source.txt", "representative_source.txt")]:
        shutil.copy2(ROOT / ".tmp/stage2b_layer0_pilot/prompt" / source, prompts / target)
    config = read_json(ROOT / "configs/rack_kv_v1.yaml")
    sources = sorted([*ROOT.glob("rack_kv/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py")])
    evidence = [ROOT / config[k] for k in ("existing_representative", "existing_full_model")]
    evidence += sorted((ROOT / config["existing_capture"]).glob("*.safetensors"))
    evidence += [ROOT / config["existing_capture"] / "capture_report.json"]
    for path in [ROOT / ".tmp/stage4_baselines_full/manifest.json", ROOT / ".tmp/stage5_quality_final/manifest.json"]:
        if path.exists():
            evidence.append(path)
    # Include small provenance/token records, never model tensors.
    for folder in ("prompt_corpus", "metric_records", "certificate_records"):
        evidence += [p for p in (ROOT / ".tmp/stage5_quality_final" / folder).rglob("*") if p.is_file()]
    lock = {"schema": "rack_kv_v1_freeze_1", "configuration_sha256": sha256(ROOT / "configs/rack_kv_v1.yaml"),
            "source_files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
            "prompt_files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(prompts.iterdir())},
            "evidence_files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in evidence},
            "exact_historical_source_linkage": "Not inferred from this new freeze; inspect original package snapshots."}
    with zipfile.ZipFile(ROOT / "configs/rack_kv_v1_source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for p in sources:
            archive.write(p, p.relative_to(ROOT).as_posix())
    lock["source_archive_sha256"] = sha256(ROOT / "configs/rack_kv_v1_source_snapshot.zip")
    write_json(LOCK, lock)
    # AST inventory supplements the hand-written scientific code map.
    lines = ["# Frozen function inventory", "", "Generated from the frozen source; private helpers are included.", ""]
    for p in sources:
        lines += ["## " + p.relative_to(ROOT).as_posix(), ""]
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                lines.append(f"- `{node.name}` (line {node.lineno})")
        lines.append("")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/baseline_function_inventory.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Frozen {len(sources)} source files and {len(evidence)} evidence files.")


if __name__ == "__main__":
    main()
