from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/rack_kv_v1.lock.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    keys = fields or sorted(set().union(*(row.keys() for row in rows)))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (list, dict, tuple)) else v
                             for k, v in row.items()})


def load_config(path):
    # JSON is a YAML 1.2 subset; no extra parser dependency is needed.
    config = read_json(path)
    lock = read_json(LOCK)
    if sha256(path) != lock["configuration_sha256"]:
        raise ValueError("Frozen v1 configuration changed; use a new experiment identity.")
    for name, expected in lock["source_files"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen scientific source changed: {name}")
    for name, expected in lock["prompt_files"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen prompt changed: {name}")
    return config


def seed_execution(config):
    import numpy as np
    import torch
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment(config):
    import psutil
    import torch
    versions = {}
    for package in ("numpy", "torch", "transformers", "safetensors", "gmpy2", "requests", "psutil", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "configuration": config,
            "git_commit": git.stdout.strip() if git.returncode == 0 else None,
            "git_status": "available" if git.returncode == 0 else "unavailable_from_workspace",
            "python": sys.version, "versions": versions, "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": psutil.virtual_memory().total,
            "cuda_version": torch.version.cuda, "device": "cpu", "hostname": platform.node(),
            "seeds": {k: config["seed"] for k in ("python", "numpy", "torch", "cuda_if_available")},
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "torch_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "command": [sys.executable, "-m", "experiments.reproduce_v1", *sys.argv[1:]],
            "freeze_lock_sha256": sha256(LOCK)}


def verify_frozen_input(path):
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    expected = read_json(LOCK)["evidence_files"].get(relative)
    if expected is None or sha256(path) != expected:
        raise ValueError(f"Evidence missing from freeze or hash mismatch: {relative}")
    return {"path": relative, "sha256": expected}


def inventory(output):
    return [{"path": p.relative_to(output).as_posix(), "sha256": sha256(p), "size_bytes": p.stat().st_size}
            for p in sorted(output.rglob("*")) if p.is_file() and p != output / "manifest.json"]


def verify_review_archive(path):
    import zipfile
    from pathlib import PurePosixPath
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member")
        for name in names:
            normalized = name.replace("\\", "/")
            if PurePosixPath(normalized).is_absolute() or ".." in PurePosixPath(normalized).parts or ":" in normalized:
                raise ValueError("Unsafe archive member")
        manifest = json.loads(archive.read("manifest.json"))
        members = {n.replace("\\", "/"): n for n in names}
        entries = manifest["files"]
        if isinstance(entries, dict):
            entries = [{"relative_path": name, **value} for name, value in entries.items()]
        for entry in entries:
            name = entry.get("relative_path", entry.get("path")).replace("\\", "/")
            if name == "manifest.json":
                raise ValueError("Self-hashed manifest")
            payload = archive.read(members[name])
            recorded_size = entry.get("size_bytes", entry.get("bytes", entry.get("size")))
            if recorded_size is None:
                raise ValueError(f"Archive manifest has no size field: {name}")
            if len(payload) != recorded_size or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError(f"Archive manifest mismatch: {name}")
    return {"path": str(path), "sha256": sha256(path), "file_count": len(entries), "mismatch_count": 0}
