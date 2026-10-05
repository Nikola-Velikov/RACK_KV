from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import socket
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

import gmpy2
import numpy as np
import psutil
import safetensors
import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file as load_safetensors_file, save_file as save_safetensors_file
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from .certificate import (
    PreparedCertificationInputs,
    certify_progressive_skipping,
    certify_progressive_skipping_prepared,
    exact_reference_output_mpfr,
    exact_reference_output_from_exact_rows,
    prepare_certification_inputs,
    prepared_full_exact_rows,
    prepared_kept_exact_rows,
    rigorous_attention_output_interval,
    rigorous_attention_output_interval_from_exact_rows,
    rigorous_output_error_norm,
    rigorous_output_error_upper_from_intervals,
)
from .llama_trace import (
    DEFAULT_LLAMA31_BASE_REPO,
    CheckpointConfigSummary,
    _DownloadPolicy,
    _RemoteSafetensorsShard,
    _RemoteTensorSpec,
    _decode_tensor_bytes,
    _fetch_repo_revision,
    _fetch_weight_index,
    _load_checkpoint_summary,
    _load_decoder_layer_from_state_dict,
    _prepare_local_assets,
    _range_get,
    _sha256_bytes,
    _sha256_file,
    ensure_minimal_llama31_assets,
    query_head_to_kv_head,
)
from .codec import CompressedBlock, encode_block, serialize_block_container, deserialize_block_container
from .rigorous import exact_vector
from .stage2 import (
    _skip_validation_flags,
)
from .stage4 import (
    ByteBreakdown,
    _decode_uniform_int8_kv,
    _section_header_payload,
    _bf16_tensor_to_bytes,
    _serialize_uniform_int8_kv,
)


STAGE5_METHOD_VERSION = "stage5_quality_v4"
STAGE5_RESULT_VERSION = "stage5_quality_results_v4"
STAGE5_STORAGE_ACCOUNTING_SCHEMA = "stage5_storage_accounting_v2"
STAGE5_PROMPT_CORPUS_VERSION = "stage5_prompt_corpus_v2"
STAGE5_RUN_SCHEMA = "stage5_quality_run_v4"
STAGE5_PROMPT_CHECKPOINT_SCHEMA = "stage5_prompt_checkpoint_v2"
STAGE5_STREAM_CHECKPOINT_SCHEMA = "stage5_stream_checkpoint_v1"
STAGE5_CERTIFICATE_RECORD_SCHEMA = "stage5_certificate_record_v2"
STAGE5_METRIC_RECORD_SCHEMA = "stage5_metric_record_v1"
STAGE5_EXPERIMENT_SCOPE_LAYER0_SAFETY_VALIDATION = "layer_0_safety_validation"
STAGE5_MODIFIED_LAYERS = (0, 8, 16, 24, 31)
STAGE5_METHODS = ("full_kv", "rack_kv_compression_only", "rack_kv_certified", "uniform_int8_kv")
STAGE5_RECENT_WINDOW = 16
STAGE5_BLOCK_SIZE = 8
STAGE5_TOLERANCE = 0.05
STAGE5_PRECISION = 256
STAGE5_SEED = 0
STAGE5_SCORE_START = 31
STAGE5_PROMPT_TOKEN_COUNT = 128
STAGE5_PASSKEY_SEED = 20260723
STAGE5_DEFAULT_OUTPUT_DIR = Path(".tmp/stage5_quality_final")
STAGE5_DEFAULT_CAPTURE_DIR = Path(".tmp/stage3_multilayer_full/capture")
STAGE5_DEFAULT_REVIEW_ZIP = Path(".tmp/stage5_quality_final_review.zip")
STAGE5_DEFAULT_TENSOR_CACHE_DIR = Path(".tmp/stage5_tensor_cache")
STAGE5_DEFAULT_MAX_RSS_GB = 8.0
STAGE5_DEFAULT_MIN_FREE_GB = 2.0
STAGE5_DEFAULT_TOKEN_CHUNK_SIZE = 8
STAGE5_CERTIFIED_TOKEN_CHUNK_SIZE = 1
STAGE5_UNIFORM_INT8_TOKEN_CHUNK_SIZE = 4
STAGE5_TEST_LOG = "test_log.txt"
STAGE5_RESULTS_JSON = "stage5_quality_final_results.json"
STAGE5_REPORT_MD = "stage5_quality_final_report.md"
STAGE5_MANIFEST_JSON = "manifest.json"
STAGE5_PROMPT_DIR = "prompt_corpus"
STAGE5_CHECKPOINT_DIR = "checkpoints"
STAGE5_METRIC_DIR = "metric_records"
STAGE5_CERTIFICATE_DIR = "certificate_records"
STAGE5_PROVENANCE_DIR = "provenance"
STAGE5_REVIEW_SOURCE_FILES = (
    "rack_kv/stage5.py",
    "rack_kv/stage4.py",
    "rack_kv/stage3.py",
    "rack_kv/stage2.py",
    "rack_kv/certificate.py",
    "rack_kv/codec.py",
    "rack_kv/rigorous.py",
    "rack_kv/llama_trace.py",
    "scripts/run_stage5_quality_smoke.py",
    "tests/test_codec_and_accounting.py",
    "tests/test_stage5_quality.py",
    "tests/test_stage4_baselines.py",
    "tests/test_stage3_multilayer.py",
    "tests/test_stage2_integration.py",
    "tests/test_stage2b.py",
    "tests/test_stage2b_prompt.py",
    "tests/test_stage2b_resumable.py",
    "tests/test_rigorous_certificate.py",
    "tests/test_llama_trace.py",
)
STAGE5_METHOD_IMPLEMENTATION_VERSIONS: dict[str, str] = {
    "full_kv": "stage5_full_kv_v2",
    "rack_kv_compression_only": "stage5_rack_kv_compression_only_v2",
    "rack_kv_certified": "stage5_rack_kv_certified_v3",
    "uniform_int8_kv": "stage5_uniform_int8_kv_v1",
}


def _normalize_stage5_methods(method_names: Sequence[str] | None) -> tuple[str, ...]:
    if method_names is None:
        return tuple(STAGE5_METHODS)
    normalized: list[str] = []
    seen: set[str] = set()
    for method_name in method_names:
        value = str(method_name).strip()
        if not value:
            continue
        if value not in STAGE5_METHODS:
            raise Stage5ExecutionError(f"Unsupported Stage 5 method selection: {value!r}.")
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    if not normalized:
        raise Stage5ExecutionError("No valid Stage 5 methods were selected.")
    if any(method != "full_kv" for method in normalized) and "full_kv" not in seen:
        normalized.insert(0, "full_kv")
    return tuple(normalized)


def _stage5_method_impl_version(method_name: str) -> str:
    if method_name not in STAGE5_METHOD_IMPLEMENTATION_VERSIONS:
        raise Stage5ExecutionError(f"Missing Stage 5 implementation version for method {method_name!r}.")
    return STAGE5_METHOD_IMPLEMENTATION_VERSIONS[method_name]


class Stage5ExecutionError(RuntimeError):
    pass


class Stage5TensorCacheError(Stage5ExecutionError):
    pass


class Stage5MemoryGuardExceeded(Stage5ExecutionError):
    pass


@dataclass(frozen=True)
class PromptMaterialized:
    name: str
    source_text: str
    source_sha256: str
    token_ids: tuple[int, ...]
    token_count: int
    decoded_text: str
    tokenizer_name: str
    tokenizer_revision: str
    metadata_path: Path
    source_path: Path
    token_ids_path: Path
    answer_text: str | None = None
    answer_token_ids: tuple[int, ...] | None = None
    final_answer_start_token_index: int | None = None
    generation_seed: int | None = None


@dataclass(frozen=True)
class Stage5PromptRunResult:
    prompt: PromptMaterialized
    scored_token_count: int
    prompt_runtime_s: float
    prompt_peak_rss_bytes: int
    method_aggregates: dict[str, Any]
    per_token_records_path: Path
    certificate_records_path: Path
    intermediate_layer_metrics: dict[str, Any]
    passkey_metrics: dict[str, Any] | None


@dataclass(frozen=True)
class TensorRangeRecord:
    repo_id: str
    revision: str
    shard_name: str
    tensor_name: str
    range_start: int
    range_end_inclusive: int
    sha256: str
    byte_count: int
    source: str
    path: str


class MemoryGuard:
    def __init__(self, *, max_rss_bytes: int, min_free_bytes: int) -> None:
        self.max_rss_bytes = int(max_rss_bytes)
        self.min_free_bytes = int(min_free_bytes)
        self.start_rss_bytes = self._rss_bytes()
        self.peak_rss_bytes = self.start_rss_bytes
        self.end_rss_bytes = self.start_rss_bytes
        self.min_available_bytes = self._available_bytes()

    @staticmethod
    def _rss_bytes() -> int:
        return int(psutil.Process().memory_info().rss)

    @staticmethod
    def _available_bytes() -> int:
        return int(psutil.virtual_memory().available)

    def check(self, stage: str) -> tuple[int, int]:
        rss = self._rss_bytes()
        free = self._available_bytes()
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self.end_rss_bytes = rss
        self.min_available_bytes = min(self.min_available_bytes, free)
        if rss > self.max_rss_bytes:
            raise Stage5MemoryGuardExceeded(
                f"Stage 5 memory guard exceeded at {stage}: RSS {rss} > limit {self.max_rss_bytes}."
            )
        if free < self.min_free_bytes:
            raise Stage5MemoryGuardExceeded(
                f"Stage 5 memory guard exceeded at {stage}: available memory {free} < floor {self.min_free_bytes}."
            )
        return rss, free


def _write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
    return path


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    return _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True))


def _json_roundtrip(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _source_snapshot_sha256(*, base_dir: Path, relative_paths: Sequence[str]) -> str:
    lines: list[str] = []
    for relative_path in sorted(relative_paths):
        full_path = base_dir / relative_path
        if not full_path.exists():
            raise Stage5ExecutionError(f"Missing source file for Stage 5 snapshot: {full_path}")
        lines.append(f"{relative_path}\t{_sha256_file(full_path)}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def stage5_dependency_versions() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "gmpy2_version": gmpy2.version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "safetensors_version": safetensors.__version__,
        "device": "cpu",
    }


def inspect_local_model_state(
    *,
    repo_id: str,
    revision: str,
    capture_dir: Path,
    tensor_cache_dir: Path,
) -> dict[str, Any]:
    hf_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_cache = hf_root / f"models--{repo_id.replace('/', '--')}"
    usage = shutil.disk_usage(capture_dir.parents[1] if len(capture_dir.parents) >= 2 else capture_dir)
    base_asset_dir = capture_dir / "llama31_base_assets" / revision
    base_assets = sorted(path.name for path in base_asset_dir.iterdir()) if base_asset_dir.exists() else []
    tensor_cache_files = list(tensor_cache_dir.rglob("*.bin")) if tensor_cache_dir.exists() else []
    tensor_cache_bytes = sum(path.stat().st_size for path in tensor_cache_files)
    return {
        "repo_id": repo_id,
        "revision": revision,
        "capture_dir": str(capture_dir),
        "stage3_capture_exists": capture_dir.exists(),
        "stage3_trace_files": sorted(path.name for path in capture_dir.glob("llama31_layer*_trace.safetensors")),
        "stage3_base_asset_dir": str(base_asset_dir),
        "stage3_base_assets_present": base_assets,
        "huggingface_repo_cache_exists": repo_cache.exists(),
        "huggingface_repo_cache_path": str(repo_cache),
        "project_tensor_cache_path": str(tensor_cache_dir),
        "project_tensor_cache_file_count": len(tensor_cache_files),
        "project_tensor_cache_bytes": tensor_cache_bytes,
        "disk_total_bytes": usage.total,
        "disk_used_bytes": usage.used,
        "disk_free_bytes": usage.free,
    }


class PersistentTensorRangeCache:
    def __init__(
        self,
        *,
        root: Path,
        repo_id: str,
        revision: str,
        download_policy: _DownloadPolicy,
        range_fetcher: Any | None = None,
    ) -> None:
        self.root = Path(root)
        self.repo_id = repo_id
        self.revision = revision
        self.download_policy = download_policy
        self.range_fetcher = range_fetcher
        self.root.mkdir(parents=True, exist_ok=True)
        self._shards: dict[str, _RemoteSafetensorsShard] = {}
        self.records: list[TensorRangeRecord] = []
        self.sources_count = {
            "project_local_persistent_cache": 0,
            "newly_fetched_remote_range": 0,
        }
        self.remote_bytes_fetched = 0
        self.invalid_cache_entries = 0

    def _repo_cache_metadata_dir(self) -> Path:
        return self._repo_cache_dir() / "_metadata"

    def _shard_metadata_path(self, shard_name: str) -> Path:
        shard_key = hashlib.sha256(shard_name.encode("utf-8")).hexdigest()[:16]
        return self._repo_cache_metadata_dir() / f"{shard_key}.json"

    def _load_cached_shard(self, shard_name: str) -> _RemoteSafetensorsShard | None:
        metadata_path = self._shard_metadata_path(shard_name)
        if not metadata_path.exists():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            payload.get("repo_id") != self.repo_id
            or payload.get("revision") != self.revision
            or payload.get("shard_name") != shard_name
        ):
            return None
        tensor_specs_raw = payload.get("tensor_specs")
        if not isinstance(tensor_specs_raw, dict):
            return None
        tensor_specs: dict[str, _RemoteTensorSpec] = {}
        try:
            for tensor_name, entry in tensor_specs_raw.items():
                tensor_specs[str(tensor_name)] = _RemoteTensorSpec(
                    dtype_code=str(entry["dtype_code"]),
                    shape=tuple(int(dim) for dim in entry["shape"]),
                    byte_start=int(entry["byte_start"]),
                    byte_end_exclusive=int(entry["byte_end_exclusive"]),
                )
        except (KeyError, TypeError, ValueError):
            return None
        return _RemoteSafetensorsShard(
            url=f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{shard_name}",
            tensor_specs=tensor_specs,
            download_policy=self.download_policy,
            fetched_tensor_hashes={},
        )

    def _save_cached_shard(self, shard_name: str, shard: _RemoteSafetensorsShard) -> None:
        metadata_path = self._shard_metadata_path(shard_name)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "shard_name": shard_name,
            "url": shard.url,
            "tensor_specs": {
                tensor_name: {
                    "dtype_code": spec.dtype_code,
                    "shape": list(spec.shape),
                    "byte_start": spec.byte_start,
                    "byte_end_exclusive": spec.byte_end_exclusive,
                }
                for tensor_name, spec in shard.tensor_specs.items()
            },
        }
        _write_json_atomic(metadata_path, payload)

    def _repo_cache_dir(self) -> Path:
        return self.root / quote(self.repo_id, safe="") / self.revision

    def _shard(self, shard_name: str) -> _RemoteSafetensorsShard:
        cached = self._shards.get(shard_name)
        if cached is not None:
            return cached
        cached = self._load_cached_shard(shard_name)
        if cached is not None:
            self._shards[shard_name] = cached
            return cached
        shard = _RemoteSafetensorsShard.from_repo(
            repo_id=self.repo_id,
            revision=self.revision,
            filename=shard_name,
            download_policy=self.download_policy,
            fetched_tensor_hashes={},
        )
        self._shards[shard_name] = shard
        self._save_cached_shard(shard_name, shard)
        return shard

    def _payload_paths(
        self,
        *,
        shard_name: str,
        tensor_name: str,
        start: int,
        end_inclusive: int,
    ) -> tuple[Path, Path]:
        shard_key = hashlib.sha256(shard_name.encode("utf-8")).hexdigest()[:16]
        tensor_key = hashlib.sha256(tensor_name.encode("utf-8")).hexdigest()[:24]
        parent = self._repo_cache_dir() / shard_key / tensor_key
        stem = f"{start}_{end_inclusive}"
        return parent / f"{stem}.bin", parent / f"{stem}.json"

    def _fetch_bytes(self, *, shard_url: str, start: int, end_inclusive: int) -> bytes:
        if self.range_fetcher is not None:
            return self.range_fetcher(shard_url, start, end_inclusive)
        return _range_get(
            shard_url,
            start,
            end_inclusive,
            download_policy=self.download_policy,
        )

    def _load_or_fetch_payload(
        self,
        *,
        shard_name: str,
        tensor_name: str,
        start: int,
        end_inclusive: int,
    ) -> tuple[bytes, str, str, Path]:
        payload_path, metadata_path = self._payload_paths(
            shard_name=shard_name,
            tensor_name=tensor_name,
            start=start,
            end_inclusive=end_inclusive,
        )
        if payload_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload = payload_path.read_bytes()
            actual_sha = _sha256_bytes(payload)
            if (
                metadata.get("repo_id") == self.repo_id
                and metadata.get("revision") == self.revision
                and metadata.get("shard_name") == shard_name
                and metadata.get("tensor_name") == tensor_name
                and int(metadata.get("range_start")) == start
                and int(metadata.get("range_end_inclusive")) == end_inclusive
                and metadata.get("sha256") == actual_sha
            ):
                self.sources_count["project_local_persistent_cache"] += 1
                return payload, actual_sha, "project_local_persistent_cache", payload_path
            self.invalid_cache_entries += 1
            try:
                payload_path.unlink()
            except OSError:
                pass
            try:
                metadata_path.unlink()
            except OSError:
                pass

        shard = self._shard(shard_name)
        payload = self._fetch_bytes(shard_url=shard.url, start=start, end_inclusive=end_inclusive)
        sha256 = _sha256_bytes(payload)
        self.remote_bytes_fetched += len(payload)
        self.sources_count["newly_fetched_remote_range"] += 1
        metadata_payload = {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "shard_name": shard_name,
            "tensor_name": tensor_name,
            "range_start": start,
            "range_end_inclusive": end_inclusive,
            "byte_count": len(payload),
            "sha256": sha256,
        }
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        temp_payload = payload_path.with_name(payload_path.name + ".tmp")
        temp_meta = metadata_path.with_name(metadata_path.name + ".tmp")
        try:
            temp_payload.write_bytes(payload)
            temp_meta.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temp_payload, payload_path)
            os.replace(temp_meta, metadata_path)
        finally:
            for temp_path in (temp_payload, temp_meta):
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
        return payload, sha256, "newly_fetched_remote_range", payload_path

    def fetch_tensor(self, *, weight_index: dict[str, Any], tensor_name: str) -> torch.Tensor:
        shard_name = weight_index["weight_map"][tensor_name]
        shard = self._shard(shard_name)
        spec = shard.tensor_spec(tensor_name)
        payload, sha256, source, payload_path = self._load_or_fetch_payload(
            shard_name=shard_name,
            tensor_name=tensor_name,
            start=spec.byte_start,
            end_inclusive=spec.byte_end_exclusive - 1,
        )
        self.records.append(
            TensorRangeRecord(
                repo_id=self.repo_id,
                revision=self.revision,
                shard_name=shard_name,
                tensor_name=tensor_name,
                range_start=spec.byte_start,
                range_end_inclusive=spec.byte_end_exclusive - 1,
                sha256=sha256,
                byte_count=len(payload),
                source=source,
                path=str(payload_path),
            )
        )
        return _decode_tensor_bytes(payload, spec)

    def fetch_row_range(
        self,
        *,
        weight_index: dict[str, Any],
        tensor_name: str,
        row_start: int,
        row_end_exclusive: int,
    ) -> torch.Tensor:
        if row_end_exclusive <= row_start:
            raise ValueError("row_end_exclusive must be greater than row_start.")
        shard_name = weight_index["weight_map"][tensor_name]
        shard = self._shard(shard_name)
        spec = shard.tensor_spec(tensor_name)
        if len(spec.shape) != 2:
            raise ValueError(f"Row-range fetch is only supported for rank-2 tensors, not {tensor_name!r}.")
        rows, width = spec.shape
        if row_start < 0 or row_end_exclusive > rows:
            raise IndexError(f"Row range [{row_start}, {row_end_exclusive}) is outside {tensor_name!r} shape {spec.shape}.")
        row_bytes = width * spec.itemsize
        start = spec.byte_start + row_start * row_bytes
        end_inclusive = spec.byte_start + row_end_exclusive * row_bytes - 1
        payload, sha256, source, payload_path = self._load_or_fetch_payload(
            shard_name=shard_name,
            tensor_name=f"{tensor_name}[rows={row_start}:{row_end_exclusive}]",
            start=start,
            end_inclusive=end_inclusive,
        )
        row_spec = _RemoteTensorSpec(
            dtype_code=spec.dtype_code,
            shape=(row_end_exclusive - row_start, width),
            byte_start=start,
            byte_end_exclusive=end_inclusive + 1,
        )
        self.records.append(
            TensorRangeRecord(
                repo_id=self.repo_id,
                revision=self.revision,
                shard_name=shard_name,
                tensor_name=f"{tensor_name}[rows={row_start}:{row_end_exclusive}]",
                range_start=start,
                range_end_inclusive=end_inclusive,
                sha256=sha256,
                byte_count=len(payload),
                source=source,
                path=str(payload_path),
            )
        )
        return _decode_tensor_bytes(payload, row_spec)

    def fetch_rows(
        self,
        *,
        weight_index: dict[str, Any],
        tensor_name: str,
        row_indices: Sequence[int],
    ) -> torch.Tensor:
        if not row_indices:
            raise ValueError("row_indices must be non-empty.")
        unique_rows = sorted({int(row) for row in row_indices})
        chunks: list[tuple[int, int]] = []
        start = unique_rows[0]
        end = start + 1
        for row in unique_rows[1:]:
            if row == end:
                end += 1
            else:
                chunks.append((start, end))
                start = row
                end = row + 1
        chunks.append((start, end))
        row_lookup: dict[int, torch.Tensor] = {}
        for chunk_start, chunk_end in chunks:
            fetched = self.fetch_row_range(
                weight_index=weight_index,
                tensor_name=tensor_name,
                row_start=chunk_start,
                row_end_exclusive=chunk_end,
            )
            for offset, row_index in enumerate(range(chunk_start, chunk_end)):
                row_lookup[row_index] = fetched[offset]
        return torch.stack([row_lookup[int(row_index)] for row_index in row_indices], dim=0)

    def provenance_summary(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "cache_root": str(self.root),
            "records_count": len(self.records),
            "sources_count": dict(self.sources_count),
            "remote_bytes_fetched": self.remote_bytes_fetched,
            "invalid_cache_entries": self.invalid_cache_entries,
            "records": [asdict(record) for record in self.records],
        }


@dataclass(frozen=True)
class _PrefixSerializationContext:
    record_index: int
    kv_head_global: int
    visible_length: int
    historical_tokens: int
    recent_exact_tokens: int
    head_dim: int
    prefix_keys_tensor: torch.Tensor
    prefix_values_tensor: torch.Tensor
    prefix_keys: np.ndarray | None
    prefix_values: np.ndarray | None
    full_kv_reference_bytes: int


@dataclass(frozen=True)
class _Stage5RackPrefixPayload:
    blocks: tuple[CompressedBlock, ...]
    reconstructed_keys: np.ndarray
    reconstructed_values: np.ndarray
    recent_keys: np.ndarray | None
    recent_values: np.ndarray | None
    byte_breakdown: ByteBreakdown
    diagnostics: dict[str, Any]


@dataclass
class _Stage5IncrementalRackState:
    recent_window: int
    block_size: int
    precision: int
    head_dim: int
    value_dim: int
    finalized_blocks: list[CompressedBlock]
    finalized_block_payload_lengths: list[int]
    finalized_key_blocks: list[np.ndarray]
    finalized_value_blocks: list[np.ndarray]
    recent_keys: list[np.ndarray]
    recent_values: list[np.ndarray]
    tail_keys: list[np.ndarray]
    tail_values: list[np.ndarray]


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _tensor_container_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return _tensor_bytes(value)
    if isinstance(value, Mapping):
        return sum(_tensor_container_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_container_bytes(item) for item in value)
    return 0


def _float64_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float64).numpy()


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            handle = kernel32.GetCurrentProcess()
            psapi.EmptyWorkingSet(handle)
        except Exception:
            pass


def _block_slices(token_count: int, block_size: int) -> tuple[tuple[int, int], ...]:
    if token_count <= 0:
        return tuple()
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    return tuple(
        (start, min(start + block_size, token_count))
        for start in range(0, token_count, block_size)
    )


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64), ord=2))


def _relative_l2_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    numerator = _norm(reference - candidate)
    denominator = _norm(reference)
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return float(numerator / denominator)


def _cosine_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    ref_norm = _norm(ref)
    cand_norm = _norm(cand)
    if ref_norm == 0.0 or cand_norm == 0.0:
        return 1.0 if ref_norm == cand_norm else 0.0
    return float(np.dot(ref, cand) / (ref_norm * cand_norm))


def _max_abs_component_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(reference, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)), initial=0.0))


def _rmse(diff: np.ndarray) -> float:
    array = np.asarray(diff, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64), dtype=np.float64)))


def _topk_ids(logits: np.ndarray, k: int = 5) -> list[int]:
    limit = min(k, int(logits.shape[0]))
    order = np.argpartition(-logits, range(limit))[:limit]
    sorted_idx = order[np.argsort(-logits[order])]
    return [int(value) for value in sorted_idx.tolist()]


def _logsumexp(logits: np.ndarray) -> float:
    maximum = float(np.max(logits))
    return float(maximum + np.log(np.sum(np.exp(logits - maximum, dtype=np.float64), dtype=np.float64)))


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    return logits - _logsumexp(logits)


def _softmax_from_log(log_probs: np.ndarray) -> np.ndarray:
    return np.exp(log_probs, dtype=np.float64)


def _kl_divergence(log_p: np.ndarray, log_q: np.ndarray) -> float:
    p = _softmax_from_log(log_p)
    return float(np.sum(p * (log_p - log_q), dtype=np.float64))


def _js_divergence(log_p: np.ndarray, log_q: np.ndarray) -> float:
    p = _softmax_from_log(log_p)
    q = _softmax_from_log(log_q)
    m = 0.5 * (p + q)
    log_m = np.log(m, dtype=np.float64)
    return float(0.5 * np.sum(p * (log_p - log_m), dtype=np.float64) + 0.5 * np.sum(q * (log_q - log_m), dtype=np.float64))


def _rank_of_token(logits: np.ndarray, token_id: int) -> int:
    order = np.argsort(-logits)
    matches = np.where(order == int(token_id))[0]
    if matches.size == 0:
        return int(logits.shape[0]) + 1
    return int(matches[0]) + 1


def _build_prompt_texts() -> dict[str, str]:
    natural = (
        "Section A. The workshop memo explains a small archival project. "
        "Three volunteers sort field notes, summarize interviews, and record dates in plain language. "
        "One person checks names, another checks numbers, and a third compares the summary with the original paragraph. "
        "The memo repeats a simple rule: preserve meaning, avoid dramatic wording, and write only what the record supports. "
        "Section B. During the afternoon, the team reviews short reports about river levels, town meetings, and library repairs. "
        "Each report is ordinary, factual, and written in a calm tone. "
        "A recurring detail says that the blue folder contains draft maps, while the gray folder contains signed copies. "
        "Later, a coordinator reminds everyone that the map numbers matter less than the final verified annotations. "
        "Section C. The closing paragraph notes that careful revision is slower than quick guessing, but the slower method avoids preventable mistakes and makes later comparison easier."
    )
    rng = random.Random(STAGE5_PASSKEY_SEED)
    syllables = ["amber", "cinder", "harbor", "juniper", "marble", "signal", "thistle", "velvet"]
    tag = rng.choice(syllables)
    digits = "".join(str(rng.randint(0, 9)) for _ in range(6))
    passkey = f"{tag}-{digits}"
    filler_sentences = [
        "The clerk copied weather observations into a ledger and then paused to label each page.",
        "A second note described chairs, lamps, windows, and an inventory of sealed envelopes.",
        "Later lines mentioned ordinary errands, routine maintenance, and a discussion about filing order.",
        "Nothing in the middle section changed the passkey, and no later sentence introduced a second answer.",
        "Another plain note followed.",
    ]
    intro = (
        f"Archive note. Early in the record, the custodian wrote that the recovery passkey is {passkey}. "
        "This statement appears once in the factual note and should be preserved exactly. "
    )
    outro = (
        "Question. What is the recovery passkey? "
        f"Answer: {passkey}."
    )
    passkey_text = intro + " ".join(filler_sentences) + " " + outro
    structured_reasoning = (
        "Reasoning note. Mira keeps the brass key in drawer A, and drawer A is inside the green cabinet. "
        "The green cabinet stands in room three beside a shelf of maps. "
        "A later memo says room three is on the east side of the archive, while room four is on the west side. "
        "Another line explains that the east-side room with the green cabinet is the place to visit first when an item is stored in drawer A. "
        "No later sentence moves the cabinet or the drawer. "
        "Question. If someone needs the brass key, which room should they visit first? "
        "Answer: room three."
    )
    code_context = (
        "Python note. A helper function named scale_value multiplies an input number by a factor. "
        "The default factor is 4 unless a different factor is passed explicitly. "
        "An example line says result = scale_value(7), and the nearby comment explains that this call uses the default factor. "
        "Another comment warns that later examples with custom factors do not change the default behavior of the original call. "
        "Question. What numeric value should result contain after running result = scale_value(7) with the default factor? "
        "Answer: 28."
    )
    return {
        "natural_language_128": natural,
        "passkey_retrieval_128": passkey_text,
        "structured_reasoning_128": structured_reasoning,
        "code_context_128": code_context,
    }


def _tokenizer_from_assets(asset_dir: Path) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(asset_dir, local_files_only=True)


def _tokenize_to_fixed_length(
    *,
    tokenizer: AutoTokenizer,
    source_text: str,
    target_tokens: int,
    extension_sentences: Sequence[str],
) -> tuple[tuple[int, ...], str]:
    working = source_text
    for sentence in extension_sentences:
        token_ids = tokenizer(working, add_special_tokens=True)["input_ids"]
        if len(token_ids) >= target_tokens:
            break
        working = f"{working} {sentence}".strip()
    token_ids = tokenizer(working, add_special_tokens=True)["input_ids"]
    if len(token_ids) < target_tokens:
        raise Stage5ExecutionError(f"Prompt could not be extended deterministically to {target_tokens} tokens.")
    selected = tuple(int(token_id) for token_id in token_ids[:target_tokens])
    decoded = tokenizer.decode(list(selected), skip_special_tokens=False)
    return selected, decoded


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> list[int]:
    matches: list[int] = []
    if not needle or len(needle) > len(haystack):
        return matches
    last = len(haystack) - len(needle) + 1
    for start in range(last):
        if tuple(haystack[start : start + len(needle)]) == tuple(needle):
            matches.append(start)
    return matches


def build_stage5_prompt_corpus(
    *,
    asset_dir: Path,
    output_dir: Path,
    revision: str,
    target_token_count: int = STAGE5_PROMPT_TOKEN_COUNT,
    prompt_names: Sequence[str] | None = None,
) -> tuple[PromptMaterialized, ...]:
    prompt_dir = output_dir / STAGE5_PROMPT_DIR
    prompt_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _tokenizer_from_assets(asset_dir)
    base_texts = _build_prompt_texts()
    if prompt_names is not None:
        requested_names = tuple(str(name) for name in prompt_names)
        base_texts = {
            name: text
            for name, text in base_texts.items()
            if name in requested_names
        }
        if not base_texts:
            raise Stage5ExecutionError(f"No Stage 5 prompts matched requested names: {requested_names}.")
    extension_sentences = (
        "A final neutral sentence extends the passage without changing the core facts.",
        "Another line repeats that the document is deterministic and intended for controlled offline testing.",
        "The note ends with one more ordinary observation about labels, shelves, and timestamps.",
    )
    prompts: list[PromptMaterialized] = []
    for name, source_text in base_texts.items():
        answer_text = None
        answer_token_ids: tuple[int, ...] | None = None
        final_answer_start_token_index: int | None = None
        generation_seed: int | None = None
        if name == "passkey_retrieval_128":
            answer_text = source_text.split("Answer:", 1)[1].strip().rstrip(".")
            encoded = tokenizer(source_text, add_special_tokens=True, return_offsets_mapping=True)
            token_ids_full = [int(token_id) for token_id in encoded["input_ids"]]
            if len(token_ids_full) < int(target_token_count):
                raise Stage5ExecutionError(f"Passkey prompt must tokenize to at least {int(target_token_count)} tokens.")
            token_ids = tuple(token_ids_full[: int(target_token_count)])
            decoded = tokenizer.decode(list(token_ids), skip_special_tokens=False)
            answer_char_start = source_text.rfind(answer_text)
            answer_char_end = answer_char_start + len(answer_text)
            answer_token_positions = [
                index
                for index, (start, end) in enumerate(encoded["offset_mapping"])
                if int(start) < answer_char_end and int(end) > answer_char_start
            ]
            answer_token_positions = [
                index
                for index in answer_token_positions
                if index < int(target_token_count)
            ]
            if not answer_token_positions:
                raise Stage5ExecutionError(
                    f"Passkey answer token span is missing from the first {int(target_token_count)} tokens."
                )
            final_answer_start_token_index = int(answer_token_positions[0])
            answer_token_ids = tuple(int(token_ids[index]) for index in answer_token_positions)
            generation_seed = STAGE5_PASSKEY_SEED
        else:
            token_ids, decoded = _tokenize_to_fixed_length(
                tokenizer=tokenizer,
                source_text=source_text,
                target_tokens=int(target_token_count),
                extension_sentences=extension_sentences,
            )
        source_sha = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        source_path = prompt_dir / f"{name}.txt"
        token_ids_path = prompt_dir / f"{name}_token_ids.json"
        metadata_path = prompt_dir / f"{name}_metadata.json"
        _write_text_atomic(source_path, source_text)
        _write_json_atomic(token_ids_path, {"name": name, "token_ids": list(token_ids)})
        metadata = {
            "schema": STAGE5_PROMPT_CORPUS_VERSION,
            "name": name,
            "tokenizer_name": str(tokenizer.name_or_path),
            "tokenizer_revision": revision,
            "source_sha256": source_sha,
            "selection_rule": f"first_{int(target_token_count)}_tokens_after_tokenization",
            "target_token_count": int(target_token_count),
            "token_count": len(token_ids),
            "token_ids_path": str(token_ids_path),
            "source_path": str(source_path),
            "decoded_text": decoded,
            "answer_text": answer_text,
            "answer_token_ids": list(answer_token_ids) if answer_token_ids is not None else None,
            "final_answer_start_token_index": final_answer_start_token_index,
            "generation_seed": generation_seed,
        }
        _write_json_atomic(metadata_path, metadata)
        prompts.append(
            PromptMaterialized(
                name=name,
                source_text=source_text,
                source_sha256=source_sha,
                token_ids=token_ids,
                token_count=len(token_ids),
                decoded_text=decoded,
                tokenizer_name=str(tokenizer.name_or_path),
                tokenizer_revision=revision,
                metadata_path=metadata_path,
                source_path=source_path,
                token_ids_path=token_ids_path,
                answer_text=answer_text,
                answer_token_ids=answer_token_ids,
                final_answer_start_token_index=final_answer_start_token_index,
                generation_seed=generation_seed,
            )
        )
    return tuple(prompts)


def _gqa_groups(num_attention_heads: int, num_key_value_heads: int) -> tuple[tuple[int, ...], ...]:
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
    group_size = num_attention_heads // num_key_value_heads
    return tuple(
        tuple(range(group_index * group_size, (group_index + 1) * group_size))
        for group_index in range(num_key_value_heads)
    )


def _prepare_local_assets_from_stage3(
    *,
    capture_dir: Path,
    repo_id: str,
    revision: str,
    download_policy: _DownloadPolicy,
) -> tuple[Path, dict[str, str], dict[str, str]]:
    stage3_asset_dir = capture_dir / "llama31_base_assets" / revision
    if stage3_asset_dir.exists():
        asset_hashes = {
            path.name: _sha256_file(path)
            for path in stage3_asset_dir.iterdir()
            if path.is_file()
        }
        asset_sources = {name: "stage3_capture_asset" for name in asset_hashes}
        return stage3_asset_dir, asset_hashes, asset_sources
    preparation = ensure_minimal_llama31_assets(
        output_dir=capture_dir,
        repo_id=repo_id,
        repo_revision=revision,
        allow_insecure_tls=not download_policy.verify_tls,
    )
    return preparation.asset_dir, preparation.asset_hashes, preparation.asset_sources


def _fetch_prompt_embeddings_with_cache(
    *,
    tensor_cache: PersistentTensorRangeCache,
    weight_index: dict[str, Any],
    token_ids: Sequence[int],
) -> torch.Tensor:
    unique = sorted(set(int(token_id) for token_id in token_ids))
    fetched_rows = tensor_cache.fetch_rows(
        weight_index=weight_index,
        tensor_name="model.embed_tokens.weight",
        row_indices=unique,
    )
    lookup = {token_id: fetched_rows[index] for index, token_id in enumerate(unique)}
    return torch.stack([lookup[int(token_id)] for token_id in token_ids], dim=0).to(torch.bfloat16)


def _fetch_named_tensors_exact(
    *,
    tensor_cache: PersistentTensorRangeCache,
    weight_index: dict[str, Any],
    tensor_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for tensor_name in tensor_names:
        tensors[tensor_name] = tensor_cache.fetch_tensor(weight_index=weight_index, tensor_name=tensor_name)
    return tensors


def _load_decoder_layer_exact(
    *,
    config: Any,
    tensor_cache: PersistentTensorRangeCache,
    weight_index: dict[str, Any],
    layer_index: int,
) -> LlamaDecoderLayer:
    prefix = f"model.layers.{layer_index}."
    tensor_names = [
        prefix + "self_attn.q_proj.weight",
        prefix + "self_attn.k_proj.weight",
        prefix + "self_attn.v_proj.weight",
        prefix + "self_attn.o_proj.weight",
        prefix + "mlp.gate_proj.weight",
        prefix + "mlp.up_proj.weight",
        prefix + "mlp.down_proj.weight",
        prefix + "input_layernorm.weight",
        prefix + "post_attention_layernorm.weight",
    ]
    fetched = _fetch_named_tensors_exact(
        tensor_cache=tensor_cache,
        weight_index=weight_index,
        tensor_names=tensor_names,
    )
    state_dict = {name[len(prefix) :]: tensor for name, tensor in fetched.items()}
    return _load_decoder_layer_from_state_dict(config, layer_index=layer_index, state_dict=state_dict)


def _load_rms_norm_weight(
    *,
    tensor_cache: PersistentTensorRangeCache,
    weight_index: dict[str, Any],
) -> torch.Tensor:
    return tensor_cache.fetch_tensor(weight_index=weight_index, tensor_name="model.norm.weight").to(torch.bfloat16)


def _rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    with torch.inference_mode():
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        normed = hidden_states.to(torch.float32) * torch.rsqrt(variance + float(eps))
        return (normed.to(weight.dtype) * weight.to(weight.dtype)).contiguous()


def _lm_head_logits_chunked(
    *,
    hidden_states: torch.Tensor,
    tensor_cache: PersistentTensorRangeCache,
    weight_index: dict[str, Any],
    chunk_rows: int = 2048,
) -> torch.Tensor:
    with torch.inference_mode():
        shard = tensor_cache._shard(weight_index["weight_map"]["lm_head.weight"])
        spec = shard.tensor_spec("lm_head.weight")
        vocab_size, hidden_size = spec.shape
        if int(hidden_states.shape[1]) != hidden_size:
            raise Stage5ExecutionError(
                f"Hidden-state width {hidden_states.shape[1]} does not match lm_head width {hidden_size}."
            )
        chunks: list[torch.Tensor] = []
        hidden_fp32 = hidden_states.to(torch.float32)
        for row_start in range(0, vocab_size, chunk_rows):
            row_end = min(row_start + chunk_rows, vocab_size)
            weight_chunk = tensor_cache.fetch_row_range(
                weight_index=weight_index,
                tensor_name="lm_head.weight",
                row_start=row_start,
                row_end_exclusive=row_end,
            ).to(torch.float32)
            chunks.append(torch.matmul(hidden_fp32, weight_chunk.transpose(0, 1)).cpu())
            del weight_chunk
            gc.collect()
        result = torch.cat(chunks, dim=1).contiguous()
        del chunks, hidden_fp32
        gc.collect()
        return result


def _stock_layer_forward_with_intermediates(
    *,
    layer: LlamaDecoderLayer,
    config: Any,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        holders: dict[str, torch.Tensor] = {}

        def _hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            holders["attention_output"] = inputs[0].detach().cpu().squeeze(0).contiguous()
            holders["projected"] = output.detach().cpu().squeeze(0).contiguous()

        handle = layer.self_attn.o_proj.register_forward_hook(_hook)
        try:
            from .llama_trace import _run_stock_decoder_layer_full_sequence

            decoder_output = _run_stock_decoder_layer_full_sequence(
                config=config,
                layer=layer,
                input_hidden_states=hidden_states,
            )
        finally:
            handle.remove()
        attention_output = holders["attention_output"].to(hidden_states.dtype)
        projected = holders["projected"].to(hidden_states.dtype)
        post_attention_residual = hidden_states + projected
        mlp_output = (decoder_output - post_attention_residual).detach().cpu().contiguous()
        return (
            attention_output.cpu(),
            projected.cpu(),
            post_attention_residual.cpu(),
            mlp_output.cpu(),
            decoder_output.cpu(),
        )


def _precompute_qkv(
    *,
    layer: LlamaDecoderLayer,
    config: Any,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotary_emb = LlamaRotaryEmbedding(config)
    with torch.no_grad():
        batch_hidden = hidden_states.unsqueeze(0).to(torch.bfloat16)
        input_shape = batch_hidden.shape[:-1]
        head_dim = layer.self_attn.head_dim
        query_states = layer.self_attn.q_proj(batch_hidden).view(*input_shape, -1, head_dim).transpose(1, 2)
        key_states = layer.self_attn.k_proj(batch_hidden).view(*input_shape, -1, head_dim).transpose(1, 2)
        value_states = layer.self_attn.v_proj(batch_hidden).view(*input_shape, -1, head_dim).transpose(1, 2)
        position_ids = torch.arange(hidden_states.shape[0], dtype=torch.long).unsqueeze(0)
        cos, sin = rotary_emb(batch_hidden, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    return (
        query_states[0].transpose(0, 1).detach().cpu().contiguous(),
        key_states[0].transpose(0, 1).detach().cpu().contiguous(),
        value_states[0].transpose(0, 1).detach().cpu().contiguous(),
    )


def _precompute_qkv_from_weights(
    *,
    config: Any,
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotary_emb = LlamaRotaryEmbedding(config)
    num_attention_heads = int(config.num_attention_heads)
    num_key_value_heads = int(config.num_key_value_heads)
    head_dim = int(config.hidden_size // config.num_attention_heads)
    with torch.inference_mode():
        batch_hidden = hidden_states.unsqueeze(0).to(torch.bfloat16)
        input_shape = batch_hidden.shape[:-1]
        query_states = F.linear(batch_hidden, q_proj_weight.to(torch.bfloat16)).view(
            *input_shape,
            num_attention_heads,
            head_dim,
        ).transpose(1, 2)
        key_states = F.linear(batch_hidden, k_proj_weight.to(torch.bfloat16)).view(
            *input_shape,
            num_key_value_heads,
            head_dim,
        ).transpose(1, 2)
        value_states = F.linear(batch_hidden, v_proj_weight.to(torch.bfloat16)).view(
            *input_shape,
            num_key_value_heads,
            head_dim,
        ).transpose(1, 2)
        position_ids = torch.arange(hidden_states.shape[0], dtype=torch.long).unsqueeze(0)
        cos, sin = rotary_emb(batch_hidden, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    return (
        query_states[0].transpose(0, 1).detach().cpu().contiguous(),
        key_states[0].transpose(0, 1).detach().cpu().contiguous(),
        value_states[0].transpose(0, 1).detach().cpu().contiguous(),
    )


def _prefix_context(
    *,
    prefix_keys_tensor: torch.Tensor,
    prefix_values_tensor: torch.Tensor,
    record_index: int,
    kv_head_global: int,
    recent_window: int,
    include_float64_arrays: bool = True,
) -> _PrefixSerializationContext:
    visible_length = int(prefix_keys_tensor.shape[0])
    recent_exact_tokens = min(recent_window, visible_length)
    historical_tokens = max(visible_length - recent_exact_tokens, 0)
    return _PrefixSerializationContext(
        record_index=record_index,
        kv_head_global=kv_head_global,
        visible_length=visible_length,
        historical_tokens=historical_tokens,
        recent_exact_tokens=recent_exact_tokens,
        head_dim=int(prefix_keys_tensor.shape[1]),
        prefix_keys_tensor=prefix_keys_tensor,
        prefix_values_tensor=prefix_values_tensor,
        prefix_keys=_float64_numpy(prefix_keys_tensor) if include_float64_arrays else None,
        prefix_values=_float64_numpy(prefix_values_tensor) if include_float64_arrays else None,
        full_kv_reference_bytes=_tensor_bytes(prefix_keys_tensor) + _tensor_bytes(prefix_values_tensor),
    )


def _manual_attention(
    *,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    with torch.inference_mode():
        scores = torch.matmul(queries.to(torch.float32), keys.to(torch.float32).transpose(0, 1)) * float(scaling)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, values.to(torch.float32)).to(torch.bfloat16)


def _float64_attention_output_numpy(
    *,
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    scaling: float,
) -> np.ndarray:
    query64 = np.asarray(query, dtype=np.float64)
    keys64 = np.asarray(keys, dtype=np.float64)
    values64 = np.asarray(values, dtype=np.float64)
    scores = keys64 @ query64 * float(scaling)
    if scores.size == 0:
        return np.zeros((values64.shape[1],), dtype=np.float64)
    shift = float(np.max(scores))
    weights = np.exp(scores - shift, dtype=np.float64)
    weights_sum = float(np.sum(weights, dtype=np.float64))
    if weights_sum <= 0.0 or not np.isfinite(weights_sum):
        raise Stage5ExecutionError("Float64 attention output encountered a nonpositive or nonfinite denominator.")
    return (weights / weights_sum) @ values64


def _shared_reconstructed_attention_output_numpy(
    *,
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    scaling: float,
) -> np.ndarray:
    return _float64_attention_output_numpy(
        query=query,
        keys=keys,
        values=values,
        scaling=scaling,
    )


def _build_stage5_rack_prefix_payload(
    *,
    prefix_context: _PrefixSerializationContext,
    recent_window: int,
    block_size: int,
    precision: int,
    retain_blocks: bool,
) -> _Stage5RackPrefixPayload:
    historical_tokens = int(prefix_context.historical_tokens)
    head_dim = int(prefix_context.head_dim)
    value_dim = int(prefix_context.prefix_values_tensor.shape[1])

    recent_keys_t = prefix_context.prefix_keys_tensor[historical_tokens:, :]
    recent_values_t = prefix_context.prefix_values_tensor[historical_tokens:, :]
    recent_keys = _float64_numpy(recent_keys_t)
    recent_values = _float64_numpy(recent_values_t)

    encoded_blocks: list[CompressedBlock] = []
    for block_start, block_end in _block_slices(historical_tokens, block_size):
        block_keys = _float64_numpy(prefix_context.prefix_keys_tensor[block_start:block_end, :])
        block_values = _float64_numpy(prefix_context.prefix_values_tensor[block_start:block_end, :])
        encoded_blocks.append(encode_block(block_keys, block_values, block_start=block_start, precision=precision))
        del block_keys, block_values

    blocks: list[CompressedBlock] = []
    reconstructed_historical_keys: list[np.ndarray] = []
    reconstructed_historical_values: list[np.ndarray] = []
    reconstructed_block_count = 0
    reconstructed_block_bytes = 0
    key_anchor_bytes = 0
    value_anchor_bytes = 0
    quantized_key_residual_bytes = 0
    quantized_value_residual_bytes = 0
    scale_bytes = 0
    certificate_metadata_bytes = 0
    block_header_bytes = 0
    container_header_bytes = 0
    container_index_bytes = 0
    padding_alignment_bytes = 0
    container_bytes = b""

    if encoded_blocks:
        emitted_container = serialize_block_container(encoded_blocks)
        authoritative_container = deserialize_block_container(emitted_container.buffer)
        container_bytes = authoritative_container.buffer
        block_count = authoritative_container.block_count
        key_anchor_bytes = block_count * 2 * head_dim
        value_anchor_bytes = block_count * 2 * value_dim
        quantized_key_residual_bytes = sum(max(block.block_len - 1, 0) * head_dim for block in encoded_blocks)
        quantized_value_residual_bytes = sum(max(block.block_len - 1, 0) * value_dim for block in encoded_blocks)
        scale_bytes = block_count * 4
        certificate_metadata_bytes = block_count * 8
        block_header_bytes = block_count * 8
        container_header_bytes = authoritative_container.header_bytes
        container_index_bytes = authoritative_container.index_bytes
        for block_index in range(authoritative_container.block_count):
            block = authoritative_container.deserialize_block(block_index)
            block_keys, block_values = block.decode_block()
            reconstructed_historical_keys.append(block_keys)
            reconstructed_historical_values.append(block_values)
            reconstructed_block_count += 1
            reconstructed_block_bytes += int(block_keys.nbytes + block_values.nbytes)
            if retain_blocks:
                blocks.append(block)
            del block_keys, block_values
        del emitted_container, authoritative_container

    reconstructed_keys = (
        np.vstack([*(reconstructed_historical_keys or []), recent_keys])
        if prefix_context.visible_length
        else np.zeros((0, head_dim), dtype=np.float64)
    )
    reconstructed_values = (
        np.vstack([*(reconstructed_historical_values or []), recent_values])
        if prefix_context.visible_length
        else np.zeros((0, value_dim), dtype=np.float64)
    )

    recent_key_payload = _bf16_tensor_to_bytes(recent_keys_t)
    recent_value_payload = _bf16_tensor_to_bytes(recent_values_t)
    payload, wrapper_header_bytes = _section_header_payload(
        method_name="rack_kv",
        parameters={
            "mode": "native",
            "recent_window": recent_window,
            "block_size": block_size,
            "precision": precision,
            "historical_tokens": historical_tokens,
            "authoritative_serialized_roundtrip_used": True,
            "certificate_mode": "rigorous_reference",
        },
        sections=(
            ("recent_keys_bf16", recent_key_payload, {"encoding": "bf16_tensor", "shape": list(recent_keys_t.shape)}),
            ("recent_values_bf16", recent_value_payload, {"encoding": "bf16_tensor", "shape": list(recent_values_t.shape)}),
            (
                "historical_block_container",
                container_bytes,
                {"encoding": "serialized_block_container", "length": len(container_bytes)},
            ),
        ),
    )
    byte_breakdown = ByteBreakdown(
        encoded_key_bytes=key_anchor_bytes + quantized_key_residual_bytes,
        encoded_value_bytes=value_anchor_bytes + quantized_value_residual_bytes,
        scales_bytes=scale_bytes,
        metadata_bytes=wrapper_header_bytes + container_header_bytes,
        indices_bytes=container_index_bytes,
        block_page_metadata_bytes=certificate_metadata_bytes + block_header_bytes + padding_alignment_bytes,
        recent_window_bytes=len(recent_key_payload) + len(recent_value_payload),
        total_serialized_bytes=len(payload),
    )
    diagnostics = {
        "historical_tokens": historical_tokens,
        "recent_exact_tokens": int(prefix_context.recent_exact_tokens),
        "reconstructed_block_count": reconstructed_block_count,
        "reconstructed_block_bytes": reconstructed_block_bytes,
        "serialization_buffer_bytes": len(payload),
        "historical_container_bytes": len(container_bytes),
        "recent_payload_bytes": len(recent_key_payload) + len(recent_value_payload),
    }
    del encoded_blocks, reconstructed_historical_keys, reconstructed_historical_values
    del recent_key_payload, recent_value_payload, payload
    _release_memory()
    return _Stage5RackPrefixPayload(
        blocks=tuple(blocks),
        reconstructed_keys=reconstructed_keys,
        reconstructed_values=reconstructed_values,
        recent_keys=recent_keys if retain_blocks else None,
        recent_values=recent_values if retain_blocks else None,
        byte_breakdown=byte_breakdown,
        diagnostics=diagnostics,
    )


def _kept_from_stage5_blocks(
    *,
    blocks: Sequence[CompressedBlock],
    decoded_block_starts: Sequence[int],
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    decoded = set(int(value) for value in decoded_block_starts)
    kept_keys: list[np.ndarray] = []
    kept_values: list[np.ndarray] = []
    for block in blocks:
        if int(block.header.block_start) in decoded:
            block_keys, block_values = block.decode_block()
            kept_keys.append(block_keys)
            kept_values.append(block_values)
    kept_keys.append(recent_keys)
    kept_values.append(recent_values)
    return np.vstack(kept_keys), np.vstack(kept_values)


def _new_incremental_rack_state(
    *,
    recent_window: int,
    block_size: int,
    precision: int,
    head_dim: int,
    value_dim: int,
) -> _Stage5IncrementalRackState:
    return _Stage5IncrementalRackState(
        recent_window=recent_window,
        block_size=block_size,
        precision=precision,
        head_dim=head_dim,
        value_dim=value_dim,
        finalized_blocks=[],
        finalized_block_payload_lengths=[],
        finalized_key_blocks=[],
        finalized_value_blocks=[],
        recent_keys=[],
        recent_values=[],
        tail_keys=[],
        tail_values=[],
    )


def _stage5_wrapper_total_bytes(
    *,
    recent_keys_shape: Sequence[int],
    recent_values_shape: Sequence[int],
    recent_keys_bytes: int,
    recent_values_bytes: int,
    historical_container_bytes: int,
    recent_window: int,
    block_size: int,
    precision: int,
    historical_tokens: int,
) -> tuple[int, int]:
    section_descriptors = [
        {
            "name": "recent_keys_bf16",
            "offset": 0,
            "length": recent_keys_bytes,
            "encoding": "bf16_tensor",
            "shape": list(recent_keys_shape),
        },
        {
            "name": "recent_values_bf16",
            "offset": recent_keys_bytes,
            "length": recent_values_bytes,
            "encoding": "bf16_tensor",
            "shape": list(recent_values_shape),
        },
        {
            "name": "historical_block_container",
            "offset": recent_keys_bytes + recent_values_bytes,
            "length": historical_container_bytes,
            "encoding": "serialized_block_container",
            "length": historical_container_bytes,
        },
    ]
    header = {
        "format": "rack_kv_stage4_payload_v1",
        "method_name": "rack_kv",
        "method_version": "stage4_baselines_v2",
        "parameters": {
            "mode": "native",
            "recent_window": recent_window,
            "block_size": block_size,
            "precision": precision,
            "historical_tokens": historical_tokens,
            "authoritative_serialized_roundtrip_used": True,
            "certificate_mode": "rigorous_reference",
        },
        "sections": section_descriptors,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    wrapper_header_bytes = 4 + len(header_bytes)
    total_payload_bytes = recent_keys_bytes + recent_values_bytes + historical_container_bytes
    return wrapper_header_bytes + total_payload_bytes, wrapper_header_bytes


def _advance_incremental_rack_state(
    state: _Stage5IncrementalRackState,
    *,
    key_token: np.ndarray,
    value_token: np.ndarray,
) -> None:
    key_copy = np.asarray(key_token, dtype=np.float64).copy()
    value_copy = np.asarray(value_token, dtype=np.float64).copy()
    state.recent_keys.append(key_copy)
    state.recent_values.append(value_copy)
    if len(state.recent_keys) > state.recent_window:
        state.tail_keys.append(state.recent_keys.pop(0))
        state.tail_values.append(state.recent_values.pop(0))
    if len(state.tail_keys) == state.block_size:
        block_start = sum(block.block_len for block in state.finalized_blocks)
        block_keys = np.vstack(state.tail_keys)
        block_values = np.vstack(state.tail_values)
        block = encode_block(block_keys, block_values, block_start=block_start, precision=state.precision)
        block_payload = block.serialize()
        authoritative_block = CompressedBlock.deserialize(block_payload, key_dim=state.head_dim, value_dim=state.value_dim)
        decoded_keys, decoded_values = authoritative_block.decode_block()
        state.finalized_blocks.append(authoritative_block)
        state.finalized_block_payload_lengths.append(len(block_payload))
        state.finalized_key_blocks.append(decoded_keys)
        state.finalized_value_blocks.append(decoded_values)
        state.tail_keys.clear()
        state.tail_values.clear()
        del block_keys, block_values, block, block_payload


def _materialize_incremental_rack_payload(
    state: _Stage5IncrementalRackState,
    *,
    retain_blocks: bool,
) -> _Stage5RackPrefixPayload:
    historical_tokens = sum(block.block_len for block in state.finalized_blocks) + len(state.tail_keys)
    tail_block: CompressedBlock | None = None
    tail_block_payload_length = 0
    tail_keys = np.zeros((0, state.head_dim), dtype=np.float64)
    tail_values = np.zeros((0, state.value_dim), dtype=np.float64)
    if state.tail_keys:
        block_start = sum(block.block_len for block in state.finalized_blocks)
        provisional_keys = np.vstack(state.tail_keys)
        provisional_values = np.vstack(state.tail_values)
        block = encode_block(provisional_keys, provisional_values, block_start=block_start, precision=state.precision)
        block_payload = block.serialize()
        tail_block = CompressedBlock.deserialize(block_payload, key_dim=state.head_dim, value_dim=state.value_dim)
        tail_block_payload_length = len(block_payload)
        tail_keys, tail_values = tail_block.decode_block()
        del provisional_keys, provisional_values, block, block_payload

    recent_keys = (
        np.vstack(state.recent_keys)
        if state.recent_keys
        else np.zeros((0, state.head_dim), dtype=np.float64)
    )
    recent_values = (
        np.vstack(state.recent_values)
        if state.recent_values
        else np.zeros((0, state.value_dim), dtype=np.float64)
    )
    all_blocks: list[CompressedBlock] = list(state.finalized_blocks)
    if tail_block is not None:
        all_blocks.append(tail_block)
    reconstructed_historical_keys = list(state.finalized_key_blocks)
    reconstructed_historical_values = list(state.finalized_value_blocks)
    if tail_block is not None:
        reconstructed_historical_keys.append(tail_keys)
        reconstructed_historical_values.append(tail_values)
    reconstructed_keys = (
        np.vstack([*(reconstructed_historical_keys or []), recent_keys])
        if historical_tokens + len(state.recent_keys) > 0
        else np.zeros((0, state.head_dim), dtype=np.float64)
    )
    reconstructed_values = (
        np.vstack([*(reconstructed_historical_values or []), recent_values])
        if historical_tokens + len(state.recent_values) > 0
        else np.zeros((0, state.value_dim), dtype=np.float64)
    )

    key_anchor_bytes = 0
    value_anchor_bytes = 0
    quantized_key_residual_bytes = 0
    quantized_value_residual_bytes = 0
    scale_bytes = 0
    certificate_metadata_bytes = 0
    block_header_bytes = 0
    container_header_bytes = 0
    container_index_bytes = 0
    historical_container_bytes = 0
    if all_blocks:
        block_count = len(all_blocks)
        key_anchor_bytes = block_count * 2 * state.head_dim
        value_anchor_bytes = block_count * 2 * state.value_dim
        quantized_key_residual_bytes = sum(max(block.block_len - 1, 0) * state.head_dim for block in all_blocks)
        quantized_value_residual_bytes = sum(max(block.block_len - 1, 0) * state.value_dim for block in all_blocks)
        scale_bytes = block_count * 4
        certificate_metadata_bytes = block_count * 8
        block_header_bytes = block_count * 8
        container_header_bytes = 16
        container_index_bytes = block_count * 8
        historical_container_bytes = container_header_bytes + container_index_bytes + sum(state.finalized_block_payload_lengths) + tail_block_payload_length

    recent_keys_t = torch.from_numpy(recent_keys.astype(np.float32, copy=False)).to(torch.bfloat16)
    recent_values_t = torch.from_numpy(recent_values.astype(np.float32, copy=False)).to(torch.bfloat16)
    recent_keys_bytes = int(recent_keys_t.numel()) * int(recent_keys_t.element_size())
    recent_values_bytes = int(recent_values_t.numel()) * int(recent_values_t.element_size())
    total_serialized_bytes, wrapper_header_bytes = _stage5_wrapper_total_bytes(
        recent_keys_shape=recent_keys_t.shape,
        recent_values_shape=recent_values_t.shape,
        recent_keys_bytes=recent_keys_bytes,
        recent_values_bytes=recent_values_bytes,
        historical_container_bytes=historical_container_bytes,
        recent_window=state.recent_window,
        block_size=state.block_size,
        precision=state.precision,
        historical_tokens=historical_tokens,
    )
    byte_breakdown = ByteBreakdown(
        encoded_key_bytes=key_anchor_bytes + quantized_key_residual_bytes,
        encoded_value_bytes=value_anchor_bytes + quantized_value_residual_bytes,
        scales_bytes=scale_bytes,
        metadata_bytes=wrapper_header_bytes + container_header_bytes,
        indices_bytes=container_index_bytes,
        block_page_metadata_bytes=certificate_metadata_bytes + block_header_bytes,
        recent_window_bytes=recent_keys_bytes + recent_values_bytes,
        total_serialized_bytes=total_serialized_bytes,
    )
    diagnostics = {
        "historical_tokens": historical_tokens,
        "recent_exact_tokens": len(state.recent_keys),
        "reconstructed_block_count": len(all_blocks),
        "reconstructed_block_bytes": int(sum(block.nbytes for block in reconstructed_historical_keys) + sum(block.nbytes for block in reconstructed_historical_values)),
        "serialization_buffer_bytes": total_serialized_bytes,
        "historical_container_bytes": historical_container_bytes,
        "recent_payload_bytes": recent_keys_bytes + recent_values_bytes,
    }
    del recent_keys_t, recent_values_t
    _release_memory()
    return _Stage5RackPrefixPayload(
        blocks=tuple(all_blocks) if retain_blocks else tuple(),
        reconstructed_keys=reconstructed_keys,
        reconstructed_values=reconstructed_values,
        recent_keys=recent_keys if retain_blocks else None,
        recent_values=recent_values if retain_blocks else None,
        byte_breakdown=byte_breakdown,
        diagnostics=diagnostics,
    )


def _prepare_certification_inputs_from_incremental_state(
    state: _Stage5IncrementalRackState,
    rack_payload: _Stage5RackPrefixPayload,
) -> Any:
    recent_keys = (
        rack_payload.recent_keys
        if rack_payload.recent_keys is not None
        else np.zeros((0, state.head_dim), dtype=np.float64)
    )
    recent_values = (
        rack_payload.recent_values
        if rack_payload.recent_values is not None
        else np.zeros((0, state.value_dim), dtype=np.float64)
    )
    return prepare_certification_inputs(
        recent_keys,
        recent_values,
        rack_payload.blocks,
        precision=state.precision,
    )


def _fast_prefilter_bound(
    *,
    query: np.ndarray,
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
    prefix_result: _Stage5RackPrefixPayload,
    scaling: float,
) -> float:
    if not prefix_result.blocks:
        return 0.0
    query64 = np.asarray(query, dtype=np.float64)
    q_norm = float(np.linalg.norm(query64, ord=2))
    recent_scores = recent_keys @ query64 * float(scaling) if recent_keys.size else np.zeros((0,), dtype=np.float64)
    if recent_scores.size == 0:
        return math.inf
    shift = max(float(np.max(recent_scores)), max(
        float(
            (query64 @ block.header.anchor_key.astype(np.float64) + q_norm * float(block.header.rho_upper)) * float(scaling)
        )
        for block in prefix_result.blocks
    ))
    z_k = float(np.sum(np.exp(recent_scores - shift, dtype=np.float64), dtype=np.float64))
    if z_k <= 0.0 or not np.isfinite(z_k):
        return math.inf
    recent_value_norms = np.linalg.norm(recent_values.astype(np.float64), axis=1) if recent_values.size else np.zeros((0,), dtype=np.float64)
    recent_scores_upper = np.exp(recent_scores - shift, dtype=np.float64)
    b_k = float(np.sum(recent_scores_upper * recent_value_norms, dtype=np.float64) / z_k) if recent_scores_upper.size else 0.0
    masses = []
    nus = []
    for block in prefix_result.blocks:
        cap = float(
            (query64 @ block.header.anchor_key.astype(np.float64) + q_norm * float(block.header.rho_upper)) * float(scaling)
        )
        masses.append(float(block.block_len) * float(np.exp(cap - shift)))
        nus.append(float(block.header.nu_upper))
    u_s = float(np.sum(masses, dtype=np.float64))
    nu_s = float(max(nus, default=0.0))
    if not np.isfinite(u_s) or not np.isfinite(nu_s):
        return math.inf
    return float((u_s / (z_k + u_s)) * (nu_s + b_k))


def _certified_attention_for_head(
    *,
    query: np.ndarray,
    reconstructed_full_keys: np.ndarray,
    reconstructed_full_values: np.ndarray,
    prefix_result: _Stage5RackPrefixPayload,
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
    tolerance: float,
    precision: int,
    scaling: float,
    prepared_inputs: Any | None = None,
    prepared_full_rows: tuple[Sequence[Sequence[gmpy2.mpfr]], Sequence[Sequence[gmpy2.mpfr]]] | None = None,
    prefilter_bound_override: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not prefix_result.blocks:
        full_output = _shared_reconstructed_attention_output_numpy(
            query=query,
            keys=reconstructed_full_keys,
            values=reconstructed_full_values,
            scaling=scaling,
        )
        return full_output, {}

    prefilter_bound = (
        float(prefilter_bound_override)
        if prefilter_bound_override is not None
        else _fast_prefilter_bound(
            query=query,
            recent_keys=recent_keys,
            recent_values=recent_values,
            prefix_result=prefix_result,
            scaling=scaling,
        )
    )
    record: dict[str, Any] = {
        "schema": STAGE5_CERTIFICATE_RECORD_SCHEMA,
        "prefilter_bound": prefilter_bound,
        "eligible_blocks": len(prefix_result.blocks),
        "mpfr_invoked": False,
        "prefilter_rejected_decisions": 0,
        "mpfr_certified_skipped_blocks": 0,
        "mpfr_rejected_candidates": 0,
        "fast_prefilter_rejected_blocks": 0,
        "potential_skip_candidates_sent_to_mpfr": 0,
        "mpfr_rejected_skip_candidates": 0,
        "certificate_bound_text": None,
        "certificate_bound_upper_float": None,
        "observed_local_skipping_error": 0.0,
        "rigorous_local_skipping_upper": 0.0,
        "rigorous_interval_violation": 0,
        "approximate_observed_violation": 0,
        "false_safe_count": 0,
        "numerical_fallback_used": 0,
        "decoded_block_starts": [],
        "skipped_block_starts": [],
        "chosen_certificate": None,
    }
    if prefilter_bound > tolerance:
        full_output = _shared_reconstructed_attention_output_numpy(
            query=query,
            keys=reconstructed_full_keys,
            values=reconstructed_full_values,
            scaling=scaling,
        )
        record["prefilter_rejected_decisions"] = len(prefix_result.blocks)
        record["fast_prefilter_rejected_blocks"] = len(prefix_result.blocks)
        record["potential_skip_candidates_sent_to_mpfr"] = 0
        return full_output, record

    record["potential_skip_candidates_sent_to_mpfr"] = len(prefix_result.blocks)
    certification = (
        certify_progressive_skipping_prepared(
            query=query,
            prepared_inputs=prepared_inputs,
            tolerance=tolerance,
            precision=precision,
            attention_scale=scaling,
        )
        if prepared_inputs is not None
        else certify_progressive_skipping(
            query=query,
            recent_keys=recent_keys,
            recent_values=recent_values,
            historical_blocks=prefix_result.blocks,
            tolerance=tolerance,
            precision=precision,
            attention_scale=scaling,
        )
    )
    record["mpfr_invoked"] = True
    record["chosen_certificate"] = certification.chosen_certificate
    record["certificate_bound_text"] = certification.certificate_value_text
    record["certificate_bound_upper_float"] = float(certification.certificate_value_upper_float or 0.0)
    record["decoded_block_starts"] = list(certification.decoded_block_starts)
    record["skipped_block_starts"] = list(certification.skipped_block_starts)
    record["mpfr_certified_skipped_blocks"] = len(certification.skipped_block_starts)
    record["mpfr_rejected_candidates"] = len(certification.decoded_block_starts)
    record["mpfr_rejected_skip_candidates"] = len(certification.decoded_block_starts)
    record["numerical_fallback_used"] = int(certification.numerical_fallback_used)

    if not certification.skipped_block_starts:
        full_output = _shared_reconstructed_attention_output_numpy(
            query=query,
            keys=reconstructed_full_keys,
            values=reconstructed_full_values,
            scaling=scaling,
        )
        return full_output, record

    kept_keys, kept_values = _kept_from_stage5_blocks(
        blocks=prefix_result.blocks,
        decoded_block_starts=certification.decoded_block_starts,
        recent_keys=recent_keys,
        recent_values=recent_values,
    )
    kept_output = _shared_reconstructed_attention_output_numpy(
        query=query,
        keys=kept_keys,
        values=kept_values,
        scaling=scaling,
    )
    query_exact = exact_vector(np.asarray(query, dtype=np.float64), precision=precision)
    if prepared_inputs is not None:
        if prepared_full_rows is None:
            prepared_full_rows = prepared_full_exact_rows(prepared_inputs)
        kept_rows = prepared_kept_exact_rows(
            prepared_inputs,
            decoded_block_starts=certification.decoded_block_starts,
        )
        kept_output_mpfr = tuple(
            exact_reference_output_from_exact_rows(
                query_exact,
                kept_rows[0],
                kept_rows[1],
                value_dim=prepared_inputs.value_dim,
                precision=precision,
                attention_scale=scaling,
            )
        )
        full_output_mpfr = tuple(
            exact_reference_output_from_exact_rows(
                query_exact,
                prepared_full_rows[0],
                prepared_full_rows[1],
                value_dim=prepared_inputs.value_dim,
                precision=precision,
                attention_scale=scaling,
            )
        )
    else:
        kept_output_mpfr = tuple(
            exact_reference_output_mpfr(
                query,
                kept_keys,
                kept_values,
                precision=precision,
                attention_scale=scaling,
            )
        )
        full_output_mpfr = tuple(
            exact_reference_output_mpfr(
                query,
                reconstructed_full_keys,
                reconstructed_full_values,
                precision=precision,
                attention_scale=scaling,
            )
        )
    observed_skip_error_mpfr = rigorous_output_error_norm(
        full_output_mpfr,
        kept_output_mpfr,
        precision=precision,
    )
    if prepared_inputs is not None:
        if prepared_full_rows is None:
            prepared_full_rows = prepared_full_exact_rows(prepared_inputs)
        full_interval, _ = rigorous_attention_output_interval_from_exact_rows(
            query_exact,
            prepared_full_rows[0],
            prepared_full_rows[1],
            value_dim=prepared_inputs.value_dim,
            precision=precision,
            attention_scale=scaling,
        )
        kept_interval, _ = rigorous_attention_output_interval_from_exact_rows(
            query_exact,
            kept_rows[0],
            kept_rows[1],
            value_dim=prepared_inputs.value_dim,
            precision=precision,
            attention_scale=scaling,
        )
    else:
        full_interval, _ = rigorous_attention_output_interval(
            query,
            reconstructed_full_keys,
            reconstructed_full_values,
            precision=precision,
            attention_scale=scaling,
        )
        kept_interval, _ = rigorous_attention_output_interval(
            query,
            kept_keys,
            kept_values,
            precision=precision,
            attention_scale=scaling,
        )
    rigorous_skip_upper = rigorous_output_error_upper_from_intervals(
        full_interval,
        kept_interval,
        precision=precision,
    )
    approximate_violation, rigorous_violation = _skip_validation_flags(
        certificate_value=certification.certificate_value_mpfr,
        observed_skip_error_mpfr=observed_skip_error_mpfr,
        rigorous_skip_error_upper=rigorous_skip_upper,
    )
    if rigorous_violation:
        raise AssertionError("Stage 5 rigorous skipped-output interval upper bound exceeds the rigorous certificate.")
    record["observed_local_skipping_error"] = float(observed_skip_error_mpfr)
    record["rigorous_local_skipping_upper"] = float(rigorous_skip_upper)
    record["rigorous_interval_violation"] = int(rigorous_violation)
    record["approximate_observed_violation"] = int(approximate_violation)
    record["false_safe_count"] = int(rigorous_violation)
    if prepared_inputs is not None:
        del kept_rows
    del kept_keys, kept_values
    del query_exact, kept_output_mpfr, full_output_mpfr, full_interval, kept_interval, rigorous_skip_upper, certification
    _release_memory()
    return kept_output, record


def _compute_method_prefix_payload(
    *,
    method_name: str,
    prefix_context: _PrefixSerializationContext,
    recent_window: int,
    block_size: int,
    precision: int,
    query_heads_group: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, ByteBreakdown, dict[str, Any], np.ndarray | None, np.ndarray | None]:
    if method_name == "full_kv":
        key_bytes = _tensor_bytes(prefix_context.prefix_keys_tensor)
        value_bytes = _tensor_bytes(prefix_context.prefix_values_tensor)
        if prefix_context.prefix_keys is None or prefix_context.prefix_values is None:
            raise AssertionError("full_kv prefix context requires float64 arrays.")
        return (
            prefix_context.prefix_values_tensor.new_zeros((0, prefix_context.head_dim)),
            ByteBreakdown(
                encoded_key_bytes=key_bytes,
                encoded_value_bytes=value_bytes,
                scales_bytes=0,
                metadata_bytes=0,
                indices_bytes=0,
                block_page_metadata_bytes=0,
                recent_window_bytes=0,
                total_serialized_bytes=key_bytes + value_bytes,
            ),
            {"kind": "full_kv"},
            prefix_context.prefix_keys,
            prefix_context.prefix_values,
        )
    if method_name == "uniform_int8_kv":
        if prefix_context.prefix_keys is None or prefix_context.prefix_values is None:
            raise AssertionError("uniform_int8_kv prefix context requires float64 arrays.")
        artifact = _serialize_uniform_int8_kv(prefix_context, mode="native")
        decoded = _decode_uniform_int8_kv(artifact, prefix_context)
        return (
            prefix_context.prefix_values_tensor.new_zeros((0, prefix_context.head_dim)),
            artifact.byte_breakdown,
            {
                "kind": "uniform_int8_kv",
                "decoded_state": decoded,
            },
            decoded.attention_keys,
            decoded.attention_values,
        )
    if method_name in {"rack_kv_compression_only", "rack_kv_certified"}:
        rack_payload = _build_stage5_rack_prefix_payload(
            prefix_context=prefix_context,
            recent_window=recent_window,
            block_size=block_size,
            precision=precision,
            retain_blocks=(method_name == "rack_kv_certified"),
        )
        return (
            prefix_context.prefix_values_tensor.new_zeros((0, prefix_context.head_dim)),
            rack_payload.byte_breakdown,
            {
                "kind": "rack_kv",
                "rack_payload": rack_payload,
            },
            rack_payload.reconstructed_keys,
            rack_payload.reconstructed_values,
        )
    raise KeyError(f"Unsupported Stage 5 method: {method_name}")


def _run_modified_layer_stream(
    *,
    method_name: str,
    layer: LlamaDecoderLayer,
    config: Any,
    hidden_states: torch.Tensor,
    layer_index: int,
    recent_window: int,
    block_size: int,
    tolerance: float,
    precision: int,
    prompt_name: str,
    certificate_records: list[dict[str, Any]],
    memory_guard: MemoryGuard | None = None,
    token_chunk_size: int = 8,
    diagnostic_records: list[dict[str, Any]] | None = None,
    use_incremental_rack_prefix: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, Any]], list[dict[str, Any]]]:
    with torch.inference_mode():
        normalized_hidden = layer.input_layernorm(hidden_states.unsqueeze(0))[0].detach().cpu().to(torch.bfloat16).contiguous()
        query_states, key_states, value_states = _precompute_qkv(
            layer=layer,
            config=config,
            hidden_states=normalized_hidden,
        )
        sequence_length, num_attention_heads, head_dim = query_states.shape
        num_key_value_heads = key_states.shape[1]
        groups = _gqa_groups(num_attention_heads, num_key_value_heads)
        projected_outputs = torch.zeros((sequence_length, hidden_states.shape[1]), dtype=torch.bfloat16)
        attention_outputs = torch.zeros((sequence_length, hidden_states.shape[1]), dtype=torch.bfloat16)
        post_attention_residual = torch.zeros_like(projected_outputs)
        mlp_outputs = torch.zeros_like(projected_outputs)
        storage_records: list[dict[str, Any]] = []
        certificate_records_local: list[dict[str, Any]] = []
        incremental_rack_states = (
            {
                kv_head_global: _new_incremental_rack_state(
                    recent_window=recent_window,
                    block_size=block_size,
                    precision=precision,
                    head_dim=head_dim,
                    value_dim=int(value_states.shape[2]),
                )
                for kv_head_global in range(num_key_value_heads)
            }
            if method_name in {"rack_kv_compression_only", "rack_kv_certified"} and use_incremental_rack_prefix
            else None
        )
        token_chunk_size = max(1, min(int(token_chunk_size), sequence_length))
        if diagnostic_records is not None and method_name == "rack_kv_compression_only":
            diagnostic_records.append(
                {
                    "stage": "enter_stream",
                    "prompt_name": prompt_name,
                    "layer_index": layer_index,
                    "method_name": method_name,
                    "rss_bytes": MemoryGuard._rss_bytes(),
                    "available_bytes": MemoryGuard._available_bytes(),
                    "live_tensor_bytes": _tensor_container_bytes((hidden_states, normalized_hidden, query_states, key_states, value_states)),
                    "peak_rss_bytes": memory_guard.peak_rss_bytes if memory_guard is not None else MemoryGuard._rss_bytes(),
                    "min_available_bytes": memory_guard.min_available_bytes if memory_guard is not None else MemoryGuard._available_bytes(),
                    "reconstructed_block_count": 0,
                    "reconstructed_block_bytes": 0,
                    "serialization_buffer_bytes": 0,
                }
            )

        for chunk_start in range(0, sequence_length, token_chunk_size):
            chunk_end = min(chunk_start + token_chunk_size, sequence_length)
            if memory_guard is not None:
                memory_guard.check(f"before_token_chunk_{layer_index}_{prompt_name}_{method_name}_{chunk_start}_{chunk_end}")
            chunk_reconstructed_block_count = 0
            chunk_reconstructed_block_bytes = 0
            chunk_serialization_buffer_bytes = 0
            for token_index in range(chunk_start, chunk_end):
                per_head_outputs = torch.zeros((num_attention_heads, head_dim), dtype=torch.bfloat16)
                for kv_head_global, head_group in enumerate(groups):
                    rack_state = None
                    prefix_keys_tensor = key_states[: token_index + 1, kv_head_global, :]
                    prefix_values_tensor = value_states[: token_index + 1, kv_head_global, :]
                    prefix_context = _prefix_context(
                        prefix_keys_tensor=prefix_keys_tensor,
                        prefix_values_tensor=prefix_values_tensor,
                        record_index=token_index,
                        kv_head_global=kv_head_global,
                        recent_window=recent_window,
                        include_float64_arrays=(
                            method_name in {"full_kv", "uniform_int8_kv"}
                            or (method_name in {"rack_kv_compression_only", "rack_kv_certified"} and not use_incremental_rack_prefix)
                        ),
                    )
                    if incremental_rack_states is not None:
                        rack_state = incremental_rack_states[kv_head_global]
                        _advance_incremental_rack_state(
                            rack_state,
                            key_token=_float64_numpy(key_states[token_index, kv_head_global, :]),
                            value_token=_float64_numpy(value_states[token_index, kv_head_global, :]),
                        )
                        rack_payload = _materialize_incremental_rack_payload(
                            rack_state,
                            retain_blocks=(method_name == "rack_kv_certified"),
                        )
                        _dummy = prefix_context.prefix_values_tensor.new_zeros((0, prefix_context.head_dim))
                        byte_breakdown = rack_payload.byte_breakdown
                        state = {"kind": "rack_kv", "rack_payload": rack_payload}
                        reconstructed_keys = rack_payload.reconstructed_keys
                        reconstructed_values = rack_payload.reconstructed_values
                    else:
                        _dummy, byte_breakdown, state, reconstructed_keys, reconstructed_values = _compute_method_prefix_payload(
                            method_name=method_name,
                            prefix_context=prefix_context,
                            recent_window=recent_window,
                            block_size=block_size,
                            precision=precision,
                            query_heads_group=query_states[token_index, head_group, :],
                            scaling=float(layer.self_attn.scaling),
                        )
                    if reconstructed_keys is None or reconstructed_values is None:
                        raise AssertionError("Stage 5 reconstructed keys/values were not produced.")
                    storage_records.append(
                        {
                            "layer_index": layer_index,
                            "token_index": token_index,
                            "prompt_name": prompt_name,
                            "method_name": method_name,
                            "kv_head_global": kv_head_global,
                            "visible_length": int(prefix_context.visible_length),
                            "historical_tokens": int(prefix_context.historical_tokens),
                            "byte_breakdown": asdict(byte_breakdown),
                        }
                    )

                    rack_payload = state.get("rack_payload")
                    if isinstance(rack_payload, _Stage5RackPrefixPayload):
                        chunk_reconstructed_block_count += int(rack_payload.diagnostics["reconstructed_block_count"])
                        chunk_reconstructed_block_bytes += int(rack_payload.diagnostics["reconstructed_block_bytes"])
                        chunk_serialization_buffer_bytes += int(rack_payload.diagnostics["serialization_buffer_bytes"])

                    if method_name == "rack_kv_certified":
                        if not isinstance(rack_payload, _Stage5RackPrefixPayload):
                            raise AssertionError("Certified Stage 5 path requires a rack payload.")
                        if rack_payload.recent_keys is None or rack_payload.recent_values is None:
                            raise AssertionError("Certified Stage 5 path requires recent exact arrays.")
                        recent_keys = rack_payload.recent_keys
                        recent_values = rack_payload.recent_values
                        prepared_inputs = None
                        if rack_payload.blocks:
                            head_queries: list[tuple[int, np.ndarray, float]] = []
                            requires_mpfr = False
                            for query_head_global in head_group:
                                query = _float64_numpy(query_states[token_index, query_head_global, :])
                                prefilter_bound = _fast_prefilter_bound(
                                    query=query,
                                    recent_keys=recent_keys,
                                    recent_values=recent_values,
                                    prefix_result=rack_payload,
                                    scaling=float(layer.self_attn.scaling),
                                )
                                head_queries.append((query_head_global, query, prefilter_bound))
                                if prefilter_bound <= tolerance:
                                    requires_mpfr = True
                            if requires_mpfr:
                                prepared_inputs = (
                                    _prepare_certification_inputs_from_incremental_state(rack_state, rack_payload)
                                    if rack_state is not None
                                    else prepare_certification_inputs(
                                        recent_keys,
                                        recent_values,
                                        rack_payload.blocks,
                                        precision=precision,
                                    )
                                )
                        else:
                            head_queries = [
                                (query_head_global, _float64_numpy(query_states[token_index, query_head_global, :]), 0.0)
                                for query_head_global in head_group
                            ]
                        for query_head_global, query, prefilter_bound in head_queries:
                            if rack_payload.blocks:
                                output, cert_record = _certified_attention_for_head(
                                    query=query,
                                    reconstructed_full_keys=rack_payload.reconstructed_keys,
                                    reconstructed_full_values=rack_payload.reconstructed_values,
                                    prefix_result=rack_payload,
                                    recent_keys=recent_keys,
                                    recent_values=recent_values,
                                    tolerance=tolerance,
                                    precision=precision,
                                    scaling=float(layer.self_attn.scaling),
                                    prepared_inputs=prepared_inputs,
                                    prepared_full_rows=None,
                                    prefilter_bound_override=prefilter_bound,
                                )
                                cert_record.update(
                                    {
                                        "prompt_name": prompt_name,
                                        "layer_index": layer_index,
                                        "token_index": token_index,
                                        "query_head_global": query_head_global,
                                        "kv_head_global": kv_head_global,
                                        "method_name": method_name,
                                    }
                                )
                                certificate_records_local.append(cert_record)
                                if cert_record["mpfr_invoked"]:
                                    certificate_records.append(cert_record)
                                per_head_outputs[query_head_global] = torch.from_numpy(np.asarray(output, dtype=np.float32)).to(torch.bfloat16)
                                del output, cert_record, query
                                _release_memory()
                            else:
                                query_tensor = query_states[token_index, query_head_global : query_head_global + 1, :]
                                out = _manual_attention(
                                    queries=query_tensor,
                                    keys=torch.from_numpy(rack_payload.reconstructed_keys.astype(np.float32, copy=False)).to(torch.bfloat16),
                                    values=torch.from_numpy(rack_payload.reconstructed_values.astype(np.float32, copy=False)).to(torch.bfloat16),
                                    scaling=float(layer.self_attn.scaling),
                                )
                                per_head_outputs[query_head_global] = out[0]
                                del query_tensor, out, query
                                _release_memory()
                        del recent_keys, recent_values, prepared_inputs, head_queries
                    elif method_name == "rack_kv_compression_only":
                        if not isinstance(rack_payload, _Stage5RackPrefixPayload):
                            raise AssertionError("Compression-only Stage 5 path requires a rack payload.")
                        for query_head_global in head_group:
                            query = _float64_numpy(query_states[token_index, query_head_global, :])
                            out = _shared_reconstructed_attention_output_numpy(
                                query=query,
                                keys=rack_payload.reconstructed_keys,
                                values=rack_payload.reconstructed_values,
                                scaling=float(layer.self_attn.scaling),
                            )
                            per_head_outputs[query_head_global] = torch.from_numpy(np.asarray(out, dtype=np.float32)).to(torch.bfloat16)
                            del out, query
                            _release_memory()
                    else:
                        keys_tensor = torch.from_numpy(reconstructed_keys.astype(np.float32, copy=False)).to(torch.bfloat16)
                        values_tensor = torch.from_numpy(reconstructed_values.astype(np.float32, copy=False)).to(torch.bfloat16)
                        queries_group = query_states[token_index, head_group, :]
                        out = _manual_attention(
                            queries=queries_group,
                            keys=keys_tensor,
                            values=values_tensor,
                            scaling=float(layer.self_attn.scaling),
                        )
                        for local_offset, query_head_global in enumerate(head_group):
                            per_head_outputs[query_head_global] = out[local_offset]
                        del keys_tensor, values_tensor, queries_group, out

                    del prefix_keys_tensor, prefix_values_tensor, prefix_context, byte_breakdown, state, reconstructed_keys, reconstructed_values
                    if "rack_payload" in locals():
                        del rack_payload
                    _release_memory()
                attention_outputs[token_index] = per_head_outputs.reshape(-1).detach().cpu().to(torch.bfloat16)
                del per_head_outputs
                _release_memory()

            _release_memory()
            current_rss = MemoryGuard._rss_bytes()
            current_free = MemoryGuard._available_bytes()
            if memory_guard is not None:
                current_rss, current_free = memory_guard.check(
                    f"after_token_chunk_{layer_index}_{prompt_name}_{method_name}_{chunk_start}_{chunk_end}"
                )
            if diagnostic_records is not None and method_name == "rack_kv_compression_only":
                diagnostic_records.append(
                    {
                        "stage": "after_token_chunk",
                        "prompt_name": prompt_name,
                        "layer_index": layer_index,
                        "method_name": method_name,
                        "token_chunk_start": chunk_start,
                        "token_chunk_end_exclusive": chunk_end,
                        "rss_bytes": current_rss,
                        "available_bytes": current_free,
                        "live_tensor_bytes": _tensor_container_bytes(
                            (hidden_states, normalized_hidden, query_states, key_states, value_states, attention_outputs)
                        ),
                        "peak_rss_bytes": memory_guard.peak_rss_bytes if memory_guard is not None else current_rss,
                        "min_available_bytes": memory_guard.min_available_bytes if memory_guard is not None else current_free,
                        "reconstructed_block_count": chunk_reconstructed_block_count,
                        "reconstructed_block_bytes": chunk_reconstructed_block_bytes,
                        "serialization_buffer_bytes": chunk_serialization_buffer_bytes,
                    }
                )
            _release_memory()
            cleanup_rss = MemoryGuard._rss_bytes()
            cleanup_free = MemoryGuard._available_bytes()
            if memory_guard is not None:
                cleanup_rss, cleanup_free = memory_guard.check(
                    f"after_token_chunk_cleanup_{layer_index}_{prompt_name}_{method_name}_{chunk_start}_{chunk_end}"
                )
            if diagnostic_records is not None and method_name == "rack_kv_compression_only":
                diagnostic_records.append(
                    {
                        "stage": "after_chunk_cleanup",
                        "prompt_name": prompt_name,
                        "layer_index": layer_index,
                        "method_name": method_name,
                        "token_chunk_start": chunk_start,
                        "token_chunk_end_exclusive": chunk_end,
                        "rss_bytes": cleanup_rss,
                        "available_bytes": cleanup_free,
                        "live_tensor_bytes": _tensor_container_bytes(
                            (hidden_states, normalized_hidden, query_states, key_states, value_states, attention_outputs)
                        ),
                        "peak_rss_bytes": memory_guard.peak_rss_bytes if memory_guard is not None else cleanup_rss,
                        "min_available_bytes": memory_guard.min_available_bytes if memory_guard is not None else cleanup_free,
                        "reconstructed_block_count": 0,
                        "reconstructed_block_bytes": 0,
                        "serialization_buffer_bytes": 0,
                    }
                )

        projected_outputs = layer.self_attn.o_proj(attention_outputs.to(torch.bfloat16)).detach().cpu().to(torch.bfloat16).contiguous()
        post_attention_residual = (hidden_states.detach().cpu().to(torch.bfloat16) + projected_outputs).contiguous()
        mlp_inputs = layer.post_attention_layernorm(post_attention_residual)
        mlp_outputs = layer.mlp(mlp_inputs).detach().cpu().to(torch.bfloat16).contiguous()
        decoder_output = (post_attention_residual + mlp_outputs).contiguous()
        del normalized_hidden, query_states, key_states, value_states, mlp_inputs
        _release_memory()
        return (
            decoder_output,
            {
                "attention_output": attention_outputs,
                "projected_output": projected_outputs,
                "post_attention_residual": post_attention_residual,
                "mlp_output": mlp_outputs,
                "decoder_output": decoder_output,
            },
            storage_records,
            certificate_records_local,
        )


def _run_modified_layer_stream_exact_from_cache(
    *,
    method_name: str,
    config: Any,
    tensor_cache: PersistentTensorRangeCache,
    weight_index: dict[str, Any],
    hidden_states: torch.Tensor,
    layer_index: int,
    recent_window: int,
    block_size: int,
    tolerance: float,
    precision: int,
    prompt_name: str,
    certificate_records: list[dict[str, Any]],
    memory_guard: MemoryGuard | None = None,
    token_chunk_size: int = 8,
    diagnostic_records: list[dict[str, Any]] | None = None,
    use_incremental_rack_prefix: bool = True,
    profile_progress_every_tokens: int | None = None,
    cache_summary: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, Any]], list[dict[str, Any]]]:
    with torch.inference_mode():
        prefix = f"model.layers.{layer_index}."
        input_norm_weight = tensor_cache.fetch_tensor(weight_index=weight_index, tensor_name=prefix + "input_layernorm.weight").to(torch.bfloat16)
        normalized_hidden = _rms_norm(hidden_states.to(torch.bfloat16), input_norm_weight, eps=float(config.rms_norm_eps)).detach().cpu().to(torch.bfloat16).contiguous()
        del input_norm_weight
        _release_memory()

        q_proj_weight = tensor_cache.fetch_tensor(weight_index=weight_index, tensor_name=prefix + "self_attn.q_proj.weight")
        k_proj_weight = tensor_cache.fetch_tensor(weight_index=weight_index, tensor_name=prefix + "self_attn.k_proj.weight")
        v_proj_weight = tensor_cache.fetch_tensor(weight_index=weight_index, tensor_name=prefix + "self_attn.v_proj.weight")
        query_states, key_states, value_states = _precompute_qkv_from_weights(
            config=config,
            hidden_states=normalized_hidden,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
        )
        del q_proj_weight, k_proj_weight, v_proj_weight
        _release_memory()

        sequence_length, num_attention_heads, head_dim = query_states.shape
        num_key_value_heads = key_states.shape[1]
        groups = _gqa_groups(num_attention_heads, num_key_value_heads)
        scaling = float(1.0 / math.sqrt(head_dim))
        attention_outputs = torch.zeros((sequence_length, hidden_states.shape[1]), dtype=torch.bfloat16)
        storage_records: list[dict[str, Any]] = []
        certificate_records_local: list[dict[str, Any]] = []
        incremental_rack_states = (
            {
                kv_head_global: _new_incremental_rack_state(
                    recent_window=recent_window,
                    block_size=block_size,
                    precision=precision,
                    head_dim=head_dim,
                    value_dim=int(value_states.shape[2]),
                )
                for kv_head_global in range(num_key_value_heads)
            }
            if method_name in {"rack_kv_compression_only", "rack_kv_certified"} and use_incremental_rack_prefix
            else None
        )
        token_chunk_size = max(1, min(int(token_chunk_size), sequence_length))
        progress_interval = (
            None
            if profile_progress_every_tokens is None
            else max(1, min(int(profile_progress_every_tokens), sequence_length))
        )
        profile_start = time.perf_counter()
        total_mpfr_evaluations = 0
        total_serialization_calls = 0

        if diagnostic_records is not None:
            diagnostic_records.append(
                {
                    "stage": "enter_stream",
                    "prompt_name": prompt_name,
                    "layer_index": layer_index,
                    "method_name": method_name,
                    "rss_bytes": MemoryGuard._rss_bytes(),
                    "available_bytes": MemoryGuard._available_bytes(),
                    "live_tensor_bytes": _tensor_container_bytes((hidden_states, normalized_hidden, query_states, key_states, value_states)),
                    "peak_rss_bytes": memory_guard.peak_rss_bytes if memory_guard is not None else MemoryGuard._rss_bytes(),
                    "min_available_bytes": memory_guard.min_available_bytes if memory_guard is not None else MemoryGuard._available_bytes(),
                    "reconstructed_block_count": 0,
                    "reconstructed_block_bytes": 0,
                    "serialization_buffer_bytes": 0,
                    "serialization_calls": 0,
                    "mpfr_evaluations": 0,
                }
            )

        for chunk_start in range(0, sequence_length, token_chunk_size):
            chunk_end = min(chunk_start + token_chunk_size, sequence_length)
            if memory_guard is not None:
                memory_guard.check(f"before_token_chunk_{layer_index}_{prompt_name}_{method_name}_{chunk_start}_{chunk_end}")
            chunk_reconstructed_block_count = 0
            chunk_reconstructed_block_bytes = 0
            chunk_serialization_buffer_bytes = 0
            chunk_serialization_calls = 0
            chunk_mpfr_evaluations = 0
            for token_index in range(chunk_start, chunk_end):
                per_head_outputs = torch.zeros((num_attention_heads, head_dim), dtype=torch.bfloat16)
                for kv_head_global, head_group in enumerate(groups):
                    rack_state = None
                    prefix_keys_tensor = key_states[: token_index + 1, kv_head_global, :]
                    prefix_values_tensor = value_states[: token_index + 1, kv_head_global, :]
                    prefix_context = _prefix_context(
                        prefix_keys_tensor=prefix_keys_tensor,
                        prefix_values_tensor=prefix_values_tensor,
                        record_index=token_index,
                        kv_head_global=kv_head_global,
                        recent_window=recent_window,
                        include_float64_arrays=(
                            method_name in {"full_kv", "uniform_int8_kv"}
                            or (method_name in {"rack_kv_compression_only", "rack_kv_certified"} and not use_incremental_rack_prefix)
                        ),
                    )
                    if incremental_rack_states is not None:
                        rack_state = incremental_rack_states[kv_head_global]
                        _advance_incremental_rack_state(
                            rack_state,
                            key_token=_float64_numpy(key_states[token_index, kv_head_global, :]),
                            value_token=_float64_numpy(value_states[token_index, kv_head_global, :]),
                        )
                        rack_payload = _materialize_incremental_rack_payload(
                            rack_state,
                            retain_blocks=(method_name == "rack_kv_certified"),
                        )
                        chunk_serialization_calls += 1
                        byte_breakdown = rack_payload.byte_breakdown
                        state = {"kind": "rack_kv", "rack_payload": rack_payload}
                        reconstructed_keys = rack_payload.reconstructed_keys
                        reconstructed_values = rack_payload.reconstructed_values
                    else:
                        _dummy, byte_breakdown, state, reconstructed_keys, reconstructed_values = _compute_method_prefix_payload(
                            method_name=method_name,
                            prefix_context=prefix_context,
                            recent_window=recent_window,
                            block_size=block_size,
                            precision=precision,
                            query_heads_group=query_states[token_index, head_group, :],
                            scaling=scaling,
                        )
                        if method_name != "full_kv":
                            chunk_serialization_calls += 1
                    storage_records.append(
                        {
                            "layer_index": layer_index,
                            "token_index": token_index,
                            "prompt_name": prompt_name,
                            "method_name": method_name,
                            "kv_head_global": kv_head_global,
                            "visible_length": int(prefix_context.visible_length),
                            "historical_tokens": int(prefix_context.historical_tokens),
                            "byte_breakdown": asdict(byte_breakdown),
                        }
                    )

                    rack_payload = state.get("rack_payload")
                    if isinstance(rack_payload, _Stage5RackPrefixPayload):
                        chunk_reconstructed_block_count += int(rack_payload.diagnostics["reconstructed_block_count"])
                        chunk_reconstructed_block_bytes += int(rack_payload.diagnostics["reconstructed_block_bytes"])
                        chunk_serialization_buffer_bytes += int(rack_payload.diagnostics["serialization_buffer_bytes"])

                    if method_name == "rack_kv_certified":
                        if not isinstance(rack_payload, _Stage5RackPrefixPayload):
                            raise AssertionError("Certified Stage 5 path requires a rack payload.")
                        if rack_payload.recent_keys is None or rack_payload.recent_values is None:
                            raise AssertionError("Certified Stage 5 path requires recent exact arrays.")
                        recent_keys = rack_payload.recent_keys
                        recent_values = rack_payload.recent_values
                        prepared_inputs = None
                        if rack_payload.blocks:
                            head_queries: list[tuple[int, np.ndarray, float]] = []
                            requires_mpfr = False
                            for query_head_global in head_group:
                                query = _float64_numpy(query_states[token_index, query_head_global, :])
                                prefilter_bound = _fast_prefilter_bound(
                                    query=query,
                                    recent_keys=recent_keys,
                                    recent_values=recent_values,
                                    prefix_result=rack_payload,
                                    scaling=scaling,
                                )
                                head_queries.append((query_head_global, query, prefilter_bound))
                                if prefilter_bound <= tolerance:
                                    requires_mpfr = True
                            if requires_mpfr:
                                prepared_inputs = (
                                    _prepare_certification_inputs_from_incremental_state(rack_state, rack_payload)
                                    if rack_state is not None
                                    else prepare_certification_inputs(
                                        recent_keys,
                                        recent_values,
                                        rack_payload.blocks,
                                        precision=precision,
                                    )
                                )
                        else:
                            head_queries = [
                                (query_head_global, _float64_numpy(query_states[token_index, query_head_global, :]), 0.0)
                                for query_head_global in head_group
                            ]
                        for query_head_global, query, prefilter_bound in head_queries:
                            if rack_payload.blocks:
                                output, cert_record = _certified_attention_for_head(
                                    query=query,
                                    reconstructed_full_keys=rack_payload.reconstructed_keys,
                                    reconstructed_full_values=rack_payload.reconstructed_values,
                                    prefix_result=rack_payload,
                                    recent_keys=recent_keys,
                                    recent_values=recent_values,
                                    tolerance=tolerance,
                                    precision=precision,
                                    scaling=scaling,
                                    prepared_inputs=prepared_inputs,
                                    prepared_full_rows=None,
                                    prefilter_bound_override=prefilter_bound,
                                )
                                cert_record.update(
                                    {
                                        "prompt_name": prompt_name,
                                        "layer_index": layer_index,
                                        "token_index": token_index,
                                        "query_head_global": query_head_global,
                                        "kv_head_global": kv_head_global,
                                        "method_name": method_name,
                                    }
                                )
                                certificate_records_local.append(cert_record)
                                if cert_record["mpfr_invoked"]:
                                    certificate_records.append(cert_record)
                                    chunk_mpfr_evaluations += 1
                                per_head_outputs[query_head_global] = torch.from_numpy(np.asarray(output, dtype=np.float32)).to(torch.bfloat16)
                                del output, cert_record, query
                                _release_memory()
                            else:
                                query_tensor = query_states[token_index, query_head_global : query_head_global + 1, :]
                                out = _manual_attention(
                                    queries=query_tensor,
                                    keys=torch.from_numpy(rack_payload.reconstructed_keys.astype(np.float32, copy=False)).to(torch.bfloat16),
                                    values=torch.from_numpy(rack_payload.reconstructed_values.astype(np.float32, copy=False)).to(torch.bfloat16),
                                    scaling=scaling,
                                )
                                per_head_outputs[query_head_global] = out[0]
                                del query_tensor, out, query
                                _release_memory()
                        del recent_keys, recent_values, prepared_inputs, head_queries
                    elif method_name == "rack_kv_compression_only":
                        if not isinstance(rack_payload, _Stage5RackPrefixPayload):
                            raise AssertionError("Compression-only Stage 5 path requires a rack payload.")
                        for query_head_global in head_group:
                            query = _float64_numpy(query_states[token_index, query_head_global, :])
                            out = _shared_reconstructed_attention_output_numpy(
                                query=query,
                                keys=rack_payload.reconstructed_keys,
                                values=rack_payload.reconstructed_values,
                                scaling=scaling,
                            )
                            per_head_outputs[query_head_global] = torch.from_numpy(np.asarray(out, dtype=np.float32)).to(torch.bfloat16)
                            del out, query
                            _release_memory()
                    else:
                        keys_tensor = torch.from_numpy(reconstructed_keys.astype(np.float32, copy=False)).to(torch.bfloat16)
                        values_tensor = torch.from_numpy(reconstructed_values.astype(np.float32, copy=False)).to(torch.bfloat16)
                        queries_group = query_states[token_index, head_group, :]
                        out = _manual_attention(
                            queries=queries_group,
                            keys=keys_tensor,
                            values=values_tensor,
                            scaling=scaling,
                        )
                        for local_offset, query_head_global in enumerate(head_group):
                            per_head_outputs[query_head_global] = out[local_offset]
                        del keys_tensor, values_tensor, queries_group, out

                    del prefix_keys_tensor, prefix_values_tensor, prefix_context, byte_breakdown, state, reconstructed_keys, reconstructed_values
                    if "rack_payload" in locals():
                        del rack_payload
                    _release_memory()
                attention_outputs[token_index] = per_head_outputs.reshape(-1).detach().cpu().to(torch.bfloat16)
                del per_head_outputs
                _release_memory()

            _release_memory()
            current_rss = MemoryGuard._rss_bytes()
            current_free = MemoryGuard._available_bytes()
            if memory_guard is not None:
                current_rss, current_free = memory_guard.check(
                    f"after_token_chunk_{layer_index}_{prompt_name}_{method_name}_{chunk_start}_{chunk_end}"
                )
            total_mpfr_evaluations += chunk_mpfr_evaluations
            total_serialization_calls += chunk_serialization_calls
            if diagnostic_records is not None:
                diagnostic_records.append(
                    {
                        "stage": "after_token_chunk",
                        "prompt_name": prompt_name,
                        "layer_index": layer_index,
                        "method_name": method_name,
                        "token_chunk_start": chunk_start,
                        "token_chunk_end_exclusive": chunk_end,
                        "rss_bytes": current_rss,
                        "available_bytes": current_free,
                        "live_tensor_bytes": _tensor_container_bytes(
                            (hidden_states, normalized_hidden, query_states, key_states, value_states, attention_outputs)
                        ),
                        "peak_rss_bytes": memory_guard.peak_rss_bytes if memory_guard is not None else current_rss,
                        "min_available_bytes": memory_guard.min_available_bytes if memory_guard is not None else current_free,
                        "reconstructed_block_count": chunk_reconstructed_block_count,
                        "reconstructed_block_bytes": chunk_reconstructed_block_bytes,
                        "serialization_buffer_bytes": chunk_serialization_buffer_bytes,
                        "serialization_calls": chunk_serialization_calls,
                        "mpfr_evaluations": chunk_mpfr_evaluations,
                    }
                )
            if progress_interval is not None and (chunk_end % progress_interval == 0 or chunk_end == sequence_length):
                elapsed = time.perf_counter() - profile_start
                cache_hits = 0 if cache_summary is None else int(cache_summary.get("project_local_persistent_cache", 0))
                print(
                    f"profile method={method_name} tokens={chunk_end}/{sequence_length} "
                    f"runtime_s={elapsed:.1f} mpfr_evaluations={total_mpfr_evaluations} "
                    f"serialization_calls={total_serialization_calls} cache_hits={cache_hits} "
                    f"rss_gb={current_rss / (1024 ** 3):.2f} free_gb={current_free / (1024 ** 3):.2f}",
                    flush=True,
                )
            _release_memory()
            cleanup_rss = MemoryGuard._rss_bytes()
            cleanup_free = MemoryGuard._available_bytes()
            if memory_guard is not None:
                cleanup_rss, cleanup_free = memory_guard.check(
                    f"after_token_chunk_cleanup_{layer_index}_{prompt_name}_{method_name}_{chunk_start}_{chunk_end}"
                )
            if diagnostic_records is not None:
                diagnostic_records.append(
                    {
                        "stage": "after_chunk_cleanup",
                        "prompt_name": prompt_name,
                        "layer_index": layer_index,
                        "method_name": method_name,
                        "token_chunk_start": chunk_start,
                        "token_chunk_end_exclusive": chunk_end,
                        "rss_bytes": cleanup_rss,
                        "available_bytes": cleanup_free,
                        "live_tensor_bytes": _tensor_container_bytes(
                            (hidden_states, normalized_hidden, query_states, key_states, value_states, attention_outputs)
                        ),
                        "peak_rss_bytes": memory_guard.peak_rss_bytes if memory_guard is not None else cleanup_rss,
                        "min_available_bytes": memory_guard.min_available_bytes if memory_guard is not None else cleanup_free,
                        "reconstructed_block_count": 0,
                        "reconstructed_block_bytes": 0,
                        "serialization_buffer_bytes": 0,
                        "serialization_calls": 0,
                        "mpfr_evaluations": 0,
                    }
                )

        o_proj_weight = tensor_cache.fetch_tensor(weight_index=weight_index, tensor_name=prefix + "self_attn.o_proj.weight").to(torch.bfloat16)
        projected_outputs = F.linear(attention_outputs.to(torch.bfloat16), o_proj_weight).detach().cpu().to(torch.bfloat16).contiguous()
        del o_proj_weight
        _release_memory()

        hidden_states_bf16 = hidden_states.detach().cpu().to(torch.bfloat16).contiguous()
        post_attention_residual = (hidden_states_bf16 + projected_outputs).contiguous()
        del hidden_states_bf16
        _release_memory()

        post_attention_norm_weight = tensor_cache.fetch_tensor(weight_index=weight_index, tensor_name=prefix + "post_attention_layernorm.weight").to(torch.bfloat16)
        mlp_inputs = _rms_norm(post_attention_residual, post_attention_norm_weight, eps=float(config.rms_norm_eps))
        del post_attention_norm_weight
        _release_memory()

        gate_proj_weight = tensor_cache.fetch_tensor(weight_index=weight_index, tensor_name=prefix + "mlp.gate_proj.weight").to(torch.bfloat16)
        up_proj_weight = tensor_cache.fetch_tensor(weight_index=weight_index, tensor_name=prefix + "mlp.up_proj.weight").to(torch.bfloat16)
        gate_hidden = F.linear(mlp_inputs, gate_proj_weight)
        up_hidden = F.linear(mlp_inputs, up_proj_weight)
        del gate_proj_weight, up_proj_weight, mlp_inputs
        _release_memory()

        mlp_hidden = (F.silu(gate_hidden.to(torch.float32)).to(torch.bfloat16) * up_hidden).contiguous()
        del gate_hidden, up_hidden
        _release_memory()

        down_proj_weight = tensor_cache.fetch_tensor(weight_index=weight_index, tensor_name=prefix + "mlp.down_proj.weight").to(torch.bfloat16)
        mlp_outputs = F.linear(mlp_hidden, down_proj_weight).detach().cpu().to(torch.bfloat16).contiguous()
        del down_proj_weight, mlp_hidden, normalized_hidden, query_states, key_states, value_states
        _release_memory()

        decoder_output = (post_attention_residual + mlp_outputs).contiguous()
        return (
            decoder_output,
            {
                "attention_output": attention_outputs,
                "projected_output": projected_outputs,
                "post_attention_residual": post_attention_residual,
                "mlp_output": mlp_outputs,
                "decoder_output": decoder_output,
            },
            storage_records,
            certificate_records_local,
        )


def _aggregate_layer_differences(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    diffs = torch.linalg.norm(reference.to(torch.float32) - candidate.to(torch.float32), dim=-1).cpu().numpy()
    return {
        "mean_l2_difference": float(np.mean(diffs)) if diffs.size else 0.0,
        "median_l2_difference": float(np.median(diffs)) if diffs.size else 0.0,
        "p95_l2_difference": float(np.quantile(diffs, 0.95)) if diffs.size else 0.0,
        "max_l2_difference": float(np.max(diffs)) if diffs.size else 0.0,
    }


def _stage4_artifact_hashes(root: Path) -> dict[str, str]:
    targets = (
        root / ".tmp" / "stage4_baselines_full_review.zip",
        root / ".tmp" / "stage4_baselines_full" / "stage4_baseline_full_results.json",
        root / ".tmp" / "stage4_baselines_full" / "stage4_baseline_full_report.md",
        root / ".tmp" / "stage4_baselines_full" / "manifest.json",
    )
    hashes: dict[str, str] = {}
    for path in targets:
        if not path.exists():
            raise Stage5ExecutionError(f"Required frozen Stage 4B artifact is missing: {path}")
        hashes[str(path)] = _sha256_file(path)
    return hashes


def _validate_stage4_artifacts_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> None:
    if dict(before) != dict(after):
        raise Stage5ExecutionError("Frozen Stage 4B artifacts changed during Stage 5 execution.")


def _validate_selected_heads(summary: CheckpointConfigSummary) -> None:
    if summary.num_attention_heads != 32 or summary.num_key_value_heads != 8 or summary.head_dim != 128:
        raise Stage5ExecutionError(
            "Stage 5A requires the validated Llama-3.1-8B geometry (32 query heads, 8 KV heads, head dim 128)."
        )


def _metric_record_for_token(
    *,
    token_index: int,
    target_token_id: int,
    full_logits: np.ndarray,
    method_logits: np.ndarray,
) -> dict[str, Any]:
    full_log_probs = _log_softmax(full_logits)
    method_log_probs = _log_softmax(method_logits)
    full_top1 = int(np.argmax(full_logits))
    method_top1 = int(np.argmax(method_logits))
    full_top5 = _topk_ids(full_logits, 5)
    method_top5 = _topk_ids(method_logits, 5)
    target_rank_full = _rank_of_token(full_logits, target_token_id)
    target_rank_method = _rank_of_token(method_logits, target_token_id)
    return {
        "schema": STAGE5_METRIC_RECORD_SCHEMA,
        "token_index": token_index,
        "target_token_id": int(target_token_id),
        "full_target_log_probability": float(full_log_probs[target_token_id]),
        "method_target_log_probability": float(method_log_probs[target_token_id]),
        "nll": float(-method_log_probs[target_token_id]),
        "delta_nll_vs_full": float(full_log_probs[target_token_id] - method_log_probs[target_token_id]),
        "top1_token": method_top1,
        "full_top1_token": full_top1,
        "top5_token_ids": method_top5,
        "full_top5_token_ids": full_top5,
        "top1_agreement_with_full": bool(method_top1 == full_top1),
        "top5_contains_full_top1": bool(full_top1 in method_top5),
        "top5_set_overlap_count": int(len(set(full_top5).intersection(method_top5))),
        "kl_divergence": _kl_divergence(full_log_probs, method_log_probs),
        "jensen_shannon_divergence": _js_divergence(full_log_probs, method_log_probs),
        "logit_l2_error": _norm(full_logits - method_logits),
        "relative_logit_l2_error": _relative_l2_error(full_logits, method_logits),
        "max_abs_logit_component_error": _max_abs_component_error(full_logits, method_logits),
        "cosine_similarity": _cosine_similarity(full_logits, method_logits),
        "target_token_rank_full": target_rank_full,
        "target_token_rank_method": target_rank_method,
        "target_token_rank_change": int(target_rank_method - target_rank_full),
    }


def _aggregate_metric_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "case_count": 0,
            "mean_nll": 0.0,
            "total_nll": 0.0,
            "perplexity": 1.0,
            "mean_delta_nll_vs_full": 0.0,
            "mean_top1_agreement": 1.0,
            "mean_top5_contains_full_top1": 1.0,
            "mean_kl_divergence": 0.0,
            "p95_kl_divergence": 0.0,
            "max_kl_divergence": 0.0,
            "mean_js_divergence": 0.0,
            "p95_js_divergence": 0.0,
            "max_js_divergence": 0.0,
            "mean_logit_l2_error": 0.0,
            "p95_logit_l2_error": 0.0,
            "max_logit_l2_error": 0.0,
            "mean_relative_logit_l2_error": 0.0,
            "p95_relative_logit_l2_error": 0.0,
            "max_relative_logit_l2_error": 0.0,
            "mean_max_abs_logit_component_error": 0.0,
            "p95_max_abs_logit_component_error": 0.0,
            "max_max_abs_logit_component_error": 0.0,
            "mean_cosine_similarity": 1.0,
            "min_cosine_similarity": 1.0,
            "metrics_valid": False,
            "invalid_reason": "no_scored_tokens",
        }
    nlls = np.asarray([record["nll"] for record in records], dtype=np.float64)
    delta_nlls = np.asarray([record["delta_nll_vs_full"] for record in records], dtype=np.float64)
    kls = np.asarray([record["kl_divergence"] for record in records], dtype=np.float64)
    jss = np.asarray([record["jensen_shannon_divergence"] for record in records], dtype=np.float64)
    logit_l2 = np.asarray([record["logit_l2_error"] for record in records], dtype=np.float64)
    rel_l2 = np.asarray([record["relative_logit_l2_error"] for record in records], dtype=np.float64)
    max_abs = np.asarray([record["max_abs_logit_component_error"] for record in records], dtype=np.float64)
    cosine = np.asarray([record["cosine_similarity"] for record in records], dtype=np.float64)
    top1 = np.asarray([1.0 if record["top1_agreement_with_full"] else 0.0 for record in records], dtype=np.float64)
    top5 = np.asarray([1.0 if record["top5_contains_full_top1"] else 0.0 for record in records], dtype=np.float64)
    return {
        "case_count": len(records),
        "mean_nll": float(np.mean(nlls)),
        "total_nll": float(np.sum(nlls)),
        "perplexity": float(math.exp(np.mean(nlls))),
        "mean_delta_nll_vs_full": float(np.mean(delta_nlls)),
        "mean_top1_agreement": float(np.mean(top1)),
        "mean_top5_contains_full_top1": float(np.mean(top5)),
        "mean_kl_divergence": float(np.mean(kls)),
        "p95_kl_divergence": float(np.quantile(kls, 0.95)),
        "max_kl_divergence": float(np.max(kls)),
        "mean_js_divergence": float(np.mean(jss)),
        "p95_js_divergence": float(np.quantile(jss, 0.95)),
        "max_js_divergence": float(np.max(jss)),
        "mean_logit_l2_error": float(np.mean(logit_l2)),
        "p95_logit_l2_error": float(np.quantile(logit_l2, 0.95)),
        "max_logit_l2_error": float(np.max(logit_l2)),
        "mean_relative_logit_l2_error": float(np.mean(rel_l2)),
        "p95_relative_logit_l2_error": float(np.quantile(rel_l2, 0.95)),
        "max_relative_logit_l2_error": float(np.max(rel_l2)),
        "mean_max_abs_logit_component_error": float(np.mean(max_abs)),
        "p95_max_abs_logit_component_error": float(np.quantile(max_abs, 0.95)),
        "max_max_abs_logit_component_error": float(np.max(max_abs)),
        "mean_cosine_similarity": float(np.mean(cosine)),
        "min_cosine_similarity": float(np.min(cosine)),
        "metrics_valid": True,
        "invalid_reason": None,
    }


def _method_storage_summary(
    *,
    method_name: str,
    storage_records: Sequence[dict[str, Any]] | None = None,
    modified_token_totals: Sequence[int] | None = None,
    category_sums_override: Mapping[str, int] | None = None,
    total_layers: int,
    modified_layers: Sequence[int],
    num_kv_heads: int | None,
    head_dim: int | None,
    sequence_length: int,
    full_layer_exact_per_token: int | None = None,
    recent_window: int = STAGE5_RECENT_WINDOW,
    block_size: int = STAGE5_BLOCK_SIZE,
    precision: int = STAGE5_PRECISION,
) -> dict[str, Any]:
    modified_set = set(modified_layers)
    if full_layer_exact_per_token is None:
        if num_kv_heads is None or head_dim is None:
            raise Stage5ExecutionError(
                "Stage 5 storage summary requires either full_layer_exact_per_token or both num_kv_heads and head_dim."
            )
        full_layer_exact_per_token = int(num_kv_heads) * int(head_dim) * 2 * 2
    else:
        full_layer_exact_per_token = int(full_layer_exact_per_token)
    if method_name == "full_kv":
        token_totals = [
            (token_index + 1) * full_layer_exact_per_token * total_layers
            for token_index in range(sequence_length)
        ]
        full_reference_totals = list(token_totals)
        category_sums = {
            "anchor_bytes": 0,
            "encoded_key_bytes": int(sum((token_index + 1) * num_kv_heads * head_dim * 2 * total_layers for token_index in range(sequence_length))),
            "encoded_value_bytes": int(sum((token_index + 1) * num_kv_heads * head_dim * 2 * total_layers for token_index in range(sequence_length))),
            "quantization_scale_bytes": 0,
            "recent_window_bytes": 0,
            "block_metadata_bytes": 0,
            "index_bytes": 0,
            "certificate_metadata_bytes": 0,
            "container_header_bytes": 0,
            "other_serialized_bytes": 0,
        }
        return {
            "storage_accounting_schema": STAGE5_STORAGE_ACCOUNTING_SCHEMA,
            "method_name": method_name,
            "token_totals_scope": "all_layers_cumulative_prefix_bytes",
            "full_reference_scope": "all_layers_exact_full_kv_cumulative_prefix_bytes",
            "category_sums_scope": "all_layers_cumulative_prefix_bytes",
            "modified_layer_count": 0,
            "unmodified_layer_count": int(total_layers),
            "modified_layers_token_totals_bytes": [0 for _ in range(sequence_length)],
            "modified_layers_cumulative_total_serialized_bytes": 0,
            "final_modified_layers_serialized_bytes": 0,
            "unmodified_exact_token_totals_bytes": list(token_totals),
            "unmodified_exact_cumulative_total_bytes": int(sum(token_totals)),
            "token_totals_bytes": token_totals,
            "full_reference_totals_bytes": full_reference_totals,
            "cumulative_total_serialized_bytes": int(sum(token_totals)),
            "cumulative_full_reference_bytes": int(sum(full_reference_totals)),
            "final_total_bytes": int(token_totals[-1]) if token_totals else 0,
            "final_full_reference_bytes": int(full_reference_totals[-1]) if full_reference_totals else 0,
            "final_compression_ratio_vs_full_kv": 1.0,
            "final_memory_saving_fraction_vs_full_kv": 0.0,
            "mean_compression_ratio_vs_full_kv": 1.0,
            "mean_memory_saving_fraction_vs_full_kv": 0.0,
            "category_sums": category_sums,
            "disjoint_category_total_bytes": _storage_category_sum(category_sums),
            "byte_accounting_consistent": True,
        }
    if storage_records is not None and (modified_token_totals is not None or category_sums_override is not None):
        raise Stage5ExecutionError(
            f"Stage 5 storage summary for {method_name} received mixed scopes: "
            "storage_records cannot be combined with modified_token_totals/category_sums_override."
        )
    if storage_records is None and ((modified_token_totals is None) != (category_sums_override is None)):
        raise Stage5ExecutionError(
            f"Stage 5 storage summary for {method_name} requires modified_token_totals and category_sums_override together."
        )
    if storage_records is None and modified_token_totals is None:
        raise Stage5ExecutionError(
            f"Stage 5 storage summary for {method_name} requires either authoritative storage_records or checkpoint aggregate totals."
        )

    modified_layers_token_totals = [0 for _ in range(sequence_length)]
    category_sums = _empty_storage_category_sums()
    if storage_records is not None:
        if head_dim is None:
            raise Stage5ExecutionError("Stage 5 storage summary from storage_records requires head_dim.")
        for record in storage_records:
            layer_index = int(record["layer_index"])
            if layer_index not in modified_set:
                raise Stage5ExecutionError(
                    f"Stage 5 storage summary for {method_name} received non-modified layer storage record {layer_index}."
                )
            token_index = int(record["token_index"])
            if token_index < 0 or token_index >= sequence_length:
                raise Stage5ExecutionError(
                    f"Stage 5 storage summary for {method_name} received token_index={token_index} outside sequence length {sequence_length}."
                )
            categories = _disjoint_storage_categories_from_record(
                method_name=method_name,
                record=record,
                head_dim=int(head_dim),
                value_dim=int(head_dim),
                recent_window=recent_window,
                block_size=block_size,
                precision=precision,
            )
            record_total = int(record["byte_breakdown"]["total_serialized_bytes"])
            if _storage_category_sum(categories) != record_total:
                raise Stage5ExecutionError(
                    f"Stage 5 storage summary record mismatch for {method_name}: "
                    f"categories={_storage_category_sum(categories)} total={record_total}."
                )
            modified_layers_token_totals[token_index] += record_total
            for key, value in categories.items():
                category_sums[key] += int(value)
    else:
        if len(modified_token_totals) != sequence_length:
            raise Stage5ExecutionError(
                f"Stage 5 storage summary for {method_name} expected {sequence_length} modified token totals, "
                f"found {len(modified_token_totals)}."
            )
        for key in category_sums:
            category_sums[key] = int(category_sums_override.get(key, 0))
        unexpected_keys = sorted(set(category_sums_override.keys()) - set(category_sums.keys()))
        if unexpected_keys:
            raise Stage5ExecutionError(
                f"Stage 5 storage summary for {method_name} received unexpected storage category keys: {unexpected_keys}."
            )
        modified_layers_token_totals = [int(value) for value in modified_token_totals]

    modified_category_total = _storage_category_sum(category_sums)
    modified_token_total_sum = int(sum(modified_layers_token_totals))
    if modified_category_total != modified_token_total_sum:
        raise Stage5ExecutionError(
            f"Stage 5 storage summary mismatch for {method_name}: "
            f"modified-layer category total {modified_category_total} != modified-layer serialized total {modified_token_total_sum}."
        )

    modified_layer_count = len([layer for layer in range(total_layers) if layer in modified_set])
    unmodified_layer_count = int(total_layers - modified_layer_count)
    unmodified_exact_token_totals = [
        (token_index + 1) * full_layer_exact_per_token * unmodified_layer_count
        for token_index in range(sequence_length)
    ]
    token_totals = [
        int(modified_layers_token_totals[token_index] + unmodified_exact_token_totals[token_index])
        for token_index in range(sequence_length)
    ]
    full_reference_totals = [
        (token_index + 1) * full_layer_exact_per_token * total_layers
        for token_index in range(sequence_length)
    ]
    ratios = [
        float(full_reference_totals[index] / token_totals[index]) if token_totals[index] > 0 else math.inf
        for index in range(sequence_length)
    ]
    savings = [
        float(1.0 - (token_totals[index] / full_reference_totals[index])) if full_reference_totals[index] > 0 else 0.0
        for index in range(sequence_length)
    ]
    return {
        "storage_accounting_schema": STAGE5_STORAGE_ACCOUNTING_SCHEMA,
        "method_name": method_name,
        "token_totals_scope": "all_layers_cumulative_prefix_bytes",
        "full_reference_scope": "all_layers_exact_full_kv_cumulative_prefix_bytes",
        "category_sums_scope": "modified_layers_cumulative_serialized_bytes",
        "modified_layer_count": int(modified_layer_count),
        "unmodified_layer_count": int(unmodified_layer_count),
        "modified_layers_token_totals_bytes": modified_layers_token_totals,
        "modified_layers_cumulative_total_serialized_bytes": int(modified_token_total_sum),
        "final_modified_layers_serialized_bytes": int(modified_layers_token_totals[-1]) if modified_layers_token_totals else 0,
        "unmodified_exact_token_totals_bytes": unmodified_exact_token_totals,
        "unmodified_exact_cumulative_total_bytes": int(sum(unmodified_exact_token_totals)),
        "token_totals_bytes": token_totals,
        "full_reference_totals_bytes": full_reference_totals,
        "cumulative_total_serialized_bytes": int(sum(token_totals)),
        "cumulative_full_reference_bytes": int(sum(full_reference_totals)),
        "final_total_bytes": int(token_totals[-1]),
        "final_full_reference_bytes": int(full_reference_totals[-1]),
        "final_compression_ratio_vs_full_kv": float(ratios[-1]),
        "final_memory_saving_fraction_vs_full_kv": float(savings[-1]),
        "mean_compression_ratio_vs_full_kv": float(np.mean(ratios)),
        "mean_memory_saving_fraction_vs_full_kv": float(np.mean(savings)),
        "category_sums": category_sums,
        "disjoint_category_total_bytes": int(modified_category_total),
        "byte_accounting_consistent": True,
    }


def _aggregate_certificate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> dict[str, Any]:
    gqa_groups = {kv_head_global: set(group) for kv_head_global, group in enumerate(_gqa_groups(num_attention_heads, num_key_value_heads))}
    unique_tokens_with_skip: set[tuple[str, int, int]] = set()
    unique_layers_with_skip: set[int] = set()
    unique_query_heads_with_skip: set[int] = set()
    unique_kv_heads_with_skip: set[int] = set()
    unique_logical_blocks_with_skip: set[tuple[int, int, int]] = set()
    full_group_block_skips: set[tuple[str, int, int, int, int]] = set()
    physical_decode_avoided: set[tuple[str, int, int, int, int]] = set()
    skipped_heads_by_logical_block: dict[tuple[str, int, int, int, int], set[int]] = {}
    token_skip_fractions: dict[tuple[str, int, int], list[float]] = {}
    skips_lacking_proof_records = 0

    total_eligible_query_head_block_decisions = 0
    prefilter_rejected_query_head_block_decisions = 0
    candidates_sent_to_mpfr = 0
    mpfr_certified_query_head_block_skips = 0
    mpfr_rejected_candidates = 0
    rigorous_interval_violations = 0
    approximate_observed_violations = 0
    false_safe_count = 0
    numerical_fallbacks = 0

    for record in records:
        eligible = int(record.get("eligible_blocks", 0))
        invoked = bool(record.get("mpfr_invoked", False))
        prompt_name = str(record.get("prompt_name", ""))
        layer_index = int(record.get("layer_index", 0))
        token_index = int(record.get("token_index", 0))
        query_head_global = int(record.get("query_head_global", 0))
        kv_head_global = int(record.get("kv_head_global", 0))
        skipped_block_starts = [int(value) for value in record.get("skipped_block_starts", ())]
        decoded_block_starts = [int(value) for value in record.get("decoded_block_starts", ())]
        certified_skips = int(record.get("mpfr_certified_skipped_blocks", len(skipped_block_starts)))
        candidates_this_record = int(record.get("potential_skip_candidates_sent_to_mpfr", 0))

        total_eligible_query_head_block_decisions += eligible
        if "prefilter_rejected_decisions" in record:
            prefilter_rejected_query_head_block_decisions += int(record.get("prefilter_rejected_decisions", 0))
        elif not invoked:
            prefilter_rejected_query_head_block_decisions += int(record.get("fast_prefilter_rejected_blocks", 0))
        candidates_sent_to_mpfr += candidates_this_record
        mpfr_certified_query_head_block_skips += certified_skips
        if "mpfr_rejected_candidates" in record:
            mpfr_rejected_candidates += int(record.get("mpfr_rejected_candidates", 0))
        elif invoked:
            mpfr_rejected_candidates += int(record.get("mpfr_rejected_skip_candidates", len(decoded_block_starts)))
        rigorous_interval_violations += int(record.get("rigorous_interval_violation", 0))
        approximate_observed_violations += int(record.get("approximate_observed_violation", 0))
        false_safe_count += int(record.get("false_safe_count", 0))
        numerical_fallbacks += int(record.get("numerical_fallback_used", 0))

        if eligible > 0:
            token_key = (prompt_name, layer_index, token_index)
            token_skip_fractions.setdefault(token_key, []).append(float(certified_skips / eligible))

        if certified_skips > 0:
            if not invoked or record.get("certificate_bound_text") in {None, ""}:
                skips_lacking_proof_records += 1
            unique_tokens_with_skip.add((prompt_name, layer_index, token_index))
            unique_layers_with_skip.add(layer_index)
            unique_query_heads_with_skip.add(query_head_global)
            unique_kv_heads_with_skip.add(kv_head_global)
            for block_start in skipped_block_starts:
                unique_logical_blocks_with_skip.add((layer_index, kv_head_global, block_start))
                logical_key = (prompt_name, layer_index, token_index, kv_head_global, block_start)
                skipped_heads_by_logical_block.setdefault(logical_key, set()).add(query_head_global)

        for block_start in record.get("physically_decoded_avoided_block_starts", ()):
            physical_decode_avoided.add((prompt_name, layer_index, token_index, kv_head_global, int(block_start)))

    for logical_key, heads in skipped_heads_by_logical_block.items():
        _prompt_name, _layer_index, _token_index, kv_head_global, _block_start = logical_key
        if heads == gqa_groups.get(kv_head_global, set()):
            full_group_block_skips.add(logical_key)

    max_per_token_skip_fraction = 0.0
    if token_skip_fractions:
        max_per_token_skip_fraction = max(max(values) for values in token_skip_fractions.values())

    return {
        "total_certificate_records": int(len(records)),
        "total_eligible_query_head_block_decisions": int(total_eligible_query_head_block_decisions),
        "prefilter_rejected_query_head_block_decisions": int(prefilter_rejected_query_head_block_decisions),
        "candidates_sent_to_mpfr": int(candidates_sent_to_mpfr),
        "mpfr_certified_query_head_block_skips": int(mpfr_certified_query_head_block_skips),
        "mpfr_rejected_candidates": int(mpfr_rejected_candidates),
        "unique_tokens_with_any_skip": int(len(unique_tokens_with_skip)),
        "unique_layers_with_any_skip": int(len(unique_layers_with_skip)),
        "unique_query_heads_with_any_skip": int(len(unique_query_heads_with_skip)),
        "unique_kv_heads_with_any_skip": int(len(unique_kv_heads_with_skip)),
        "unique_logical_kv_blocks_with_any_skip": int(len(unique_logical_blocks_with_skip)),
        "gqa_physical_blocks_skippable_by_all_mapped_query_heads": int(len(full_group_block_skips)),
        "physical_block_decodes_actually_avoided": int(len(physical_decode_avoided)),
        "weighted_skip_fraction": float(
            mpfr_certified_query_head_block_skips / total_eligible_query_head_block_decisions
        ) if total_eligible_query_head_block_decisions > 0 else 0.0,
        "max_per_token_skip_fraction": float(max_per_token_skip_fraction),
        "rigorous_interval_violations": int(rigorous_interval_violations),
        "approximate_observed_violations": int(approximate_observed_violations),
        "false_safe_count": int(false_safe_count),
        "numerical_fallbacks": int(numerical_fallbacks),
        "skips_lacking_proof_records": int(skips_lacking_proof_records),
    }


def _prompt_checkpoint_path(output_dir: Path, prompt_name: str) -> Path:
    return output_dir / STAGE5_CHECKPOINT_DIR / f"{prompt_name}.json"


def _stream_checkpoint_path(output_dir: Path, prompt_name: str, method_name: str) -> Path:
    return output_dir / STAGE5_CHECKPOINT_DIR / f"{prompt_name}__{method_name}.json"


def _stream_hidden_state_path(output_dir: Path, prompt_name: str, method_name: str) -> Path:
    return output_dir / STAGE5_CHECKPOINT_DIR / f"{prompt_name}__{method_name}__hidden.safetensors"


def _stream_reference_layer_path(output_dir: Path, prompt_name: str, layer_index: int) -> Path:
    return output_dir / STAGE5_CHECKPOINT_DIR / f"{prompt_name}__full_kv__layer_{layer_index}.safetensors"


def _stream_full_logits_path(output_dir: Path, prompt_name: str) -> Path:
    return output_dir / STAGE5_CHECKPOINT_DIR / f"{prompt_name}__full_kv__logits.safetensors"


def _stream_full_hidden_path(output_dir: Path, prompt_name: str) -> Path:
    return output_dir / STAGE5_CHECKPOINT_DIR / f"{prompt_name}__full_kv__final_hidden.safetensors"


def _save_tensor_artifact(path: Path, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, Any] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_strings = {
        key: json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
        for key, value in (metadata or {}).items()
    }
    temp_path = path.with_name(path.name + ".tmp")
    try:
        save_safetensors_file(
            {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()},
            str(temp_path),
            metadata=metadata_strings,
        )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
    return path


def _load_tensor_artifact(path: Path) -> dict[str, torch.Tensor]:
    return load_safetensors_file(str(path))


def _stream_settings_payload(
    *,
    method_name: str,
    prompt: PromptMaterialized,
    repo_id: str,
    repo_revision: str,
    recent_window: int,
    block_size: int,
    tolerance: float,
    precision: int,
    seed: int,
    total_layers_to_run: int,
) -> dict[str, Any]:
    return {
        "prompt_name": prompt.name,
        "prompt_source_sha256": prompt.source_sha256,
        "token_count": prompt.token_count,
        "repo_id": repo_id,
        "repo_revision": repo_revision,
        "recent_window": int(recent_window),
        "block_size": int(block_size),
        "tolerance": float(tolerance),
        "precision": int(precision),
        "seed": int(seed),
        "total_layers_to_run": int(total_layers_to_run),
        "method_version": STAGE5_METHOD_VERSION,
        "method_implementation_version": _stage5_method_impl_version(method_name),
        "result_version": STAGE5_RESULT_VERSION,
    }


def _save_prompt_checkpoint(output_dir: Path, prompt_name: str, payload: dict[str, Any]) -> Path:
    path = _prompt_checkpoint_path(output_dir, prompt_name)
    return _write_json_atomic(path, payload)


def _load_prompt_checkpoint(output_dir: Path, prompt_name: str) -> dict[str, Any] | None:
    path = _prompt_checkpoint_path(output_dir, prompt_name)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != STAGE5_PROMPT_CHECKPOINT_SCHEMA:
        return None
    return payload


def _save_stream_checkpoint(output_dir: Path, prompt_name: str, method_name: str, payload: dict[str, Any]) -> Path:
    return _write_json_atomic(_stream_checkpoint_path(output_dir, prompt_name, method_name), payload)


def _load_stream_checkpoint(
    output_dir: Path,
    prompt_name: str,
    method_name: str,
    *,
    expected_settings: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = _stream_checkpoint_path(output_dir, prompt_name, method_name)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != STAGE5_STREAM_CHECKPOINT_SCHEMA:
        return None
    if payload.get("method_name") != method_name:
        return None
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        return None
    if dict(settings) != dict(expected_settings):
        return None
    hidden_state_path = payload.get("hidden_state_path")
    if hidden_state_path is not None and not Path(hidden_state_path).exists():
        return None
    return payload


def _stage5_progress(
    *,
    prompt_name: str,
    layer_index: int,
    total_layers: int,
    active_streams: Sequence[str],
    elapsed_s: float,
    cache_summary: Mapping[str, Any],
    guard: MemoryGuard,
    current_rss_bytes: int,
    current_available_bytes: int,
    live_tensor_bytes: int,
    tensor_cache_process_bytes: int,
    prompts_done: int,
    total_prompts: int,
) -> None:
    average_layers = elapsed_s / max((layer_index + 1) + prompts_done * total_layers, 1)
    remaining_layers = (total_prompts - prompts_done - 1) * total_layers + (total_layers - layer_index - 1)
    eta = average_layers * remaining_layers
    print(
        f"prompt={prompt_name} layer={layer_index + 1}/{total_layers} streams={','.join(active_streams)} "
        f"elapsed={elapsed_s:.1f}s cache_hits={cache_summary.get('project_local_persistent_cache', 0)} "
        f"remote_fetches={cache_summary.get('newly_fetched_remote_range', 0)} "
        f"rss_gb={current_rss_bytes / (1024 ** 3):.2f} "
        f"peak_rss_gb={guard.peak_rss_bytes / (1024 ** 3):.2f} "
        f"live_tensors_mb={live_tensor_bytes / (1024 ** 2):.1f} "
        f"cache_mem_mb={tensor_cache_process_bytes / (1024 ** 2):.1f} "
        f"free_gb={current_available_bytes / (1024 ** 3):.2f} eta={eta:.1f}s",
        flush=True,
    )


def _write_provenance_json(output_dir: Path, filename: str, payload: Mapping[str, Any]) -> Path:
    return _write_json_atomic(output_dir / STAGE5_PROVENANCE_DIR / filename, payload)


def _empty_storage_category_sums() -> dict[str, int]:
    return {
        "anchor_bytes": 0,
        "encoded_key_bytes": 0,
        "encoded_value_bytes": 0,
        "quantization_scale_bytes": 0,
        "recent_window_bytes": 0,
        "block_metadata_bytes": 0,
        "index_bytes": 0,
        "certificate_metadata_bytes": 0,
        "container_header_bytes": 0,
        "other_serialized_bytes": 0,
    }


def _save_reference_layer_outputs(
    output_dir: Path,
    prompt_name: str,
    layer_index: int,
    layer_outputs: Mapping[str, torch.Tensor],
) -> Path:
    return _save_tensor_artifact(
        _stream_reference_layer_path(output_dir, prompt_name, layer_index),
        {
            "attention_output": layer_outputs["attention_output"],
            "projected_output": layer_outputs["projected_output"],
            "post_attention_residual": layer_outputs["post_attention_residual"],
            "mlp_output": layer_outputs["mlp_output"],
            "decoder_output": layer_outputs["decoder_output"],
        },
        metadata={"prompt_name": prompt_name, "layer_index": int(layer_index)},
    )


def _load_reference_layer_outputs(output_dir: Path, prompt_name: str, layer_index: int) -> dict[str, torch.Tensor]:
    return _load_tensor_artifact(_stream_reference_layer_path(output_dir, prompt_name, layer_index))


def _storage_category_sum(categories: Mapping[str, int]) -> int:
    return int(sum(int(value) for value in categories.values()))


def _block_lengths_for_historical_tokens(historical_tokens: int, block_size: int) -> list[int]:
    lengths: list[int] = []
    cursor = 0
    historical = max(int(historical_tokens), 0)
    step = max(int(block_size), 1)
    while cursor < historical:
        block_len = min(step, historical - cursor)
        lengths.append(block_len)
        cursor += block_len
    return lengths


def _disjoint_storage_categories_from_record(
    *,
    method_name: str,
    record: Mapping[str, Any],
    head_dim: int,
    value_dim: int,
    recent_window: int,
    block_size: int,
    precision: int,
) -> dict[str, int]:
    breakdown = record["byte_breakdown"]
    total_serialized_bytes = int(breakdown["total_serialized_bytes"])
    visible_length = int(record["visible_length"])
    historical_tokens = int(record["historical_tokens"])

    if method_name == "full_kv":
        categories = {
            "anchor_bytes": 0,
            "encoded_key_bytes": int(breakdown["encoded_key_bytes"]),
            "encoded_value_bytes": int(breakdown["encoded_value_bytes"]),
            "quantization_scale_bytes": 0,
            "recent_window_bytes": 0,
            "block_metadata_bytes": 0,
            "index_bytes": 0,
            "certificate_metadata_bytes": 0,
            "container_header_bytes": int(breakdown["metadata_bytes"]),
            "other_serialized_bytes": total_serialized_bytes
            - int(breakdown["encoded_key_bytes"])
            - int(breakdown["encoded_value_bytes"])
            - int(breakdown["metadata_bytes"]),
        }
    elif method_name == "uniform_int8_kv":
        categories = {
            "anchor_bytes": 0,
            "encoded_key_bytes": int(breakdown["encoded_key_bytes"]),
            "encoded_value_bytes": int(breakdown["encoded_value_bytes"]),
            "quantization_scale_bytes": int(breakdown["scales_bytes"]),
            "recent_window_bytes": int(breakdown["recent_window_bytes"]),
            "block_metadata_bytes": int(breakdown["block_page_metadata_bytes"]),
            "index_bytes": int(breakdown["indices_bytes"]),
            "certificate_metadata_bytes": 0,
            "container_header_bytes": int(breakdown["metadata_bytes"]),
            "other_serialized_bytes": total_serialized_bytes
            - int(breakdown["encoded_key_bytes"])
            - int(breakdown["encoded_value_bytes"])
            - int(breakdown["scales_bytes"])
            - int(breakdown["recent_window_bytes"])
            - int(breakdown["block_page_metadata_bytes"])
            - int(breakdown["indices_bytes"])
            - int(breakdown["metadata_bytes"]),
        }
    elif method_name in {"rack_kv_compression_only", "rack_kv_certified"}:
        recent_exact_tokens = max(visible_length - historical_tokens, 0)
        block_lengths = _block_lengths_for_historical_tokens(historical_tokens, block_size)
        block_count = len(block_lengths)
        anchor_bytes = block_count * (2 * head_dim + 2 * value_dim)
        encoded_key_bytes = sum(max(block_len - 1, 0) * head_dim for block_len in block_lengths)
        encoded_value_bytes = sum(max(block_len - 1, 0) * value_dim for block_len in block_lengths)
        quantization_scale_bytes = block_count * 4
        block_metadata_bytes = block_count * 16
        index_bytes = block_count * 8
        recent_window_bytes = recent_exact_tokens * head_dim * 2 + recent_exact_tokens * value_dim * 2
        historical_container_bytes = 0
        serialized_container_header_bytes = 0
        if block_count > 0:
            serialized_container_header_bytes = 16
            historical_container_bytes = (
                serialized_container_header_bytes
                + index_bytes
                + block_metadata_bytes
                + anchor_bytes
                + quantization_scale_bytes
                + encoded_key_bytes
                + encoded_value_bytes
            )
        computed_total_serialized_bytes, wrapper_header_bytes = _stage5_wrapper_total_bytes(
            recent_keys_shape=(recent_exact_tokens, head_dim),
            recent_values_shape=(recent_exact_tokens, value_dim),
            recent_keys_bytes=recent_exact_tokens * head_dim * 2,
            recent_values_bytes=recent_exact_tokens * value_dim * 2,
            historical_container_bytes=historical_container_bytes,
            recent_window=recent_window,
            block_size=block_size,
            precision=precision,
            historical_tokens=historical_tokens,
        )
        if computed_total_serialized_bytes != total_serialized_bytes:
            raise Stage5ExecutionError(
                "Stage 5 runtime rack accounting mismatch between computed serialized total "
                f"{computed_total_serialized_bytes} and recorded total {total_serialized_bytes}."
            )
        categories = {
            "anchor_bytes": int(anchor_bytes),
            "encoded_key_bytes": int(encoded_key_bytes),
            "encoded_value_bytes": int(encoded_value_bytes),
            "quantization_scale_bytes": int(quantization_scale_bytes),
            "recent_window_bytes": int(recent_window_bytes),
            "block_metadata_bytes": int(block_metadata_bytes),
            "index_bytes": int(index_bytes),
            "certificate_metadata_bytes": 0,
            "container_header_bytes": int(wrapper_header_bytes + serialized_container_header_bytes),
            "other_serialized_bytes": 0,
        }
    else:
        raise Stage5ExecutionError(f"Unsupported Stage 5 storage-accounting method: {method_name}.")

    if categories["certificate_metadata_bytes"] != 0 and method_name == "rack_kv_compression_only":
        raise Stage5ExecutionError("rack_kv_compression_only must not report certificate metadata bytes.")
    if _storage_category_sum(categories) != total_serialized_bytes:
        raise Stage5ExecutionError(
            f"Stage 5 disjoint storage categories do not sum to total bytes for {method_name}: "
            f"{_storage_category_sum(categories)} != {total_serialized_bytes}."
        )
    return categories


def _accumulate_stage5_storage_records(
    *,
    method_name: str,
    storage_records: Sequence[Mapping[str, Any]],
    storage_modified_token_totals: list[int],
    storage_category_sums: dict[str, int],
    head_dim: int,
    value_dim: int,
    recent_window: int,
    block_size: int,
    precision: int,
) -> None:
    for item in storage_records:
        categories = _disjoint_storage_categories_from_record(
            method_name=method_name,
            record=item,
            head_dim=head_dim,
            value_dim=value_dim,
            recent_window=recent_window,
            block_size=block_size,
            precision=precision,
        )
        total_serialized_bytes = int(item["byte_breakdown"]["total_serialized_bytes"])
        storage_modified_token_totals[int(item["token_index"])] += total_serialized_bytes
        for key, value in categories.items():
            storage_category_sums[key] += int(value)


def _save_full_stream_artifacts(
    *,
    output_dir: Path,
    prompt_name: str,
    full_hidden: torch.Tensor,
    full_logits: torch.Tensor,
) -> tuple[Path, Path]:
    hidden_path = _stream_full_hidden_path(output_dir, prompt_name)
    logits_path = _stream_full_logits_path(output_dir, prompt_name)
    _save_tensor_artifact(hidden_path, {"hidden_states": full_hidden}, metadata={"prompt_name": prompt_name})
    _save_tensor_artifact(logits_path, {"logits": full_logits}, metadata={"prompt_name": prompt_name})
    return hidden_path, logits_path


def _load_full_stream_artifacts(output_dir: Path, prompt_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = _load_tensor_artifact(_stream_full_hidden_path(output_dir, prompt_name))["hidden_states"]
    logits = _load_tensor_artifact(_stream_full_logits_path(output_dir, prompt_name))["logits"]
    return hidden, logits


def _run_full_kv_custom_equivalence_regression(
    *,
    config: Any,
    checkpoint_summary: CheckpointConfigSummary,
    tensor_cache: PersistentTensorRangeCache,
    weight_index: dict[str, Any],
    prompt_name: str,
    hidden_states: torch.Tensor,
    recent_window: int,
    block_size: int,
    precision: int,
) -> dict[str, Any]:
    with torch.inference_mode():
        stock_hidden = hidden_states.detach().cpu().clone().to(torch.bfloat16)
        custom_hidden = hidden_states.detach().cpu().clone().to(torch.bfloat16)
        modified_layer_metrics: dict[str, Any] = {}
        modified_set = set(STAGE5_MODIFIED_LAYERS)
        for layer_index in range(checkpoint_summary.num_hidden_layers):
            layer = _load_decoder_layer_exact(
                config=config,
                tensor_cache=tensor_cache,
                weight_index=weight_index,
                layer_index=layer_index,
            )
            try:
                stock_attention, stock_projected, stock_post_attn, stock_mlp, stock_decoder = _stock_layer_forward_with_intermediates(
                    layer=layer,
                    config=config,
                    hidden_states=stock_hidden,
                )
                stock_hidden = stock_decoder
                if layer_index in modified_set:
                    custom_decoder, custom_outputs, _storage, _certs = _run_modified_layer_stream(
                        method_name="full_kv",
                        layer=layer,
                        config=config,
                        hidden_states=custom_hidden,
                        layer_index=layer_index,
                        recent_window=recent_window,
                        block_size=block_size,
                        tolerance=0.0,
                        precision=precision,
                        prompt_name=prompt_name,
                        certificate_records=[],
                    )
                    custom_hidden = custom_decoder
                    modified_layer_metrics[str(layer_index)] = {
                        "attention_output": _aggregate_layer_differences(stock_attention, custom_outputs["attention_output"]),
                        "projected_output": _aggregate_layer_differences(stock_projected, custom_outputs["projected_output"]),
                        "post_attention_residual": _aggregate_layer_differences(stock_post_attn, custom_outputs["post_attention_residual"]),
                        "mlp_output": _aggregate_layer_differences(stock_mlp, custom_outputs["mlp_output"]),
                        "decoder_output": _aggregate_layer_differences(stock_decoder, custom_outputs["decoder_output"]),
                    }
                    del custom_outputs
                else:
                    _attention, _projected, _post_attn, _mlp, custom_decoder = _stock_layer_forward_with_intermediates(
                        layer=layer,
                        config=config,
                        hidden_states=custom_hidden,
                    )
                    custom_hidden = custom_decoder
            finally:
                del layer
                _release_memory()

        norm_weight = _load_rms_norm_weight(
            tensor_cache=tensor_cache,
            weight_index=weight_index,
        )
        stock_logits = _lm_head_logits_chunked(
            hidden_states=_rms_norm(stock_hidden, norm_weight, eps=float(config.rms_norm_eps)),
            tensor_cache=tensor_cache,
            weight_index=weight_index,
        )
        custom_logits = _lm_head_logits_chunked(
            hidden_states=_rms_norm(custom_hidden, norm_weight, eps=float(config.rms_norm_eps)),
            tensor_cache=tensor_cache,
            weight_index=weight_index,
        )
        stock_np = stock_logits.detach().cpu().numpy()
        custom_np = custom_logits.detach().cpu().numpy()
        diff = stock_np - custom_np
        per_token_l2 = np.linalg.norm(diff, axis=1)
        result = {
            "prompt_name": prompt_name,
            "token_count": int(hidden_states.shape[0]),
            "modified_layers": list(STAGE5_MODIFIED_LAYERS),
            "max_abs_logit_difference": float(np.max(np.abs(diff), initial=0.0)),
            "mean_logit_l2_difference": float(np.mean(per_token_l2)) if per_token_l2.size else 0.0,
            "max_logit_l2_difference": float(np.max(per_token_l2)) if per_token_l2.size else 0.0,
            "relative_logit_l2_difference": _relative_l2_error(stock_np, custom_np),
            "cosine_similarity": _cosine_similarity(stock_np.reshape(-1), custom_np.reshape(-1)),
            "modified_layer_metrics": modified_layer_metrics,
        }
        del stock_hidden, custom_hidden, norm_weight, stock_logits, custom_logits, stock_np, custom_np, diff, per_token_l2
        _release_memory()
        return result


def _run_full_kv_prompt_stream(
    *,
    output_dir: Path,
    prompt: PromptMaterialized,
    config: Any,
    checkpoint_summary: CheckpointConfigSummary,
    tensor_cache: PersistentTensorRangeCache,
    weight_index: dict[str, Any],
    guard: MemoryGuard,
    total_layers_to_run: int,
    repo_id: str,
    repo_revision: str,
    recent_window: int,
    block_size: int,
    tolerance: float,
    precision: int,
    seed: int,
    start_time: float,
    memory_profile_records: list[dict[str, Any]],
) -> dict[str, Any]:
    method_name = "full_kv"
    settings = _stream_settings_payload(
        method_name="full_kv",
        prompt=prompt,
        repo_id=repo_id,
        repo_revision=repo_revision,
        recent_window=recent_window,
        block_size=block_size,
        tolerance=tolerance,
        precision=precision,
        seed=seed,
        total_layers_to_run=total_layers_to_run,
    )
    checkpoint = _load_stream_checkpoint(
        output_dir,
        prompt.name,
        method_name,
        expected_settings=settings,
    )
    if checkpoint is not None and checkpoint.get("status") == "complete":
        return checkpoint

    hidden_state_path = _stream_hidden_state_path(output_dir, prompt.name, method_name)
    if checkpoint is not None and checkpoint.get("status") == "in_progress":
        hidden_states = _load_tensor_artifact(Path(checkpoint["hidden_state_path"]))["hidden_states"].to(torch.bfloat16).contiguous()
        next_layer_index = int(checkpoint["next_layer_index"])
        accumulated_runtime_s = float(checkpoint.get("runtime_s", 0.0))
    else:
        prompt_embeddings = _fetch_prompt_embeddings_with_cache(
            tensor_cache=tensor_cache,
            weight_index=weight_index,
            token_ids=prompt.token_ids,
        )
        hidden_states = prompt_embeddings.detach().cpu().clone().to(torch.bfloat16)
        del prompt_embeddings
        _release_memory()
        next_layer_index = 0
        accumulated_runtime_s = 0.0

    stream_start = time.perf_counter()
    modified_set = {layer_index for layer_index in STAGE5_MODIFIED_LAYERS if layer_index < total_layers_to_run}
    with torch.inference_mode():
        for layer_index in range(next_layer_index, total_layers_to_run):
            before_rss, before_free = guard.check(f"before_layer_{layer_index}_{prompt.name}_{method_name}")
            layer = _load_decoder_layer_exact(
                config=config,
                tensor_cache=tensor_cache,
                weight_index=weight_index,
                layer_index=layer_index,
            )
            try:
                attention_output, projected_output, post_attention_residual, mlp_output, decoder_output = _stock_layer_forward_with_intermediates(
                    layer=layer,
                    config=config,
                    hidden_states=hidden_states,
                )
                hidden_states = decoder_output.detach().cpu().to(torch.bfloat16).contiguous()
                if layer_index in modified_set:
                    _save_reference_layer_outputs(
                        output_dir,
                        prompt.name,
                        layer_index,
                        {
                            "attention_output": attention_output,
                            "projected_output": projected_output,
                            "post_attention_residual": post_attention_residual,
                            "mlp_output": mlp_output,
                            "decoder_output": decoder_output,
                        },
                    )
            finally:
                del layer
                if "attention_output" in locals():
                    del attention_output, projected_output, post_attention_residual, mlp_output, decoder_output
                _release_memory()
            _save_tensor_artifact(
                hidden_state_path,
                {"hidden_states": hidden_states},
                metadata={"prompt_name": prompt.name, "method_name": method_name, "next_layer_index": layer_index + 1},
            )
            current_rss, current_free = guard.check(f"after_layer_{layer_index}_{prompt.name}_{method_name}")
            memory_profile_records.append(
                {
                    "prompt_name": prompt.name,
                    "method_name": method_name,
                    "layer_index": layer_index,
                    "rss_bytes_before": before_rss,
                    "rss_bytes_after": current_rss,
                    "available_bytes_before": before_free,
                    "available_bytes_after": current_free,
                    "peak_rss_bytes": guard.peak_rss_bytes,
                    "live_tensor_bytes": _tensor_container_bytes(hidden_states),
                }
            )
            _stage5_progress(
                prompt_name=prompt.name,
                layer_index=layer_index,
                total_layers=total_layers_to_run,
                active_streams=(method_name,),
                elapsed_s=time.perf_counter() - start_time,
                cache_summary=tensor_cache.sources_count,
                guard=guard,
                current_rss_bytes=current_rss,
                current_available_bytes=current_free,
                live_tensor_bytes=_tensor_container_bytes(hidden_states),
                tensor_cache_process_bytes=0,
                prompts_done=0,
                total_prompts=1,
            )
            _save_stream_checkpoint(
                output_dir,
                prompt.name,
                method_name,
                {
                    "schema": STAGE5_STREAM_CHECKPOINT_SCHEMA,
                    "status": "in_progress",
                    "method_name": method_name,
                    "settings": settings,
                    "next_layer_index": layer_index + 1,
                    "hidden_state_path": str(hidden_state_path),
                    "runtime_s": accumulated_runtime_s + (time.perf_counter() - stream_start),
                },
            )

        norm_weight = _load_rms_norm_weight(
            tensor_cache=tensor_cache,
            weight_index=weight_index,
        )
        full_logits = _lm_head_logits_chunked(
            hidden_states=_rms_norm(hidden_states, norm_weight, eps=float(config.rms_norm_eps)),
            tensor_cache=tensor_cache,
            weight_index=weight_index,
        ).detach().cpu()
        del norm_weight
        _release_memory()

    _save_full_stream_artifacts(
        output_dir=output_dir,
        prompt_name=prompt.name,
        full_hidden=hidden_states,
        full_logits=full_logits,
    )
    records, aggregate, passkey_metrics = _prompt_method_metrics(
        prompt=prompt,
        method_name=method_name,
        logits=full_logits,
        full_logits=full_logits,
    )
    aggregate["delta_mean_nll_vs_full_kv"] = 0.0
    aggregate["perplexity_ratio_vs_full_kv"] = 1.0
    aggregate["storage"] = _method_storage_summary(
        method_name=method_name,
        total_layers=total_layers_to_run,
        modified_layers=STAGE5_MODIFIED_LAYERS,
        num_kv_heads=checkpoint_summary.num_key_value_heads,
        head_dim=checkpoint_summary.head_dim,
        sequence_length=len(prompt.token_ids),
    )
    result = {
        "schema": STAGE5_STREAM_CHECKPOINT_SCHEMA,
        "status": "complete",
        "method_name": method_name,
        "settings": settings,
        "next_layer_index": total_layers_to_run,
        "hidden_state_path": str(hidden_state_path),
        "runtime_s": accumulated_runtime_s + (time.perf_counter() - stream_start),
        "intermediate_layer_metrics": {},
        "metric_records": records,
        "method_aggregate": aggregate,
        "passkey_metrics": passkey_metrics,
        "certificate_records": [],
        "full_hidden_path": str(_stream_full_hidden_path(output_dir, prompt.name)),
        "full_logits_path": str(_stream_full_logits_path(output_dir, prompt.name)),
    }
    _save_stream_checkpoint(output_dir, prompt.name, method_name, result)
    del hidden_states, full_logits
    _release_memory()
    return result


def _run_nonfull_prompt_stream(
    *,
    output_dir: Path,
    prompt: PromptMaterialized,
    method_name: str,
    config: Any,
    checkpoint_summary: CheckpointConfigSummary,
    tensor_cache: PersistentTensorRangeCache,
    weight_index: dict[str, Any],
    guard: MemoryGuard,
    total_layers_to_run: int,
    repo_id: str,
    repo_revision: str,
    recent_window: int,
    block_size: int,
    tolerance: float,
    precision: int,
    seed: int,
    start_time: float,
    memory_profile_records: list[dict[str, Any]],
    profile_progress_every_tokens: int | None = None,
) -> dict[str, Any]:
    settings = _stream_settings_payload(
        method_name=method_name,
        prompt=prompt,
        repo_id=repo_id,
        repo_revision=repo_revision,
        recent_window=recent_window,
        block_size=block_size,
        tolerance=tolerance,
        precision=precision,
        seed=seed,
        total_layers_to_run=total_layers_to_run,
    )
    checkpoint = _load_stream_checkpoint(
        output_dir,
        prompt.name,
        method_name,
        expected_settings=settings,
    )
    if checkpoint is not None and checkpoint.get("status") == "complete":
        return checkpoint

    hidden_state_path = _stream_hidden_state_path(output_dir, prompt.name, method_name)
    if checkpoint is not None and checkpoint.get("status") == "in_progress":
        hidden_states = _load_tensor_artifact(Path(checkpoint["hidden_state_path"]))["hidden_states"].to(torch.bfloat16).contiguous()
        next_layer_index = int(checkpoint["next_layer_index"])
        accumulated_runtime_s = float(checkpoint.get("runtime_s", 0.0))
        intermediate_layer_metrics = dict(checkpoint.get("intermediate_layer_metrics", {}))
        storage_modified_token_totals = [int(value) for value in checkpoint.get("storage_modified_token_totals", [0 for _ in range(len(prompt.token_ids))])]
        storage_category_sums = _empty_storage_category_sums()
        storage_category_sums.update({key: int(value) for key, value in checkpoint.get("storage_category_sums", {}).items()})
        certificate_records = list(checkpoint.get("certificate_records", []))
    else:
        prompt_embeddings = _fetch_prompt_embeddings_with_cache(
            tensor_cache=tensor_cache,
            weight_index=weight_index,
            token_ids=prompt.token_ids,
        )
        hidden_states = prompt_embeddings.detach().cpu().clone().to(torch.bfloat16)
        del prompt_embeddings
        _release_memory()
        next_layer_index = 0
        accumulated_runtime_s = 0.0
        intermediate_layer_metrics = {}
        storage_modified_token_totals = [0 for _ in range(len(prompt.token_ids))]
        storage_category_sums = _empty_storage_category_sums()
        certificate_records: list[dict[str, Any]] = []

    stream_start = time.perf_counter()
    modified_set = {layer_index for layer_index in STAGE5_MODIFIED_LAYERS if layer_index < total_layers_to_run}
    token_chunk_size = (
        STAGE5_CERTIFIED_TOKEN_CHUNK_SIZE
        if method_name == "rack_kv_certified"
        else STAGE5_UNIFORM_INT8_TOKEN_CHUNK_SIZE
        if method_name == "uniform_int8_kv"
        else STAGE5_DEFAULT_TOKEN_CHUNK_SIZE
    )
    with torch.inference_mode():
        for layer_index in range(next_layer_index, total_layers_to_run):
            before_rss, before_free = guard.check(f"before_layer_{layer_index}_{prompt.name}_{method_name}")
            if layer_index in modified_set:
                layer_diagnostics: list[dict[str, Any]] | None = [] if profile_progress_every_tokens is not None else None
                new_hidden, layer_outputs, layer_storage, layer_cert_records = _run_modified_layer_stream_exact_from_cache(
                    method_name=method_name,
                    config=config,
                    tensor_cache=tensor_cache,
                    weight_index=weight_index,
                    hidden_states=hidden_states,
                    layer_index=layer_index,
                    recent_window=recent_window,
                    block_size=block_size,
                    tolerance=tolerance,
                    precision=precision,
                    prompt_name=prompt.name,
                    certificate_records=[],
                    memory_guard=guard,
                    token_chunk_size=token_chunk_size,
                    diagnostic_records=layer_diagnostics,
                    profile_progress_every_tokens=profile_progress_every_tokens,
                    cache_summary=tensor_cache.sources_count,
                )
                full_layer_outputs = _load_reference_layer_outputs(output_dir, prompt.name, layer_index)
                intermediate_layer_metrics[str(layer_index)] = {
                    "attention_output": _aggregate_layer_differences(full_layer_outputs["attention_output"], layer_outputs["attention_output"]),
                    "projected_output": _aggregate_layer_differences(full_layer_outputs["projected_output"], layer_outputs["projected_output"]),
                    "post_attention_residual": _aggregate_layer_differences(full_layer_outputs["post_attention_residual"], layer_outputs["post_attention_residual"]),
                    "mlp_output": _aggregate_layer_differences(full_layer_outputs["mlp_output"], layer_outputs["mlp_output"]),
                    "decoder_output": _aggregate_layer_differences(full_layer_outputs["decoder_output"], layer_outputs["decoder_output"]),
                }
                _accumulate_stage5_storage_records(
                    method_name=method_name,
                    storage_records=layer_storage,
                    storage_modified_token_totals=storage_modified_token_totals,
                    storage_category_sums=storage_category_sums,
                    head_dim=checkpoint_summary.head_dim,
                    value_dim=checkpoint_summary.head_dim,
                    recent_window=recent_window,
                    block_size=block_size,
                    precision=precision,
                )
                if method_name == "rack_kv_certified":
                    certificate_records.extend(layer_cert_records)
                if layer_diagnostics:
                    memory_profile_records.extend(layer_diagnostics)
                hidden_states = new_hidden.detach().cpu().to(torch.bfloat16).contiguous()
                del new_hidden, layer_outputs, layer_storage, layer_cert_records, full_layer_outputs, layer_diagnostics
                _release_memory()
            else:
                layer = _load_decoder_layer_exact(
                    config=config,
                    tensor_cache=tensor_cache,
                    weight_index=weight_index,
                    layer_index=layer_index,
                )
                try:
                    _attention, _projected, _post_attn, _mlp, decoder_output = _stock_layer_forward_with_intermediates(
                        layer=layer,
                        config=config,
                        hidden_states=hidden_states,
                    )
                    hidden_states = decoder_output.detach().cpu().to(torch.bfloat16).contiguous()
                    del _attention, _projected, _post_attn, _mlp, decoder_output
                finally:
                    del layer
                    _release_memory()

            _save_tensor_artifact(
                hidden_state_path,
                {"hidden_states": hidden_states},
                metadata={"prompt_name": prompt.name, "method_name": method_name, "next_layer_index": layer_index + 1},
            )
            current_rss, current_free = guard.check(f"after_layer_{layer_index}_{prompt.name}_{method_name}")
            memory_profile_records.append(
                {
                    "prompt_name": prompt.name,
                    "method_name": method_name,
                    "layer_index": layer_index,
                    "rss_bytes_before": before_rss,
                    "rss_bytes_after": current_rss,
                    "available_bytes_before": before_free,
                    "available_bytes_after": current_free,
                    "peak_rss_bytes": guard.peak_rss_bytes,
                    "live_tensor_bytes": _tensor_container_bytes(hidden_states),
                }
            )
            _stage5_progress(
                prompt_name=prompt.name,
                layer_index=layer_index,
                total_layers=total_layers_to_run,
                active_streams=(method_name,),
                elapsed_s=time.perf_counter() - start_time,
                cache_summary=tensor_cache.sources_count,
                guard=guard,
                current_rss_bytes=current_rss,
                current_available_bytes=current_free,
                live_tensor_bytes=_tensor_container_bytes(hidden_states),
                tensor_cache_process_bytes=0,
                prompts_done=0,
                total_prompts=1,
            )
            _save_stream_checkpoint(
                output_dir,
                prompt.name,
                method_name,
                {
                    "schema": STAGE5_STREAM_CHECKPOINT_SCHEMA,
                    "status": "in_progress",
                    "method_name": method_name,
                    "settings": settings,
                    "next_layer_index": layer_index + 1,
                    "hidden_state_path": str(hidden_state_path),
                    "runtime_s": accumulated_runtime_s + (time.perf_counter() - stream_start),
                    "intermediate_layer_metrics": intermediate_layer_metrics,
                    "storage_modified_token_totals": storage_modified_token_totals,
                    "storage_category_sums": storage_category_sums,
                    "certificate_records": certificate_records,
                },
            )

        norm_weight = _load_rms_norm_weight(
            tensor_cache=tensor_cache,
            weight_index=weight_index,
        )
        method_logits = _lm_head_logits_chunked(
            hidden_states=_rms_norm(hidden_states, norm_weight, eps=float(config.rms_norm_eps)),
            tensor_cache=tensor_cache,
            weight_index=weight_index,
        ).detach().cpu()
        del norm_weight
        _release_memory()

    full_hidden, full_logits = _load_full_stream_artifacts(output_dir, prompt.name)
    records, aggregate, passkey_metrics = _prompt_method_metrics(
        prompt=prompt,
        method_name=method_name,
        logits=method_logits,
        full_logits=full_logits,
    )
    full_mean_nll = float(_aggregate_metric_records(_prompt_method_metrics(prompt=prompt, method_name="full_kv", logits=full_logits, full_logits=full_logits)[0])["mean_nll"])
    full_perplexity = float(_aggregate_metric_records(_prompt_method_metrics(prompt=prompt, method_name="full_kv", logits=full_logits, full_logits=full_logits)[0])["perplexity"])
    aggregate["delta_mean_nll_vs_full_kv"] = float(aggregate["mean_nll"] - full_mean_nll)
    aggregate["perplexity_ratio_vs_full_kv"] = float(aggregate["perplexity"] / full_perplexity)
    aggregate["storage"] = _method_storage_summary(
        method_name=method_name,
        modified_token_totals=storage_modified_token_totals,
        category_sums_override=storage_category_sums,
        total_layers=total_layers_to_run,
        modified_layers=STAGE5_MODIFIED_LAYERS,
        num_kv_heads=checkpoint_summary.num_key_value_heads,
        head_dim=checkpoint_summary.head_dim,
        sequence_length=len(prompt.token_ids),
    )
    intermediate_layer_metrics["final_hidden_state"] = _aggregate_layer_differences(full_hidden, hidden_states)
    intermediate_layer_metrics["final_logits"] = _aggregate_layer_differences(full_logits, method_logits)
    if method_name == "rack_kv_certified":
        aggregate["certificate"] = _aggregate_certificate_records(
            certificate_records,
            num_attention_heads=checkpoint_summary.num_attention_heads,
            num_key_value_heads=checkpoint_summary.num_key_value_heads,
        )
    result = {
        "schema": STAGE5_STREAM_CHECKPOINT_SCHEMA,
        "status": "complete",
        "method_name": method_name,
        "settings": settings,
        "next_layer_index": total_layers_to_run,
        "hidden_state_path": str(hidden_state_path),
        "runtime_s": accumulated_runtime_s + (time.perf_counter() - stream_start),
        "intermediate_layer_metrics": intermediate_layer_metrics,
        "storage_modified_token_totals": storage_modified_token_totals,
        "storage_category_sums": storage_category_sums,
        "metric_records": records,
        "method_aggregate": aggregate,
        "passkey_metrics": passkey_metrics,
        "certificate_records": certificate_records,
    }
    _save_stream_checkpoint(output_dir, prompt.name, method_name, result)
    del hidden_states, method_logits, full_hidden, full_logits
    _release_memory()
    return result


def _prompt_method_metrics(
    *,
    prompt: PromptMaterialized,
    method_name: str,
    logits: torch.Tensor,
    full_logits: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    records: list[dict[str, Any]] = []
    token_ids = list(prompt.token_ids)
    start = STAGE5_SCORE_START
    end = len(token_ids) - 1
    for token_index in range(start, end):
        records.append(
            _metric_record_for_token(
                token_index=token_index,
                target_token_id=token_ids[token_index + 1],
                full_logits=full_logits[token_index].numpy().astype(np.float64),
                method_logits=logits[token_index].numpy().astype(np.float64),
            )
        )
    aggregate = _aggregate_metric_records(records)
    passkey_metrics = None
    if prompt.answer_token_ids is not None and prompt.final_answer_start_token_index is not None:
        start_index = int(prompt.final_answer_start_token_index)
        answer_ids = list(prompt.answer_token_ids)
        log_probs = [_log_softmax(logits[index - 1].numpy().astype(np.float64)) for index in range(start_index, start_index + len(answer_ids))]
        full_log_probs = [_log_softmax(full_logits[index - 1].numpy().astype(np.float64)) for index in range(start_index, start_index + len(answer_ids))]
        answer_log_prob = float(sum(log_probs[offset][answer_ids[offset]] for offset in range(len(answer_ids))))
        full_answer_log_prob = float(sum(full_log_probs[offset][answer_ids[offset]] for offset in range(len(answer_ids))))
        first_answer_token = answer_ids[0]
        first_answer_logits = logits[start_index - 1].numpy().astype(np.float64)
        passkey_metrics = {
            "answer_text": prompt.answer_text,
            "answer_token_ids": answer_ids,
            "teacher_forced_answer_log_probability": answer_log_prob,
            "teacher_forced_answer_log_probability_full_kv": full_answer_log_prob,
            "delta_answer_log_probability_vs_full": float(answer_log_prob - full_answer_log_prob),
            "first_answer_token_id": int(first_answer_token),
            "first_answer_token_rank": _rank_of_token(first_answer_logits, first_answer_token),
            "first_answer_token_top1": int(np.argmax(first_answer_logits)),
            "first_answer_token_is_top1": bool(int(np.argmax(first_answer_logits)) == int(first_answer_token)),
            "first_answer_token_in_top5": bool(int(first_answer_token) in _topk_ids(first_answer_logits, 5)),
        }
    return records, aggregate, passkey_metrics


def run_stage5_quality_smoke(
    *,
    output_dir: str | Path = STAGE5_DEFAULT_OUTPUT_DIR,
    review_zip_path: str | Path = STAGE5_DEFAULT_REVIEW_ZIP,
    capture_dir: str | Path = STAGE5_DEFAULT_CAPTURE_DIR,
    tensor_cache_dir: str | Path = STAGE5_DEFAULT_TENSOR_CACHE_DIR,
    repo_id: str = DEFAULT_LLAMA31_BASE_REPO,
    repo_revision: str = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f",
    recent_window: int = STAGE5_RECENT_WINDOW,
    block_size: int = STAGE5_BLOCK_SIZE,
    tolerance: float = STAGE5_TOLERANCE,
    precision: int = STAGE5_PRECISION,
    seed: int = STAGE5_SEED,
    allow_insecure_tls: bool = False,
    max_rss_bytes: int | None = None,
    min_free_bytes: int | None = None,
    prompt_names: Sequence[str] | None = None,
    method_names: Sequence[str] | None = None,
    prompt_token_count: int = STAGE5_PROMPT_TOKEN_COUNT,
    profile_progress_every_tokens: int | None = None,
    max_layer_index: int | None = None,
    run_full_kv_equivalence_regression: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    review_zip_path = Path(review_zip_path)
    capture_dir = Path(capture_dir)
    tensor_cache_dir = Path(tensor_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / STAGE5_CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    (output_dir / STAGE5_METRIC_DIR).mkdir(parents=True, exist_ok=True)
    (output_dir / STAGE5_CERTIFICATE_DIR).mkdir(parents=True, exist_ok=True)
    (output_dir / STAGE5_PROVENANCE_DIR).mkdir(parents=True, exist_ok=True)

    before_stage4 = _stage4_artifact_hashes(Path.cwd())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    download_policy = _DownloadPolicy(verify_tls=not allow_insecure_tls)
    asset_dir, asset_hashes, asset_sources = _prepare_local_assets_from_stage3(
        capture_dir=capture_dir,
        repo_id=repo_id,
        revision=repo_revision,
        download_policy=download_policy,
    )
    config, checkpoint_summary = _load_checkpoint_summary(
        asset_dir,
        repo_id=repo_id,
        repo_revision=repo_revision,
    )
    _validate_selected_heads(checkpoint_summary)
    weight_index, model_index_sha256, model_index_source = _fetch_weight_index(
        repo_id,
        repo_revision,
        asset_dir,
        download_policy=download_policy,
    )
    local_state = inspect_local_model_state(
        repo_id=repo_id,
        revision=repo_revision,
        capture_dir=capture_dir,
        tensor_cache_dir=tensor_cache_dir,
    )
    tensor_cache = PersistentTensorRangeCache(
        root=tensor_cache_dir,
        repo_id=repo_id,
        revision=repo_revision,
        download_policy=download_policy,
    )
    selected_methods = _normalize_stage5_methods(method_names)
    full_method_set_requested = tuple(selected_methods) == tuple(STAGE5_METHODS)
    prompts = build_stage5_prompt_corpus(
        asset_dir=asset_dir,
        output_dir=output_dir,
        revision=repo_revision,
        target_token_count=int(prompt_token_count),
        prompt_names=prompt_names,
    )
    total_layers_to_run = checkpoint_summary.num_hidden_layers if max_layer_index is None else min(checkpoint_summary.num_hidden_layers, int(max_layer_index) + 1)
    if total_layers_to_run <= 0:
        raise Stage5ExecutionError(f"Invalid max_layer_index for Stage 5 run: {max_layer_index}.")
    _write_provenance_json(
        output_dir,
        "local_model_state.json",
        {"schema": "stage5_local_model_state_v1", **local_state},
    )
    _write_provenance_json(
        output_dir,
        "asset_provenance.json",
        {
            "schema": "stage5_asset_provenance_v1",
            "repo_id": repo_id,
            "revision": repo_revision,
            "asset_hashes": asset_hashes,
            "asset_sources": asset_sources,
            "asset_dir": str(asset_dir),
        },
    )
    _write_provenance_json(
        output_dir,
        "model_index_provenance.json",
        {
            "schema": "stage5_model_index_provenance_v1",
            "repo_id": repo_id,
            "revision": repo_revision,
            "model_index_sha256": model_index_sha256,
            "model_index_source": model_index_source,
        },
    )
    full_kv_equivalence: dict[str, Any] | None = None
    if run_full_kv_equivalence_regression:
        regression_prompt_embeddings = _fetch_prompt_embeddings_with_cache(
            tensor_cache=tensor_cache,
            weight_index=weight_index,
            token_ids=prompts[0].token_ids,
        )
        full_kv_equivalence = _run_full_kv_custom_equivalence_regression(
            config=config,
            checkpoint_summary=checkpoint_summary,
            tensor_cache=tensor_cache,
            weight_index=weight_index,
            prompt_name=prompts[0].name,
            hidden_states=regression_prompt_embeddings[:16],
            recent_window=recent_window,
            block_size=block_size,
            precision=precision,
        )
        del regression_prompt_embeddings
        gc.collect()
        _write_provenance_json(
            output_dir,
            "full_kv_equivalence_regression.json",
            {"schema": "stage5_full_kv_equivalence_v1", **full_kv_equivalence},
        )

    if max_rss_bytes is None:
        max_rss_bytes = int(STAGE5_DEFAULT_MAX_RSS_GB * (1024 ** 3))
    if min_free_bytes is None:
        min_free_bytes = int(STAGE5_DEFAULT_MIN_FREE_GB * (1024 ** 3))
    guard = MemoryGuard(max_rss_bytes=max_rss_bytes, min_free_bytes=min_free_bytes)
    guard.check("before_stage5_smoke")

    prompt_results: list[Stage5PromptRunResult] = []
    start_time = time.perf_counter()
    memory_profile_records: list[dict[str, Any]] = []

    for prompt_index, prompt in enumerate(prompts):
        cached_checkpoint = _load_prompt_checkpoint(output_dir, prompt.name) if full_method_set_requested else None
        if cached_checkpoint is not None:
            prompt_results.append(
                Stage5PromptRunResult(
                    prompt=prompt,
                    scored_token_count=int(cached_checkpoint["scored_token_count"]),
                    prompt_runtime_s=float(cached_checkpoint["prompt_runtime_s"]),
                    prompt_peak_rss_bytes=int(cached_checkpoint["prompt_peak_rss_bytes"]),
                    method_aggregates=dict(cached_checkpoint["method_aggregates"]),
                    per_token_records_path=Path(cached_checkpoint["per_token_records_path"]),
                    certificate_records_path=Path(cached_checkpoint["certificate_records_path"]),
                    intermediate_layer_metrics=dict(cached_checkpoint["intermediate_layer_metrics"]),
                    passkey_metrics=cached_checkpoint.get("passkey_metrics"),
                )
            )
            continue

        prompt_start = time.perf_counter()
        full_checkpoint = _run_full_kv_prompt_stream(
            output_dir=output_dir,
            prompt=prompt,
            config=config,
            checkpoint_summary=checkpoint_summary,
            tensor_cache=tensor_cache,
            weight_index=weight_index,
            guard=guard,
            total_layers_to_run=total_layers_to_run,
            repo_id=repo_id,
            repo_revision=repo_revision,
            recent_window=recent_window,
            block_size=block_size,
            tolerance=tolerance,
            precision=precision,
            seed=seed,
            start_time=start_time,
            memory_profile_records=memory_profile_records,
        )
        method_checkpoints: dict[str, dict[str, Any]] = {"full_kv": full_checkpoint}
        for method_name in STAGE5_METHODS:
            if method_name == "full_kv" or method_name not in selected_methods:
                continue
            method_checkpoints[method_name] = _run_nonfull_prompt_stream(
                output_dir=output_dir,
                prompt=prompt,
                method_name=method_name,
                config=config,
                checkpoint_summary=checkpoint_summary,
                tensor_cache=tensor_cache,
                weight_index=weight_index,
                guard=guard,
                total_layers_to_run=total_layers_to_run,
                repo_id=repo_id,
                repo_revision=repo_revision,
                recent_window=recent_window,
                block_size=block_size,
                tolerance=tolerance,
                precision=precision,
                seed=seed,
                start_time=start_time,
                memory_profile_records=memory_profile_records,
                profile_progress_every_tokens=profile_progress_every_tokens,
            )

        method_aggregates = {
            method_name: dict(checkpoint_payload["method_aggregate"])
            for method_name, checkpoint_payload in method_checkpoints.items()
        }
        passkey_metrics_by_method = {
            method_name: checkpoint_payload["passkey_metrics"]
            for method_name, checkpoint_payload in method_checkpoints.items()
            if checkpoint_payload.get("passkey_metrics") is not None
        }
        per_token_records = {
            method_name: list(checkpoint_payload["metric_records"])
            for method_name, checkpoint_payload in method_checkpoints.items()
        }
        cert_records_for_prompt = list(method_checkpoints.get("rack_kv_certified", {}).get("certificate_records", []))
        intermediate_layer_metrics = {
            method_name: dict(checkpoint_payload.get("intermediate_layer_metrics", {}))
            for method_name, checkpoint_payload in method_checkpoints.items()
            if method_name != "full_kv"
        }

        metric_path = output_dir / STAGE5_METRIC_DIR / f"{prompt.name}.json"
        cert_path = output_dir / STAGE5_CERTIFICATE_DIR / f"{prompt.name}.json"
        _write_json_atomic(
            metric_path,
            {
                "schema": STAGE5_METRIC_RECORD_SCHEMA,
                "prompt_name": prompt.name,
                "records_by_method": per_token_records,
            },
        )
        _write_json_atomic(
            cert_path,
            {
                "schema": STAGE5_CERTIFICATE_RECORD_SCHEMA,
                "prompt_name": prompt.name,
                "records": cert_records_for_prompt,
            },
        )
        prompt_result = Stage5PromptRunResult(
            prompt=prompt,
            scored_token_count=len(per_token_records["full_kv"]),
            prompt_runtime_s=time.perf_counter() - prompt_start,
            prompt_peak_rss_bytes=guard.peak_rss_bytes,
            method_aggregates=method_aggregates,
            per_token_records_path=metric_path,
            certificate_records_path=cert_path,
            intermediate_layer_metrics=intermediate_layer_metrics,
            passkey_metrics=passkey_metrics_by_method or None,
        )
        prompt_results.append(prompt_result)
        if full_method_set_requested:
            _save_prompt_checkpoint(
                output_dir,
                prompt.name,
                {
                    "schema": STAGE5_PROMPT_CHECKPOINT_SCHEMA,
                    "prompt_name": prompt.name,
                    "scored_token_count": prompt_result.scored_token_count,
                    "prompt_runtime_s": prompt_result.prompt_runtime_s,
                    "prompt_peak_rss_bytes": prompt_result.prompt_peak_rss_bytes,
                    "method_aggregates": prompt_result.method_aggregates,
                    "per_token_records_path": str(prompt_result.per_token_records_path),
                    "certificate_records_path": str(prompt_result.certificate_records_path),
                    "intermediate_layer_metrics": prompt_result.intermediate_layer_metrics,
                    "passkey_metrics": prompt_result.passkey_metrics,
                },
            )
        gc.collect()
        guard.check(f"after_prompt_{prompt.name}")

    results = {
        "schema": STAGE5_RUN_SCHEMA,
        "result_version": STAGE5_RESULT_VERSION,
        "method_version": STAGE5_METHOD_VERSION,
        "repo_id": repo_id,
        "revision": repo_revision,
        "tls_verification": not allow_insecure_tls,
        "configuration": {
            "recent_window": recent_window,
            "block_size": block_size,
            "tolerance": tolerance,
            "precision": precision,
            "seed": seed,
            "modified_layers": list(STAGE5_MODIFIED_LAYERS),
            "executed_layers": list(range(total_layers_to_run)),
            "score_start_position": STAGE5_SCORE_START,
            "methods": list(STAGE5_METHODS),
            "selected_methods": list(selected_methods),
            "prompt_token_count": int(prompt_token_count),
        },
        "local_model_state": local_state,
        "asset_hashes": asset_hashes,
        "asset_sources": asset_sources,
        "model_index_sha256": model_index_sha256,
        "model_index_source": model_index_source,
        "checkpoint_summary": asdict(checkpoint_summary),
        "dependency_versions": stage5_dependency_versions(),
        "full_kv_equivalence_regression": full_kv_equivalence,
        "prompts": [
            {
                "name": result.prompt.name,
                "source_sha256": result.prompt.source_sha256,
                "token_count": result.prompt.token_count,
                "token_ids_path": str(result.prompt.token_ids_path),
                "source_path": str(result.prompt.source_path),
                "metadata_path": str(result.prompt.metadata_path),
                "scored_token_count": result.scored_token_count,
                "runtime_s": result.prompt_runtime_s,
                "peak_rss_bytes": result.prompt_peak_rss_bytes,
                "method_aggregates": result.method_aggregates,
                "intermediate_layer_metrics": result.intermediate_layer_metrics,
                "passkey_metrics": result.passkey_metrics,
            }
            for result in prompt_results
        ],
        "tensor_cache_provenance": tensor_cache.provenance_summary(),
        "memory_profile_records": memory_profile_records,
        "memory_guard": {
            "start_rss_bytes": guard.start_rss_bytes,
            "peak_rss_bytes": guard.peak_rss_bytes,
            "end_rss_bytes": guard.end_rss_bytes,
            "min_available_bytes": guard.min_available_bytes,
            "max_rss_bytes": guard.max_rss_bytes,
            "min_free_bytes": guard.min_free_bytes,
        },
        "runtime_s": time.perf_counter() - start_time,
        "scientific_scope": {
            "teacher_forced_only": "This is a teacher-forced quality smoke, not yet the full task-quality benchmark.",
            "representative_layers_only": "Only layers 0, 8, 16, 24, and 31 use modified KV methods; all other layers use Full KV within each stream.",
            "all_attention_heads_processed": "All 32 query heads and all 8 KV heads are processed in each modified layer.",
            "compression_scope": "Compression error remains empirical.",
            "certificate_scope": "Only skipping relative to reconstructed compressed KV is rigorously certified.",
            "runtime_scope": "CPU offline runtime is experimental and must not be interpreted as production inference latency or speedup.",
        },
    }
    _write_provenance_json(
        output_dir,
        "tensor_cache_provenance.json",
        {"schema": "stage5_tensor_cache_provenance_v1", **tensor_cache.provenance_summary()},
    )
    _write_provenance_json(
        output_dir,
        "memory_profile.json",
        {
            "schema": "stage5_memory_profile_v1",
            "records": memory_profile_records,
            "memory_guard": {
                "start_rss_bytes": guard.start_rss_bytes,
                "peak_rss_bytes": guard.peak_rss_bytes,
                "end_rss_bytes": guard.end_rss_bytes,
                "min_available_bytes": guard.min_available_bytes,
                "max_rss_bytes": guard.max_rss_bytes,
                "min_free_bytes": guard.min_free_bytes,
            },
        },
    )
    results_path = output_dir / STAGE5_RESULTS_JSON
    _write_json_atomic(results_path, results)
    after_stage4 = _stage4_artifact_hashes(Path.cwd())
    _validate_stage4_artifacts_unchanged(before_stage4, after_stage4)
    return results
