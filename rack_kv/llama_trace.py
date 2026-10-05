from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Sequence
from urllib.parse import quote

import gmpy2
import numpy as np
import psutil
import requests
import safetensors
import torch
import transformers
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    create_causal_mask,
    eager_attention_forward,
)

DEFAULT_LLAMA31_BASE_REPO = "NousResearch/Meta-Llama-3.1-8B"
DEFAULT_PROMPT = "RACK-KV trace prompt for offline attention replay."
DEFAULT_TRACE_FILENAME = "llama31_layer0_trace.safetensors"
DEFAULT_REPORT_FILENAME = "capture_report.json"
DEFAULT_DEPENDENCY_REPORT_FILENAME = "dependency_versions.json"
DEFAULT_ASSET_FILENAMES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
LAYER_PARAMETER_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)
TRACE_SCHEMA_COMPACT_V1 = "compact_final_cache_v1"
_SAFETENSOR_NUMPY_DTYPES: dict[str, tuple[np.dtype[Any], int]] = {
    "BOOL": (np.dtype(np.bool_), 1),
    "U8": (np.dtype(np.uint8), 1),
    "I8": (np.dtype(np.int8), 1),
    "I16": (np.dtype(np.int16), 2),
    "I32": (np.dtype(np.int32), 4),
    "I64": (np.dtype(np.int64), 8),
    "F16": (np.dtype(np.float16), 2),
    "F32": (np.dtype(np.float32), 4),
    "F64": (np.dtype(np.float64), 8),
    "BF16": (np.dtype(np.uint16), 2),
}
_SAFETENSOR_TORCH_DTYPES: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "BF16": torch.bfloat16,
}


@dataclass(frozen=True)
class CheckpointConfigSummary:
    repo_id: str
    repo_revision: str
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    model_dtype: str
    rope_theta: float
    rope_scaling: dict[str, Any] | None
    max_position_embeddings: int


@dataclass(frozen=True)
class MinimalTraceCaptureResult:
    checkpoint: CheckpointConfigSummary
    trace_path: Path
    capture_report_path: Path
    dependency_report_path: Path
    prompt: str
    token_ids: tuple[int, ...]
    layer_index: int
    selected_query_heads: tuple[int, ...]
    selected_kv_heads: tuple[int, ...]
    query_to_kv_heads: tuple[int, ...]
    queries_shape: tuple[int, ...]
    final_keys_shape: tuple[int, ...]
    final_values_shape: tuple[int, ...]
    model_head_outputs_shape: tuple[int, ...]
    source_dtype: str
    storage_dtype: str
    tls_verification: bool
    trace_schema: str
    max_roundtrip_abs_diff: float
    max_compact_replay_abs_diff: float
    compact_replay_tolerance: float
    stock_projected_output_max_abs_diff: float
    stock_projected_output_max_rel_diff: float
    stock_decoder_output_max_abs_diff: float
    stock_decoder_output_max_rel_diff: float
    stock_cache_key_max_abs_diff: float
    stock_cache_key_max_rel_diff: float
    stock_cache_value_max_abs_diff: float
    stock_cache_value_max_rel_diff: float
    peak_rss_bytes: int


@dataclass(frozen=True)
class LocalAssetPreparationResult:
    repo_id: str
    repo_revision: str
    asset_dir: Path
    asset_hashes: dict[str, str]
    asset_sources: dict[str, str]
    tls_verification: bool
    repo_revision_source: str


@dataclass
class _DownloadPolicy:
    verify_tls: bool
    warning_emitted: bool = False


@dataclass(frozen=True)
class _RemoteTensorSpec:
    dtype_code: str
    shape: tuple[int, ...]
    byte_start: int
    byte_end_exclusive: int

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def itemsize(self) -> int:
        return _SAFETENSOR_NUMPY_DTYPES[self.dtype_code][1]


@dataclass(frozen=True)
class _LayerTraceRecord:
    position: int
    token_id: int
    queries: torch.Tensor
    model_head_outputs: torch.Tensor
    projected_output: torch.Tensor
    decoder_output: torch.Tensor


@dataclass(frozen=True)
class _LayerTraceRun:
    records: tuple[_LayerTraceRecord, ...]
    final_keys: torch.Tensor
    final_values: torch.Tensor
    next_hidden_states: torch.Tensor


@dataclass(frozen=True)
class _LayerStockRun:
    projected_outputs: tuple[torch.Tensor, ...]
    decoder_outputs: tuple[torch.Tensor, ...]
    final_keys: torch.Tensor
    final_values: torch.Tensor
    next_hidden_states: torch.Tensor


@dataclass(frozen=True)
class PerLayerTraceCaptureResult:
    layer_index: int
    trace_path: Path
    trace_sha256: str
    queries_shape: tuple[int, ...]
    final_keys_shape: tuple[int, ...]
    final_values_shape: tuple[int, ...]
    model_head_outputs_shape: tuple[int, ...]
    source_dtype: str
    storage_dtype: str
    max_roundtrip_abs_diff: float
    max_compact_replay_abs_diff: float
    compact_replay_tolerance: float
    stock_projected_output_max_abs_diff: float
    stock_projected_output_max_rel_diff: float
    stock_decoder_output_max_abs_diff: float
    stock_decoder_output_max_rel_diff: float
    stock_cache_key_max_abs_diff: float
    stock_cache_key_max_rel_diff: float
    stock_cache_value_max_abs_diff: float
    stock_cache_value_max_rel_diff: float


@dataclass(frozen=True)
class MultiLayerTraceCaptureResult:
    checkpoint: CheckpointConfigSummary
    capture_report_path: Path
    dependency_report_path: Path
    prompt: str
    token_ids: tuple[int, ...]
    selected_layer_indices: tuple[int, ...]
    selected_query_heads: tuple[int, ...]
    selected_kv_heads: tuple[int, ...]
    query_to_kv_heads: tuple[int, ...]
    tls_verification: bool
    layer_results: tuple[PerLayerTraceCaptureResult, ...]
    peak_rss_bytes: int


class _RemoteSafetensorsShard:
    def __init__(
        self,
        *,
        url: str,
        tensor_specs: dict[str, _RemoteTensorSpec],
        download_policy: _DownloadPolicy,
        fetched_tensor_hashes: dict[str, str],
    ) -> None:
        self.url = url
        self.tensor_specs = tensor_specs
        self.download_policy = download_policy
        self.fetched_tensor_hashes = fetched_tensor_hashes

    @classmethod
    def from_repo(
        cls,
        *,
        repo_id: str,
        revision: str,
        filename: str,
        download_policy: _DownloadPolicy,
        fetched_tensor_hashes: dict[str, str],
        timeout_s: float = 120.0,
    ) -> _RemoteSafetensorsShard:
        url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
        header_prefix = _range_get(url, 0, 7, timeout_s=timeout_s, download_policy=download_policy)
        if len(header_prefix) != 8:
            raise ValueError("Failed to read the safetensors header length.")
        header_length = int.from_bytes(header_prefix, byteorder="little", signed=False)
        header_bytes = _range_get(url, 8, 8 + header_length - 1, timeout_s=timeout_s, download_policy=download_policy)
        raw_header = json.loads(header_bytes.decode("utf-8"))
        tensor_specs: dict[str, _RemoteTensorSpec] = {}
        data_start = 8 + header_length
        for tensor_name, entry in raw_header.items():
            if tensor_name == "__metadata__":
                continue
            offsets = entry["data_offsets"]
            tensor_specs[tensor_name] = _RemoteTensorSpec(
                dtype_code=entry["dtype"],
                shape=tuple(int(dim) for dim in entry["shape"]),
                byte_start=data_start + int(offsets[0]),
                byte_end_exclusive=data_start + int(offsets[1]),
            )
        return cls(
            url=url,
            tensor_specs=tensor_specs,
            download_policy=download_policy,
            fetched_tensor_hashes=fetched_tensor_hashes,
        )

    def tensor_spec(self, tensor_name: str) -> _RemoteTensorSpec:
        try:
            return self.tensor_specs[tensor_name]
        except KeyError as error:
            raise KeyError(f"Tensor {tensor_name!r} not found in remote shard {self.url!r}.") from error

    def fetch_tensor(self, tensor_name: str) -> torch.Tensor:
        spec = self.tensor_spec(tensor_name)
        payload = _range_get(
            self.url,
            spec.byte_start,
            spec.byte_end_exclusive - 1,
            download_policy=self.download_policy,
        )
        self.fetched_tensor_hashes[tensor_name] = _sha256_bytes(payload)
        return _decode_tensor_bytes(payload, spec)

    def fetch_tensors(
        self,
        tensor_names: Sequence[str],
        *,
        max_gap_bytes: int = 8 * 1024 * 1024,
    ) -> dict[str, torch.Tensor]:
        if not tensor_names:
            raise ValueError("tensor_names must be non-empty.")
        ordered_specs = [
            (tensor_name, self.tensor_spec(tensor_name))
            for tensor_name in tensor_names
        ]
        ordered_specs.sort(key=lambda item: item[1].byte_start)
        groups: list[list[tuple[str, _RemoteTensorSpec]]] = []
        current_group: list[tuple[str, _RemoteTensorSpec]] = []
        current_end = -1
        for tensor_name, spec in ordered_specs:
            if not current_group:
                current_group = [(tensor_name, spec)]
                current_end = spec.byte_end_exclusive
                continue
            if spec.byte_start - current_end <= max_gap_bytes:
                current_group.append((tensor_name, spec))
                current_end = max(current_end, spec.byte_end_exclusive)
            else:
                groups.append(current_group)
                current_group = [(tensor_name, spec)]
                current_end = spec.byte_end_exclusive
        if current_group:
            groups.append(current_group)

        tensors: dict[str, torch.Tensor] = {}
        for group in groups:
            group_start = min(spec.byte_start for _, spec in group)
            group_end_exclusive = max(spec.byte_end_exclusive for _, spec in group)
            payload = _range_get(
                self.url,
                group_start,
                group_end_exclusive - 1,
                download_policy=self.download_policy,
            )
            for tensor_name, spec in group:
                local_start = spec.byte_start - group_start
                local_end = spec.byte_end_exclusive - group_start
                tensor_payload = payload[local_start:local_end]
                self.fetched_tensor_hashes[tensor_name] = _sha256_bytes(tensor_payload)
                tensors[tensor_name] = _decode_tensor_bytes(tensor_payload, spec)
        return tensors

    def fetch_rows(self, tensor_name: str, rows: Iterable[int]) -> torch.Tensor:
        spec = self.tensor_spec(tensor_name)
        if len(spec.shape) != 2:
            raise ValueError("Row slicing is only implemented for rank-2 tensors.")
        row_indices = list(rows)
        if not row_indices:
            raise ValueError("At least one row index is required.")
        row_width = spec.shape[1]
        row_bytes = row_width * spec.itemsize
        assembled_rows: list[torch.Tensor] = []
        for row_index in row_indices:
            if row_index < 0 or row_index >= spec.shape[0]:
                raise IndexError(f"Row index {row_index} is outside tensor shape {spec.shape}.")
            start = spec.byte_start + row_index * row_bytes
            end = start + row_bytes - 1
            row_spec = _RemoteTensorSpec(
                dtype_code=spec.dtype_code,
                shape=(1, row_width),
                byte_start=start,
                byte_end_exclusive=end + 1,
            )
            payload = _range_get(self.url, start, end, download_policy=self.download_policy)
            self.fetched_tensor_hashes[f"{tensor_name}[row={row_index}]"] = _sha256_bytes(payload)
            assembled_rows.append(_decode_tensor_bytes(payload, row_spec))
        return torch.cat(assembled_rows, dim=0)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _mark_insecure_warning(download_policy: _DownloadPolicy) -> _DownloadPolicy:
    if not download_policy.verify_tls and not download_policy.warning_emitted:
        print(
            "WARNING: TLS verification is disabled for this capture run. "
            "Downloaded assets are hashed and the run is marked tls_verification=false.",
            file=sys.stderr,
        )
        download_policy.warning_emitted = True
    return download_policy


def _http_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = 120.0,
    download_policy: _DownloadPolicy,
) -> tuple[requests.Response, _DownloadPolicy]:
    updated_policy = _mark_insecure_warning(download_policy)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                headers=headers,
                verify=updated_policy.verify_tls,
                timeout=timeout_s,
            )
            response.raise_for_status()
            return response, updated_policy
        except requests.exceptions.RequestException as error:
            last_error = error
            if attempt == 2:
                break
            time.sleep(1.0 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _range_get(
    url: str,
    start: int,
    end: int,
    *,
    timeout_s: float = 120.0,
    download_policy: _DownloadPolicy,
) -> bytes:
    response, _ = _http_get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        timeout_s=timeout_s,
        download_policy=download_policy,
    )
    expected_length = end - start + 1
    if len(response.content) != expected_length:
        raise ValueError(f"Expected {expected_length} bytes from {url!r}, received {len(response.content)}.")
    return response.content


def _decode_tensor_bytes(payload: bytes, spec: _RemoteTensorSpec) -> torch.Tensor:
    numpy_dtype = _SAFETENSOR_NUMPY_DTYPES[spec.dtype_code][0]
    flat = np.frombuffer(payload, dtype=numpy_dtype).copy()
    if flat.size != spec.numel:
        raise ValueError(f"Decoded {flat.size} elements but expected {spec.numel} for tensor with shape {spec.shape}.")
    tensor = torch.from_numpy(flat)
    if spec.dtype_code == "BF16":
        tensor = tensor.view(torch.bfloat16)
    else:
        tensor = tensor.to(dtype=_SAFETENSOR_TORCH_DTYPES[spec.dtype_code])
    return tensor.reshape(spec.shape)


def query_head_to_kv_head(query_head_index: int, *, num_attention_heads: int, num_key_value_heads: int) -> int:
    if query_head_index < 0 or query_head_index >= num_attention_heads:
        raise IndexError("query_head_index is outside the configured query-head range.")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
    return query_head_index // (num_attention_heads // num_key_value_heads)


def _default_capture_query_heads(*, num_attention_heads: int, num_key_value_heads: int) -> tuple[int, ...]:
    if num_attention_heads <= 0 or num_key_value_heads <= 0:
        raise ValueError("Head counts must be positive.")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
    group_size = num_attention_heads // num_key_value_heads
    proposed = [0]
    if num_attention_heads > 1 and group_size > 1:
        proposed.append(1)
    if num_key_value_heads > 1:
        proposed.append(group_size)
    for candidate in range(num_attention_heads):
        if candidate not in proposed:
            proposed.append(candidate)
        if len(proposed) >= min(3, num_attention_heads):
            break
    unique_in_order: list[int] = []
    for candidate in proposed:
        if 0 <= candidate < num_attention_heads and candidate not in unique_in_order:
            unique_in_order.append(candidate)
    return tuple(unique_in_order)


def _validate_selected_head_configuration(
    *,
    selected_query_heads: tuple[int, ...],
    query_to_kv_heads: tuple[int, ...],
    selected_kv_heads: tuple[int, ...],
    num_attention_heads: int,
    num_key_value_heads: int,
) -> None:
    if not selected_query_heads:
        raise ValueError("selected_query_heads must be non-empty.")
    if len(set(selected_query_heads)) != len(selected_query_heads):
        raise ValueError("selected_query_heads must be unique.")
    if len(set(selected_kv_heads)) != len(selected_kv_heads):
        raise ValueError("selected_kv_heads must be unique.")
    if len(query_to_kv_heads) != len(selected_query_heads):
        raise ValueError("query_to_kv_heads length must match selected_query_heads length.")
    if any(head < 0 or head >= num_attention_heads for head in selected_query_heads):
        raise ValueError("selected_query_heads must stay within the configured query-head range.")
    if any(head < 0 or head >= num_key_value_heads for head in selected_kv_heads):
        raise ValueError("selected_kv_heads must stay within the configured KV-head range.")
    for query_head, mapped_kv_head in zip(selected_query_heads, query_to_kv_heads):
        recomputed = query_head_to_kv_head(
            query_head_index=query_head,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
        if mapped_kv_head != recomputed:
            raise ValueError("Stored query_to_kv_heads mapping does not match the configured GQA geometry.")
        if mapped_kv_head not in selected_kv_heads:
            raise ValueError("Every mapped KV head must be included in selected_kv_heads.")


def _validate_reloaded_trace_heads(
    *,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    tensor_selected_query_heads = [int(value) for value in tensors["selected_query_heads"].tolist()]
    tensor_selected_kv_heads = [int(value) for value in tensors["selected_kv_heads"].tolist()]
    metadata_selected_query_heads = [int(value) for value in metadata["selected_query_heads"]]
    metadata_selected_kv_heads = [int(value) for value in metadata["selected_kv_heads"]]
    query_to_kv_heads = tuple(int(value) for value in metadata["query_to_kv_heads"])
    num_attention_heads = int(metadata["num_attention_heads"])
    num_key_value_heads = int(metadata["num_key_value_heads"])

    if tensor_selected_query_heads != metadata_selected_query_heads:
        raise ValueError("Reload validation failed: tensor selected_query_heads does not match metadata selected_query_heads.")
    if tensor_selected_kv_heads != metadata_selected_kv_heads:
        raise ValueError("Reload validation failed: tensor selected_kv_heads does not match metadata selected_kv_heads.")

    _validate_selected_head_configuration(
        selected_query_heads=tuple(tensor_selected_query_heads),
        query_to_kv_heads=query_to_kv_heads,
        selected_kv_heads=tuple(tensor_selected_kv_heads),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )


def _fetch_repo_revision(repo_id: str, *, download_policy: _DownloadPolicy) -> str:
    api_url = f"https://huggingface.co/api/models/{quote(repo_id, safe='/')}"
    response, _ = _http_get(api_url, download_policy=download_policy)
    payload = response.json()
    sha = payload.get("sha")
    if not isinstance(sha, str) or not sha:
        raise ValueError(f"Failed to resolve an immutable revision for {repo_id!r}.")
    return sha


def _download_text_file(
    repo_id: str,
    revision: str,
    filename: str,
    destination: Path,
    *,
    download_policy: _DownloadPolicy,
) -> tuple[Path, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination, "cache"
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    response, _ = _http_get(url, download_policy=download_policy)
    destination.write_bytes(response.content)
    return destination, "network"


def _prepare_local_assets(
    repo_id: str,
    revision: str,
    asset_dir: Path,
    *,
    download_policy: _DownloadPolicy,
) -> tuple[dict[str, str], dict[str, str]]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    asset_hashes: dict[str, str] = {}
    asset_sources: dict[str, str] = {}
    for filename in DEFAULT_ASSET_FILENAMES:
        try:
            path, source = _download_text_file(
                repo_id,
                revision,
                filename,
                asset_dir / filename,
                download_policy=download_policy,
            )
        except requests.HTTPError:
            if filename == "special_tokens_map.json":
                continue
            raise
        asset_hashes[filename] = _sha256_file(path)
        asset_sources[filename] = source
    return asset_hashes, asset_sources


def _load_checkpoint_summary(
    asset_dir: Path,
    *,
    repo_id: str,
    repo_revision: str,
) -> tuple[Any, CheckpointConfigSummary]:
    config = AutoConfig.from_pretrained(asset_dir, local_files_only=True)
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        rope_scaling = getattr(config, "rope_parameters", None)
    rope_theta = getattr(config, "rope_theta", None)
    if (rope_theta is None or float(rope_theta) == 0.0) and isinstance(rope_scaling, dict):
        rope_theta = rope_scaling.get("rope_theta", rope_scaling.get("original_max_position_embeddings", 0.0))
    summary = CheckpointConfigSummary(
        repo_id=repo_id,
        repo_revision=repo_revision,
        num_hidden_layers=int(config.num_hidden_layers),
        num_attention_heads=int(config.num_attention_heads),
        num_key_value_heads=int(config.num_key_value_heads),
        head_dim=int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
        hidden_size=int(config.hidden_size),
        intermediate_size=int(config.intermediate_size),
        model_dtype=str(getattr(config, "torch_dtype", getattr(config, "dtype", "unknown"))),
        rope_theta=float(rope_theta or 0.0),
        rope_scaling=dict(rope_scaling) if rope_scaling is not None else None,
        max_position_embeddings=int(getattr(config, "max_position_embeddings", 0)),
    )
    return config, summary


def _tokenize_prompt(asset_dir: Path, prompt: str) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(asset_dir, local_files_only=True)
    encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    return encoded["input_ids"][0].to(dtype=torch.long)


def _fetch_weight_index(
    repo_id: str,
    revision: str,
    asset_dir: Path,
    *,
    download_policy: _DownloadPolicy,
) -> tuple[dict[str, Any], str, str]:
    index_path, index_source = _download_text_file(
        repo_id,
        revision,
        "model.safetensors.index.json",
        asset_dir / "model.safetensors.index.json",
        download_policy=download_policy,
    )
    return json.loads(index_path.read_text(encoding="utf-8")), _sha256_file(index_path), index_source


def _fetch_layer_state_dict(
    *,
    repo_id: str,
    revision: str,
    weight_index: dict[str, Any],
    download_policy: _DownloadPolicy,
    fetched_tensor_hashes: dict[str, str],
    layer_index: int,
    shard_cache: dict[str, _RemoteSafetensorsShard] | None = None,
) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_index}."
    full_names = [prefix + suffix for suffix in LAYER_PARAMETER_SUFFIXES]
    names_by_shard: dict[str, list[str]] = {}
    for full_name in full_names:
        shard_name = weight_index["weight_map"][full_name]
        names_by_shard.setdefault(shard_name, []).append(full_name)

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name, shard_names in names_by_shard.items():
        shard = shard_cache.get(shard_name) if shard_cache is not None else None
        if shard is None:
            shard = _RemoteSafetensorsShard.from_repo(
                repo_id=repo_id,
                revision=revision,
                filename=shard_name,
                download_policy=download_policy,
                fetched_tensor_hashes=fetched_tensor_hashes,
            )
            if shard_cache is not None:
                shard_cache[shard_name] = shard
        fetched = shard.fetch_tensors(shard_names, max_gap_bytes=512 * 1024 * 1024)
        for full_name, tensor in fetched.items():
            state_dict[full_name[len(prefix):]] = tensor
    return state_dict


def _remote_shard_for_tensor(
    *,
    repo_id: str,
    revision: str,
    weight_index: dict[str, Any],
    tensor_name: str,
    download_policy: _DownloadPolicy,
    fetched_tensor_hashes: dict[str, str],
    shard_cache: dict[str, _RemoteSafetensorsShard] | None = None,
) -> _RemoteSafetensorsShard:
    shard_name = weight_index["weight_map"][tensor_name]
    if shard_cache is not None:
        cached = shard_cache.get(shard_name)
        if cached is not None:
            return cached
    shard = _RemoteSafetensorsShard.from_repo(
        repo_id=repo_id,
        revision=revision,
        filename=shard_name,
        download_policy=download_policy,
        fetched_tensor_hashes=fetched_tensor_hashes,
    )
    if shard_cache is not None:
        shard_cache[shard_name] = shard
    return shard


def _fetch_prompt_embeddings(
    shard: _RemoteSafetensorsShard,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    token_id_list = [int(token_id) for token_id in token_ids.tolist()]
    unique_token_ids = sorted(set(token_id_list))
    fetched_rows = shard.fetch_rows("model.embed_tokens.weight", unique_token_ids)
    row_lookup = {token_id: fetched_rows[index] for index, token_id in enumerate(unique_token_ids)}
    stacked = torch.stack([row_lookup[token_id] for token_id in token_id_list], dim=0)
    prompt_bytes = stacked.contiguous().view(torch.uint16).numpy().tobytes()
    shard.fetched_tensor_hashes["prompt_embeddings/combined"] = _sha256_bytes(prompt_bytes)
    return stacked


def _fetch_prompt_embeddings_from_weight_index(
    *,
    repo_id: str,
    revision: str,
    weight_index: dict[str, Any],
    token_ids: torch.Tensor,
    download_policy: _DownloadPolicy,
    fetched_tensor_hashes: dict[str, str],
    shard_cache: dict[str, _RemoteSafetensorsShard] | None = None,
) -> torch.Tensor:
    embedding_shard = _remote_shard_for_tensor(
        repo_id=repo_id,
        revision=revision,
        weight_index=weight_index,
        tensor_name="model.embed_tokens.weight",
        download_policy=download_policy,
        fetched_tensor_hashes=fetched_tensor_hashes,
        shard_cache=shard_cache,
    )
    return _fetch_prompt_embeddings(embedding_shard, token_ids)


def _load_decoder_layer_from_state_dict(config: Any, *, layer_index: int, state_dict: dict[str, torch.Tensor]) -> LlamaDecoderLayer:
    config._attn_implementation = "eager"
    layer = LlamaDecoderLayer(config, layer_idx=layer_index).to(dtype=torch.bfloat16)
    missing, unexpected = layer.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"Layer {layer_index} state loading mismatch. missing={missing} unexpected={unexpected}")
    layer.eval()
    return layer


def _manual_decoder_layer_attention_step(
    module: Any,
    hidden_states: torch.Tensor,
    *,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    cache: DynamicCache,
    selected_query_heads: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states, value_states = cache.update(key_states, value_states, module.layer_idx)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(module.config._attn_implementation, eager_attention_forward)
    attn_output_heads, _ = attention_interface(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=module.scaling,
    )
    attn_output = attn_output_heads.reshape(*input_shape, -1).contiguous()
    projected_output = module.o_proj(attn_output)
    selected_queries = query_states[0, list(selected_query_heads), -1, :].detach().cpu().clone()
    selected_head_outputs = attn_output_heads[0, -1, list(selected_query_heads), :].detach().cpu().clone()
    return projected_output, selected_queries, selected_head_outputs


def _run_manual_decoder_layer_trace(
    *,
    config: Any,
    layer: LlamaDecoderLayer,
    input_hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    selected_query_heads: tuple[int, ...],
) -> _LayerTraceRun:
    rotary_emb = LlamaRotaryEmbedding(config)
    cache = DynamicCache(config=config)
    records: list[_LayerTraceRecord] = []
    next_hidden_states: list[torch.Tensor] = []
    with torch.no_grad():
        for position, token_id in enumerate(input_ids.tolist()):
            hidden_states = input_hidden_states[position].unsqueeze(0).unsqueeze(0)
            position_ids = torch.tensor([[position]], dtype=torch.long)
            attention_mask = create_causal_mask(
                config=config,
                inputs_embeds=hidden_states,
                attention_mask=None,
                past_key_values=cache,
                position_ids=position_ids,
            )
            position_embeddings = rotary_emb(hidden_states, position_ids=position_ids)

            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            projected_output, selected_queries, selected_head_outputs = _manual_decoder_layer_attention_step(
                layer.self_attn,
                normalized,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                cache=cache,
                selected_query_heads=selected_query_heads,
            )
            hidden_states = residual + projected_output
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            decoder_output = residual + hidden_states
            next_hidden_states.append(decoder_output[0, 0].detach().cpu().clone())

            records.append(
                _LayerTraceRecord(
                    position=position,
                    token_id=token_id,
                    queries=selected_queries,
                    model_head_outputs=selected_head_outputs,
                    projected_output=projected_output[0, 0].detach().cpu().clone(),
                    decoder_output=decoder_output[0, 0].detach().cpu().clone(),
                )
            )
    final_layer_cache = cache.layers[layer.self_attn.layer_idx]
    return _LayerTraceRun(
        records=tuple(records),
        final_keys=final_layer_cache.keys[0].detach().cpu().clone(),
        final_values=final_layer_cache.values[0].detach().cpu().clone(),
        next_hidden_states=torch.stack(next_hidden_states, dim=0).contiguous(),
    )


def _run_stock_decoder_layer_reference(
    *,
    config: Any,
    layer: LlamaDecoderLayer,
    input_hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
) -> _LayerStockRun:
    rotary_emb = LlamaRotaryEmbedding(config)
    cache = DynamicCache(config=config)
    projected_outputs: list[torch.Tensor] = []
    decoder_outputs: list[torch.Tensor] = []

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        projected_outputs.append(output[0][0, 0].detach().cpu().clone())

    hook_handle = layer.self_attn.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for position, _token_id in enumerate(input_ids.tolist()):
                hidden_states = input_hidden_states[position].unsqueeze(0).unsqueeze(0)
                position_ids = torch.tensor([[position]], dtype=torch.long)
                attention_mask = create_causal_mask(
                    config=config,
                    inputs_embeds=hidden_states,
                    attention_mask=None,
                    past_key_values=cache,
                    position_ids=position_ids,
                )
                position_embeddings = rotary_emb(hidden_states, position_ids=position_ids)
                decoder_output = layer.forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )
                decoder_outputs.append(decoder_output[0, 0].detach().cpu().clone())
    finally:
        hook_handle.remove()

    final_layer_cache = cache.layers[layer.self_attn.layer_idx]
    return _LayerStockRun(
        projected_outputs=tuple(projected_outputs),
        decoder_outputs=tuple(decoder_outputs),
        final_keys=final_layer_cache.keys[0].detach().cpu().clone(),
        final_values=final_layer_cache.values[0].detach().cpu().clone(),
        next_hidden_states=torch.stack(list(decoder_outputs), dim=0).contiguous(),
    )


def _run_stock_decoder_layer_full_sequence(
    *,
    config: Any,
    layer: LlamaDecoderLayer,
    input_hidden_states: torch.Tensor,
) -> torch.Tensor:
    rotary_emb = LlamaRotaryEmbedding(config)
    with torch.no_grad():
        hidden_states = input_hidden_states.unsqueeze(0)
        position_ids = torch.arange(input_hidden_states.shape[0], dtype=torch.long).unsqueeze(0)
        attention_mask = create_causal_mask(
            config=config,
            inputs_embeds=hidden_states,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = rotary_emb(hidden_states, position_ids=position_ids)
        decoder_output = layer.forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
    return decoder_output[0].detach().cpu().squeeze(0).contiguous()


def _build_compact_trace_tensors(
    manual_run: _LayerTraceRun,
    *,
    selected_query_heads: tuple[int, ...],
    selected_kv_heads: tuple[int, ...],
    layer_index: int,
) -> dict[str, torch.Tensor]:
    if not manual_run.records:
        raise ValueError("At least one attention capture is required.")
    queries = torch.stack([record.queries for record in manual_run.records], dim=0)
    model_head_outputs = torch.stack([record.model_head_outputs for record in manual_run.records], dim=0)
    visible_lengths = torch.tensor([record.position + 1 for record in manual_run.records], dtype=torch.int64)
    token_ids = torch.tensor([record.token_id for record in manual_run.records], dtype=torch.int64)
    token_positions = torch.tensor([record.position for record in manual_run.records], dtype=torch.int64)
    layer_indices = torch.full((len(manual_run.records),), layer_index, dtype=torch.int64)
    final_keys = manual_run.final_keys[list(selected_kv_heads), :, :].contiguous()
    final_values = manual_run.final_values[list(selected_kv_heads), :, :].contiguous()
    return {
        "queries": queries.contiguous(),
        "final_keys": final_keys,
        "final_values": final_values,
        "model_head_outputs": model_head_outputs.contiguous(),
        "visible_lengths": visible_lengths,
        "record_token_ids": token_ids,
        "record_token_positions": token_positions,
        "record_layer_indices": layer_indices,
        "selected_query_heads": torch.tensor(selected_query_heads, dtype=torch.int64),
        "selected_kv_heads": torch.tensor(selected_kv_heads, dtype=torch.int64),
    }


def save_attention_trace(trace_path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> Path:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_strings = {key: json.dumps(value, sort_keys=True) if not isinstance(value, str) else value for key, value in metadata.items()}
    save_safetensors_file(tensors, str(trace_path), metadata=metadata_strings)
    return trace_path


def load_attention_trace(trace_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors = load_safetensors_file(str(trace_path))
    from safetensors import safe_open

    parsed_metadata: dict[str, Any] = {}
    with safe_open(str(trace_path), framework="pt") as handle:
        for key, value in handle.metadata().items():
            try:
                parsed_metadata[key] = json.loads(value)
            except json.JSONDecodeError:
                parsed_metadata[key] = value
    return tensors, parsed_metadata


def replay_head_attention_output(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    scaling: float,
) -> torch.Tensor:
    scores = torch.matmul(query.unsqueeze(0), keys.transpose(0, 1)) * scaling
    weights = torch.softmax(scores.to(torch.float32), dim=-1).to(query.dtype)
    return torch.matmul(weights, values).squeeze(0)


def replay_compact_trace_outputs(
    tensors: dict[str, torch.Tensor],
    *,
    query_to_kv_heads: tuple[int, ...],
    selected_kv_heads: tuple[int, ...],
    scaling: float,
) -> torch.Tensor:
    kv_lookup = {kv_head: index for index, kv_head in enumerate(selected_kv_heads)}
    outputs: list[torch.Tensor] = []
    for record_index in range(tensors["queries"].shape[0]):
        visible_len = int(tensors["visible_lengths"][record_index].item())
        per_head: list[torch.Tensor] = []
        for query_local_index, kv_head in enumerate(query_to_kv_heads):
            kv_local_index = kv_lookup[kv_head]
            per_head.append(
                replay_head_attention_output(
                    tensors["queries"][record_index, query_local_index],
                    tensors["final_keys"][kv_local_index, :visible_len, :],
                    tensors["final_values"][kv_local_index, :visible_len, :],
                    scaling=scaling,
                )
            )
        outputs.append(torch.stack(per_head, dim=0))
    return torch.stack(outputs, dim=0)


def _max_abs_rel_diff(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    left32 = left.to(torch.float32)
    right32 = right.to(torch.float32)
    diff = torch.abs(left32 - right32)
    denom = torch.maximum(torch.maximum(torch.abs(left32), torch.abs(right32)), torch.tensor(1e-12, dtype=torch.float32))
    rel = diff / denom
    return float(torch.max(diff).item()), float(torch.max(rel).item())


def _dependency_versions() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "safetensors_version": safetensors.__version__,
        "gmpy2_version": gmpy2.version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": "cpu",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _materialize_layer_trace_capture(
    *,
    trace_path: Path,
    repo_id: str,
    repo_revision: str,
    prompt: str,
    input_ids: torch.Tensor,
    layer_index: int,
    checkpoint_summary: CheckpointConfigSummary,
    selected_query_heads: tuple[int, ...],
    selected_kv_heads: tuple[int, ...],
    query_to_kv_heads: tuple[int, ...],
    scaling: float,
    source_dtype: str,
    asset_hashes: dict[str, str],
    asset_sources: dict[str, str],
    model_index_sha256: str,
    model_index_source: str,
    fetched_tensor_hashes: dict[str, str],
    dependency_versions: dict[str, Any],
    tls_verification: bool,
    manual_run: _LayerTraceRun,
    stock_run: _LayerStockRun,
) -> tuple[PerLayerTraceCaptureResult, dict[str, Any]]:
    tensors = _build_compact_trace_tensors(
        manual_run,
        selected_query_heads=selected_query_heads,
        selected_kv_heads=selected_kv_heads,
        layer_index=layer_index,
    )
    metadata = {
        "trace_schema": TRACE_SCHEMA_COMPACT_V1,
        "checkpoint_repo": repo_id,
        "checkpoint_revision": repo_revision,
        "prompt": prompt,
        "prompt_token_ids": input_ids.tolist(),
        "layer_index": layer_index,
        "selected_query_heads": list(selected_query_heads),
        "selected_kv_heads": list(selected_kv_heads),
        "query_to_kv_heads": list(query_to_kv_heads),
        "num_attention_heads": checkpoint_summary.num_attention_heads,
        "num_key_value_heads": checkpoint_summary.num_key_value_heads,
        "head_dim": checkpoint_summary.head_dim,
        "scaling": float(scaling),
        "source_dtype": source_dtype,
        "storage_dtype": str(tensors["queries"].dtype),
        "gqa_group_size": checkpoint_summary.num_attention_heads // checkpoint_summary.num_key_value_heads,
        "rope_theta": checkpoint_summary.rope_theta,
        "rope_scaling": checkpoint_summary.rope_scaling,
        "max_position_embeddings": checkpoint_summary.max_position_embeddings,
        "tls_verification": tls_verification,
        "asset_hashes": asset_hashes,
        "model_index_sha256": model_index_sha256,
        "fetched_tensor_hashes": fetched_tensor_hashes,
        "dependency_versions": dependency_versions,
    }
    save_attention_trace(trace_path, tensors, metadata)
    reloaded_tensors, reloaded_metadata = load_attention_trace(trace_path)
    _validate_reloaded_trace_heads(tensors=reloaded_tensors, metadata=reloaded_metadata)

    max_roundtrip_abs_diff = 0.0
    for key, tensor in tensors.items():
        reloaded = reloaded_tensors[key]
        if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            diff = torch.max(torch.abs(tensor.to(torch.float32) - reloaded.to(torch.float32))).item()
        else:
            diff = 0.0 if torch.equal(tensor, reloaded) else math.inf
        max_roundtrip_abs_diff = max(max_roundtrip_abs_diff, float(diff))

    compact_outputs = replay_compact_trace_outputs(
        reloaded_tensors,
        query_to_kv_heads=query_to_kv_heads,
        selected_kv_heads=selected_kv_heads,
        scaling=float(reloaded_metadata["scaling"]),
    )
    max_compact_replay_abs_diff, _ = _max_abs_rel_diff(compact_outputs, reloaded_tensors["model_head_outputs"])

    manual_projected = torch.stack([record.projected_output for record in manual_run.records], dim=0)
    manual_decoder = torch.stack([record.decoder_output for record in manual_run.records], dim=0)
    stock_projected = torch.stack(list(stock_run.projected_outputs), dim=0)
    stock_decoder = torch.stack(list(stock_run.decoder_outputs), dim=0)
    stock_projected_output_max_abs_diff, stock_projected_output_max_rel_diff = _max_abs_rel_diff(manual_projected, stock_projected)
    stock_decoder_output_max_abs_diff, stock_decoder_output_max_rel_diff = _max_abs_rel_diff(manual_decoder, stock_decoder)
    stock_cache_key_max_abs_diff, stock_cache_key_max_rel_diff = _max_abs_rel_diff(manual_run.final_keys, stock_run.final_keys)
    stock_cache_value_max_abs_diff, stock_cache_value_max_rel_diff = _max_abs_rel_diff(manual_run.final_values, stock_run.final_values)

    result = PerLayerTraceCaptureResult(
        layer_index=layer_index,
        trace_path=trace_path,
        trace_sha256=_sha256_file(trace_path),
        queries_shape=tuple(int(dim) for dim in reloaded_tensors["queries"].shape),
        final_keys_shape=tuple(int(dim) for dim in reloaded_tensors["final_keys"].shape),
        final_values_shape=tuple(int(dim) for dim in reloaded_tensors["final_values"].shape),
        model_head_outputs_shape=tuple(int(dim) for dim in reloaded_tensors["model_head_outputs"].shape),
        source_dtype=source_dtype,
        storage_dtype=str(reloaded_tensors["queries"].dtype),
        max_roundtrip_abs_diff=max_roundtrip_abs_diff,
        max_compact_replay_abs_diff=max_compact_replay_abs_diff,
        compact_replay_tolerance=1e-2,
        stock_projected_output_max_abs_diff=stock_projected_output_max_abs_diff,
        stock_projected_output_max_rel_diff=stock_projected_output_max_rel_diff,
        stock_decoder_output_max_abs_diff=stock_decoder_output_max_abs_diff,
        stock_decoder_output_max_rel_diff=stock_decoder_output_max_rel_diff,
        stock_cache_key_max_abs_diff=stock_cache_key_max_abs_diff,
        stock_cache_key_max_rel_diff=stock_cache_key_max_rel_diff,
        stock_cache_value_max_abs_diff=stock_cache_value_max_abs_diff,
        stock_cache_value_max_rel_diff=stock_cache_value_max_rel_diff,
    )
    report_entry = {
        "layer_index": layer_index,
        "trace_path": str(trace_path),
        "trace_sha256": result.trace_sha256,
        "trace_bytes": trace_path.stat().st_size,
        "queries_shape": list(result.queries_shape),
        "final_keys_shape": list(result.final_keys_shape),
        "final_values_shape": list(result.final_values_shape),
        "model_head_outputs_shape": list(result.model_head_outputs_shape),
        "selected_query_heads": list(selected_query_heads),
        "selected_kv_heads": list(selected_kv_heads),
        "query_to_kv_heads": list(query_to_kv_heads),
        "source_dtype": source_dtype,
        "storage_dtype": result.storage_dtype,
        "roundtrip": {"max_abs_diff": result.max_roundtrip_abs_diff},
        "compact_replay": {"max_abs_diff": result.max_compact_replay_abs_diff},
        "stock_forward_comparison": {
            "projected_output_max_abs_diff": result.stock_projected_output_max_abs_diff,
            "projected_output_max_rel_diff": result.stock_projected_output_max_rel_diff,
            "decoder_output_max_abs_diff": result.stock_decoder_output_max_abs_diff,
            "decoder_output_max_rel_diff": result.stock_decoder_output_max_rel_diff,
        },
        "cache_comparison": {
            "key_max_abs_diff": result.stock_cache_key_max_abs_diff,
            "key_max_rel_diff": result.stock_cache_key_max_rel_diff,
            "value_max_abs_diff": result.stock_cache_value_max_abs_diff,
            "value_max_rel_diff": result.stock_cache_value_max_rel_diff,
        },
    }
    return result, report_entry


def ensure_minimal_llama31_assets(
    *,
    output_dir: str | Path,
    repo_id: str = DEFAULT_LLAMA31_BASE_REPO,
    repo_revision: str | None = None,
    allow_insecure_tls: bool = False,
) -> LocalAssetPreparationResult:
    download_policy = _DownloadPolicy(verify_tls=not allow_insecure_tls)
    if repo_revision is None:
        repo_revision = _fetch_repo_revision(repo_id, download_policy=download_policy)
        repo_revision_source = "resolved_api"
    else:
        repo_revision = str(repo_revision)
        repo_revision_source = "provided"
    asset_dir = Path(output_dir) / "llama31_base_assets" / repo_revision
    asset_hashes, asset_sources = _prepare_local_assets(
        repo_id,
        repo_revision,
        asset_dir,
        download_policy=download_policy,
    )
    return LocalAssetPreparationResult(
        repo_id=repo_id,
        repo_revision=repo_revision,
        asset_dir=asset_dir,
        asset_hashes=asset_hashes,
        asset_sources=asset_sources,
        tls_verification=download_policy.verify_tls,
        repo_revision_source=repo_revision_source,
    )


def run_minimal_llama31_capture(
    *,
    output_dir: str | Path,
    repo_id: str = DEFAULT_LLAMA31_BASE_REPO,
    repo_revision: str | None = None,
    prompt: str = DEFAULT_PROMPT,
    prompt_token_ids: Sequence[int] | None = None,
    layer_index: int = 0,
    selected_query_heads: tuple[int, ...] | None = None,
    allow_insecure_tls: bool = False,
) -> MinimalTraceCaptureResult:
    if layer_index != 0:
        raise ValueError(
            "The current capture implementation is validated only for layer 0. "
            "It feeds token embeddings directly into the selected decoder layer and therefore rejects layer_index != 0."
        )

    download_policy = _DownloadPolicy(verify_tls=not allow_insecure_tls)
    output_path = Path(output_dir)
    process = psutil.Process()
    rss_samples = [process.memory_info().rss]

    if repo_revision is None:
        repo_revision = _fetch_repo_revision(repo_id, download_policy=download_policy)
        repo_revision_source = "resolved_api"
    else:
        repo_revision = str(repo_revision)
        repo_revision_source = "provided"
    asset_dir = output_path / "llama31_base_assets" / repo_revision
    trace_path = output_path / DEFAULT_TRACE_FILENAME
    capture_report_path = output_path / DEFAULT_REPORT_FILENAME
    dependency_report_path = output_path / DEFAULT_DEPENDENCY_REPORT_FILENAME

    asset_hashes, asset_sources = _prepare_local_assets(
        repo_id,
        repo_revision,
        asset_dir,
        download_policy=download_policy,
    )
    config, checkpoint_summary = _load_checkpoint_summary(asset_dir, repo_id=repo_id, repo_revision=repo_revision)
    if prompt_token_ids is None:
        input_ids = _tokenize_prompt(asset_dir, prompt)
        prompt_token_source = "tokenized_prompt"
    else:
        token_list = [int(token_id) for token_id in prompt_token_ids]
        if not token_list:
            raise ValueError("prompt_token_ids must be non-empty when provided.")
        input_ids = torch.tensor(token_list, dtype=torch.long)
        prompt_token_source = "provided_token_ids"
    rss_samples.append(process.memory_info().rss)

    weight_index, model_index_sha256, model_index_source = _fetch_weight_index(
        repo_id,
        repo_revision,
        asset_dir,
        download_policy=download_policy,
    )
    fetched_tensor_hashes: dict[str, str] = {}
    remote_shard_cache: dict[str, _RemoteSafetensorsShard] = {}
    layer_state_dict = _fetch_layer_state_dict(
        repo_id=repo_id,
        revision=repo_revision,
        weight_index=weight_index,
        download_policy=download_policy,
        fetched_tensor_hashes=fetched_tensor_hashes,
        layer_index=layer_index,
        shard_cache=remote_shard_cache,
    )
    input_embeddings = _fetch_prompt_embeddings_from_weight_index(
        repo_id=repo_id,
        revision=repo_revision,
        weight_index=weight_index,
        token_ids=input_ids,
        download_policy=download_policy,
        fetched_tensor_hashes=fetched_tensor_hashes,
        shard_cache=remote_shard_cache,
    ).to(dtype=torch.bfloat16)
    rss_samples.append(process.memory_info().rss)

    layer = _load_decoder_layer_from_state_dict(config, layer_index=layer_index, state_dict=layer_state_dict)
    num_attention_heads = checkpoint_summary.num_attention_heads
    num_key_value_heads = checkpoint_summary.num_key_value_heads
    if selected_query_heads is None:
        selected_query_heads = _default_capture_query_heads(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
    query_to_kv_heads = tuple(
        query_head_to_kv_head(
            query_head_index=query_head,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
        for query_head in selected_query_heads
    )
    selected_kv_heads = tuple(sorted(set(query_to_kv_heads)))
    _validate_selected_head_configuration(
        selected_query_heads=selected_query_heads,
        query_to_kv_heads=query_to_kv_heads,
        selected_kv_heads=selected_kv_heads,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )

    manual_run = _run_manual_decoder_layer_trace(
        config=config,
        layer=layer,
        input_hidden_states=input_embeddings,
        input_ids=input_ids,
        selected_query_heads=selected_query_heads,
    )
    stock_run = _run_stock_decoder_layer_reference(
        config=config,
        layer=layer,
        input_hidden_states=input_embeddings,
        input_ids=input_ids,
    )
    dependency_versions = _dependency_versions()
    layer_result, layer_report_entry = _materialize_layer_trace_capture(
        trace_path=trace_path,
        repo_id=repo_id,
        repo_revision=repo_revision,
        prompt=prompt,
        input_ids=input_ids,
        layer_index=layer_index,
        checkpoint_summary=checkpoint_summary,
        selected_query_heads=selected_query_heads,
        selected_kv_heads=selected_kv_heads,
        query_to_kv_heads=query_to_kv_heads,
        scaling=float(layer.self_attn.scaling),
        source_dtype=str(input_embeddings.dtype),
        asset_hashes=asset_hashes,
        asset_sources=asset_sources,
        model_index_sha256=model_index_sha256,
        model_index_source=model_index_source,
        fetched_tensor_hashes=fetched_tensor_hashes,
        dependency_versions=dependency_versions,
        tls_verification=download_policy.verify_tls,
        manual_run=manual_run,
        stock_run=stock_run,
    )
    rss_samples.append(process.memory_info().rss)

    dependency_report = {
        **dependency_versions,
        "checkpoint_repo": repo_id,
        "checkpoint_revision": repo_revision,
        "tls_verification": download_policy.verify_tls,
    }
    _write_json(dependency_report_path, dependency_report)

    capture_report = {
        "trace_schema": TRACE_SCHEMA_COMPACT_V1,
        "trace_path": str(trace_path),
        "trace_sha256": _sha256_file(trace_path),
        "trace_bytes": trace_path.stat().st_size,
        "capture_report_path": str(capture_report_path),
        "dependency_report_path": str(dependency_report_path),
        "checkpoint": {
            "repo_id": repo_id,
            "revision": repo_revision,
            "num_hidden_layers": checkpoint_summary.num_hidden_layers,
            "num_attention_heads": checkpoint_summary.num_attention_heads,
            "num_key_value_heads": checkpoint_summary.num_key_value_heads,
            "head_dim": checkpoint_summary.head_dim,
            "hidden_size": checkpoint_summary.hidden_size,
            "intermediate_size": checkpoint_summary.intermediate_size,
            "model_dtype": checkpoint_summary.model_dtype,
            "rope_theta": checkpoint_summary.rope_theta,
            "rope_scaling": checkpoint_summary.rope_scaling,
            "max_position_embeddings": checkpoint_summary.max_position_embeddings,
        },
        "tls_verification": download_policy.verify_tls,
        "repo_revision_source": repo_revision_source,
        "prompt": prompt,
        "prompt_token_ids": input_ids.tolist(),
        "prompt_token_source": prompt_token_source,
        "layer_index": layer_index,
        "selected_query_heads": list(selected_query_heads),
        "selected_kv_heads": list(selected_kv_heads),
        "query_to_kv_heads": list(query_to_kv_heads),
        "queries_shape": list(layer_result.queries_shape),
        "final_keys_shape": list(layer_result.final_keys_shape),
        "final_values_shape": list(layer_result.final_values_shape),
        "model_head_outputs_shape": list(layer_result.model_head_outputs_shape),
        "source_dtype": str(input_embeddings.dtype),
        "storage_dtype": layer_result.storage_dtype,
        "validation_tolerances": {
            "roundtrip_abs_tolerance": 0.0,
            "compact_replay_abs_tolerance": 1e-2,
            "stock_forward_abs_tolerance": 1e-2,
            "stock_forward_rel_tolerance": 1e-2,
            "reason": "All comparisons use the same eager layer-0 checkpoint weights and bfloat16 tensors; tolerances only cover implementation noise.",
        },
        "roundtrip": layer_report_entry["roundtrip"],
        "compact_replay": layer_report_entry["compact_replay"],
        "stock_forward_comparison": layer_report_entry["stock_forward_comparison"],
        "cache_comparison": layer_report_entry["cache_comparison"],
        "asset_hashes": asset_hashes,
        "asset_sources": asset_sources,
        "model_index_sha256": model_index_sha256,
        "model_index_source": model_index_source,
        "fetched_tensor_hashes": fetched_tensor_hashes,
        "peak_rss_bytes": max(rss_samples),
        "dependency_versions": dependency_versions,
    }
    _write_json(capture_report_path, capture_report)

    return MinimalTraceCaptureResult(
        checkpoint=checkpoint_summary,
        trace_path=trace_path,
        capture_report_path=capture_report_path,
        dependency_report_path=dependency_report_path,
        prompt=prompt,
        token_ids=tuple(int(token_id) for token_id in input_ids.tolist()),
        layer_index=layer_index,
        selected_query_heads=selected_query_heads,
        selected_kv_heads=selected_kv_heads,
        query_to_kv_heads=query_to_kv_heads,
        queries_shape=layer_result.queries_shape,
        final_keys_shape=layer_result.final_keys_shape,
        final_values_shape=layer_result.final_values_shape,
        model_head_outputs_shape=layer_result.model_head_outputs_shape,
        source_dtype=str(input_embeddings.dtype),
        storage_dtype=layer_result.storage_dtype,
        tls_verification=download_policy.verify_tls,
        trace_schema=TRACE_SCHEMA_COMPACT_V1,
        max_roundtrip_abs_diff=layer_result.max_roundtrip_abs_diff,
        max_compact_replay_abs_diff=layer_result.max_compact_replay_abs_diff,
        compact_replay_tolerance=layer_result.compact_replay_tolerance,
        stock_projected_output_max_abs_diff=layer_result.stock_projected_output_max_abs_diff,
        stock_projected_output_max_rel_diff=layer_result.stock_projected_output_max_rel_diff,
        stock_decoder_output_max_abs_diff=layer_result.stock_decoder_output_max_abs_diff,
        stock_decoder_output_max_rel_diff=layer_result.stock_decoder_output_max_rel_diff,
        stock_cache_key_max_abs_diff=layer_result.stock_cache_key_max_abs_diff,
        stock_cache_key_max_rel_diff=layer_result.stock_cache_key_max_rel_diff,
        stock_cache_value_max_abs_diff=layer_result.stock_cache_value_max_abs_diff,
        stock_cache_value_max_rel_diff=layer_result.stock_cache_value_max_rel_diff,
        peak_rss_bytes=max(rss_samples),
    )


def run_multilayer_llama31_capture(
    *,
    output_dir: str | Path,
    repo_id: str = DEFAULT_LLAMA31_BASE_REPO,
    repo_revision: str | None = None,
    prompt: str = DEFAULT_PROMPT,
    prompt_token_ids: Sequence[int] | None = None,
    capture_layer_indices: Sequence[int] = (0, 8, 16, 24, 31),
    selected_query_heads: tuple[int, ...] | None = None,
    allow_insecure_tls: bool = False,
) -> MultiLayerTraceCaptureResult:
    if not capture_layer_indices:
        raise ValueError("capture_layer_indices must be non-empty.")

    unique_layers = tuple(sorted({int(layer_index) for layer_index in capture_layer_indices}))
    if unique_layers[0] < 0:
        raise ValueError("capture_layer_indices must be non-negative.")

    download_policy = _DownloadPolicy(verify_tls=not allow_insecure_tls)
    output_path = Path(output_dir)
    process = psutil.Process()
    rss_samples = [process.memory_info().rss]

    if repo_revision is None:
        repo_revision = _fetch_repo_revision(repo_id, download_policy=download_policy)
        repo_revision_source = "resolved_api"
    else:
        repo_revision = str(repo_revision)
        repo_revision_source = "provided"
    asset_dir = output_path / "llama31_base_assets" / repo_revision
    capture_report_path = output_path / DEFAULT_REPORT_FILENAME
    dependency_report_path = output_path / DEFAULT_DEPENDENCY_REPORT_FILENAME

    asset_hashes, asset_sources = _prepare_local_assets(
        repo_id,
        repo_revision,
        asset_dir,
        download_policy=download_policy,
    )
    config, checkpoint_summary = _load_checkpoint_summary(asset_dir, repo_id=repo_id, repo_revision=repo_revision)
    if unique_layers[-1] >= checkpoint_summary.num_hidden_layers:
        raise ValueError(
            f"capture_layer_indices must stay below num_hidden_layers={checkpoint_summary.num_hidden_layers}."
        )
    if prompt_token_ids is None:
        input_ids = _tokenize_prompt(asset_dir, prompt)
        prompt_token_source = "tokenized_prompt"
    else:
        token_list = [int(token_id) for token_id in prompt_token_ids]
        if not token_list:
            raise ValueError("prompt_token_ids must be non-empty when provided.")
        input_ids = torch.tensor(token_list, dtype=torch.long)
        prompt_token_source = "provided_token_ids"
    rss_samples.append(process.memory_info().rss)

    weight_index, model_index_sha256, model_index_source = _fetch_weight_index(
        repo_id,
        repo_revision,
        asset_dir,
        download_policy=download_policy,
    )
    fetched_tensor_hashes: dict[str, str] = {}
    remote_shard_cache: dict[str, _RemoteSafetensorsShard] = {}
    input_embeddings = _fetch_prompt_embeddings_from_weight_index(
        repo_id=repo_id,
        revision=repo_revision,
        weight_index=weight_index,
        token_ids=input_ids,
        download_policy=download_policy,
        fetched_tensor_hashes=fetched_tensor_hashes,
        shard_cache=remote_shard_cache,
    ).to(dtype=torch.bfloat16)
    rss_samples.append(process.memory_info().rss)

    num_attention_heads = checkpoint_summary.num_attention_heads
    num_key_value_heads = checkpoint_summary.num_key_value_heads
    if selected_query_heads is None:
        selected_query_heads = _default_capture_query_heads(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
    query_to_kv_heads = tuple(
        query_head_to_kv_head(
            query_head_index=query_head,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
        for query_head in selected_query_heads
    )
    selected_kv_heads = tuple(sorted(set(query_to_kv_heads)))
    _validate_selected_head_configuration(
        selected_query_heads=selected_query_heads,
        query_to_kv_heads=query_to_kv_heads,
        selected_kv_heads=selected_kv_heads,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )

    dependency_versions = _dependency_versions()
    current_hidden = input_embeddings.detach().cpu().clone()
    layer_results: list[PerLayerTraceCaptureResult] = []
    layer_report_entries: list[dict[str, Any]] = []
    streamed_layer_validation: list[dict[str, Any]] = []
    layer_tolerance_abs = 1e-2
    layer_tolerance_rel = 1e-2
    captured_layer_set = set(unique_layers)

    for layer_index in range(unique_layers[-1] + 1):
        layer_state_dict = _fetch_layer_state_dict(
            repo_id=repo_id,
            revision=repo_revision,
            weight_index=weight_index,
            download_policy=download_policy,
            fetched_tensor_hashes=fetched_tensor_hashes,
            layer_index=layer_index,
            shard_cache=remote_shard_cache,
        )
        layer = _load_decoder_layer_from_state_dict(config, layer_index=layer_index, state_dict=layer_state_dict)
        manual_run: _LayerTraceRun | None = None
        stock_run: _LayerStockRun | None = None
        if layer_index in captured_layer_set:
            manual_run = _run_manual_decoder_layer_trace(
                config=config,
                layer=layer,
                input_hidden_states=current_hidden,
                input_ids=input_ids,
                selected_query_heads=selected_query_heads,
            )
            stock_run = _run_stock_decoder_layer_reference(
                config=config,
                layer=layer,
                input_hidden_states=current_hidden,
                input_ids=input_ids,
            )
            next_hidden = stock_run.next_hidden_states.to(dtype=torch.bfloat16).contiguous()
            manual_next_hidden = manual_run.next_hidden_states.to(dtype=torch.bfloat16).contiguous()
            layer_decoder_output_max_abs_diff, layer_decoder_output_max_rel_diff = _max_abs_rel_diff(
                manual_next_hidden,
                next_hidden,
            )
            if (
                layer_decoder_output_max_abs_diff > layer_tolerance_abs
                or layer_decoder_output_max_rel_diff > layer_tolerance_rel
            ):
                raise ValueError(
                    f"Manual-versus-stock decoder mismatch at layer {layer_index}: "
                    f"abs={layer_decoder_output_max_abs_diff} rel={layer_decoder_output_max_rel_diff}."
                )
            streamed_layer_validation.append(
                {
                    "layer_index": layer_index,
                    "validation_mode": "manual_vs_stock_sequential",
                    "decoder_output_max_abs_diff": layer_decoder_output_max_abs_diff,
                    "decoder_output_max_rel_diff": layer_decoder_output_max_rel_diff,
                }
            )
            trace_path = output_path / f"llama31_layer{layer_index}_trace.safetensors"
            layer_result, layer_report_entry = _materialize_layer_trace_capture(
                trace_path=trace_path,
                repo_id=repo_id,
                repo_revision=repo_revision,
                prompt=prompt,
                input_ids=input_ids,
                layer_index=layer_index,
                checkpoint_summary=checkpoint_summary,
                selected_query_heads=selected_query_heads,
                selected_kv_heads=selected_kv_heads,
                query_to_kv_heads=query_to_kv_heads,
                scaling=float(layer.self_attn.scaling),
                source_dtype=str(input_embeddings.dtype),
                asset_hashes=asset_hashes,
                asset_sources=asset_sources,
                model_index_sha256=model_index_sha256,
                model_index_source=model_index_source,
                fetched_tensor_hashes=fetched_tensor_hashes,
                dependency_versions=dependency_versions,
                tls_verification=download_policy.verify_tls,
                manual_run=manual_run,
                stock_run=stock_run,
            )
            layer_results.append(layer_result)
            layer_report_entry["streamed_decoder_output_max_abs_diff"] = layer_decoder_output_max_abs_diff
            layer_report_entry["streamed_decoder_output_max_rel_diff"] = layer_decoder_output_max_rel_diff
            layer_report_entries.append(layer_report_entry)
            del manual_next_hidden
        else:
            next_hidden = _run_stock_decoder_layer_full_sequence(
                config=config,
                layer=layer,
                input_hidden_states=current_hidden,
            ).to(dtype=torch.bfloat16)
            streamed_layer_validation.append(
                {
                    "layer_index": layer_index,
                    "validation_mode": "stock_full_sequence_only",
                }
            )

        current_hidden = next_hidden
        rss_samples.append(process.memory_info().rss)

        del layer_state_dict
        del layer
        if manual_run is not None:
            del manual_run
        if stock_run is not None:
            del stock_run
        del next_hidden
        gc.collect()
        rss_samples.append(process.memory_info().rss)

    dependency_report = {
        **dependency_versions,
        "checkpoint_repo": repo_id,
        "checkpoint_revision": repo_revision,
        "tls_verification": download_policy.verify_tls,
    }
    _write_json(dependency_report_path, dependency_report)

    capture_report = {
        "trace_schema": TRACE_SCHEMA_COMPACT_V1,
        "capture_report_path": str(capture_report_path),
        "dependency_report_path": str(dependency_report_path),
        "checkpoint": {
            "repo_id": repo_id,
            "revision": repo_revision,
            "num_hidden_layers": checkpoint_summary.num_hidden_layers,
            "num_attention_heads": checkpoint_summary.num_attention_heads,
            "num_key_value_heads": checkpoint_summary.num_key_value_heads,
            "head_dim": checkpoint_summary.head_dim,
            "hidden_size": checkpoint_summary.hidden_size,
            "intermediate_size": checkpoint_summary.intermediate_size,
            "model_dtype": checkpoint_summary.model_dtype,
            "rope_theta": checkpoint_summary.rope_theta,
            "rope_scaling": checkpoint_summary.rope_scaling,
            "max_position_embeddings": checkpoint_summary.max_position_embeddings,
        },
        "tls_verification": download_policy.verify_tls,
        "repo_revision_source": repo_revision_source,
        "prompt": prompt,
        "prompt_token_ids": input_ids.tolist(),
        "prompt_token_source": prompt_token_source,
        "captured_layers": list(unique_layers),
        "selected_query_heads": list(selected_query_heads),
        "selected_kv_heads": list(selected_kv_heads),
        "query_to_kv_heads": list(query_to_kv_heads),
        "validation_tolerances": {
            "roundtrip_abs_tolerance": 0.0,
            "compact_replay_abs_tolerance": 1e-2,
            "stock_forward_abs_tolerance": 1e-2,
            "stock_forward_rel_tolerance": 1e-2,
            "streamed_decoder_abs_tolerance": layer_tolerance_abs,
            "streamed_decoder_rel_tolerance": layer_tolerance_rel,
            "reason": "All comparisons use the same eager checkpoint weights and bfloat16 tensors; tolerances only cover implementation noise.",
        },
        "asset_hashes": asset_hashes,
        "asset_sources": asset_sources,
        "model_index_sha256": model_index_sha256,
        "model_index_source": model_index_source,
        "fetched_tensor_hashes": fetched_tensor_hashes,
        "streamed_layer_validation": streamed_layer_validation,
        "layers": layer_report_entries,
        "peak_rss_bytes": max(rss_samples),
        "dependency_versions": dependency_versions,
    }
    _write_json(capture_report_path, capture_report)

    return MultiLayerTraceCaptureResult(
        checkpoint=checkpoint_summary,
        capture_report_path=capture_report_path,
        dependency_report_path=dependency_report_path,
        prompt=prompt,
        token_ids=tuple(int(token_id) for token_id in input_ids.tolist()),
        selected_layer_indices=unique_layers,
        selected_query_heads=selected_query_heads,
        selected_kv_heads=selected_kv_heads,
        query_to_kv_heads=query_to_kv_heads,
        tls_verification=download_policy.verify_tls,
        layer_results=tuple(layer_results),
        peak_rss_bytes=max(rss_samples),
    )
