"""Fail-closed physical KV payload access for RACK-KV V2.

Certification metadata is loaded independently from compressed leaf payloads.
This module does not change the V1 codec or theorem; it only controls when an
already-authorized physical block is fetched and decoded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import struct
import time
from typing import Mapping, Sequence

import numpy as np

from .anisotropic import reconstructed_attention_output
from .codec import CompressedBlock


_MAGIC = b"RACKV2P1"
_PREFIX = struct.Struct("<8sQ")


@dataclass(frozen=True)
class PhysicalBlockMetadata:
    block_id: int
    block_start: int
    block_len: int
    offset: int
    byte_length: int
    certification: Mapping[str, object] = field(default_factory=dict)


@dataclass
class PayloadIOStats:
    payload_read_calls: int = 0
    payload_bytes_read: int = 0
    blocks_loaded: int = 0
    blocks_decoded: int = 0
    metadata_bytes_read: int = 0
    metadata_lookup_calls: int = 0
    metadata_lookup_seconds: float = 0.0
    payload_io_seconds: float = 0.0
    decode_seconds: float = 0.0


class KVPayloadStore:
    """Indexed file-backed store with lazy block payload reads."""

    def __init__(self, path: Path, metadata: tuple[PhysicalBlockMetadata, ...], metadata_bytes: int):
        self.path = Path(path)
        self._metadata = metadata
        self.stats = PayloadIOStats(metadata_bytes_read=metadata_bytes)
        self.forbidden_block_ids: set[int] = set()

    @classmethod
    def create(cls, path: Path, blocks: Sequence[CompressedBlock], certification_metadata: Mapping[int, Mapping[str, object]] | None = None) -> "KVPayloadStore":
        if not blocks:
            raise ValueError("cannot create an empty physical payload store")
        payloads = [block.serialize() for block in blocks]
        metadata_start = _PREFIX.size
        # Offsets are absolute file offsets, after the JSON metadata region.
        metadata_entries = [
            {"block_id": i, "block_start": int(block.header.block_start if hasattr(block, "header") else block.block_start), "block_len": int(block.block_len),
             "byte_length": len(payload), "certification": dict((certification_metadata or {}).get(i, {}))}
            for i, (block, payload) in enumerate(zip(blocks, payloads))
        ]
        metadata_blob = json.dumps(metadata_entries, separators=(",", ":")).encode("utf-8")
        payload_start = metadata_start + len(metadata_blob)
        cursor = payload_start
        finalized = []
        for entry, payload in zip(metadata_entries, payloads):
            entry = dict(entry)
            entry["offset"] = cursor
            finalized.append(entry)
            cursor += len(payload)
        metadata_blob = json.dumps(finalized, separators=(",", ":")).encode("utf-8")
        # Recompute once because JSON offsets can change the metadata length.
        while payload_start != _PREFIX.size + len(metadata_blob):
            payload_start = _PREFIX.size + len(metadata_blob)
            cursor = payload_start
            finalized = []
            for entry, payload in zip(metadata_entries, payloads):
                entry = dict(entry)
                entry["offset"] = cursor
                finalized.append(entry)
                cursor += len(payload)
            metadata_blob = json.dumps(finalized, separators=(",", ":")).encode("utf-8")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_PREFIX.pack(_MAGIC, len(metadata_blob)) + metadata_blob + b"".join(payloads))
        return cls.open(target)

    @classmethod
    def open(cls, path: Path) -> "KVPayloadStore":
        target = Path(path)
        with target.open("rb") as handle:
            prefix = handle.read(_PREFIX.size)
            if len(prefix) != _PREFIX.size:
                raise ValueError("physical payload store is truncated")
            magic, metadata_length = _PREFIX.unpack(prefix)
            if magic != _MAGIC:
                raise ValueError("physical payload store magic mismatch")
            metadata_blob = handle.read(metadata_length)
        if len(metadata_blob) != metadata_length:
            raise ValueError("physical payload store metadata is truncated")
        entries = json.loads(metadata_blob.decode("utf-8"))
        metadata = tuple(PhysicalBlockMetadata(**entry) for entry in entries)
        return cls(target, metadata, _PREFIX.size + metadata_length)

    @property
    def block_ids(self) -> tuple[int, ...]:
        return tuple(item.block_id for item in self._metadata)

    @property
    def metadata_bytes(self) -> int:
        return self.stats.metadata_bytes_read

    @property
    def full_payload_bytes(self) -> int:
        return sum(item.byte_length for item in self._metadata)

    @property
    def full_block_count(self) -> int:
        return len(self._metadata)

    def get_metadata(self, block_id: int) -> PhysicalBlockMetadata:
        started = time.perf_counter()
        self.stats.metadata_lookup_calls += 1
        try:
            return self._metadata[block_id]
        finally:
            self.stats.metadata_lookup_seconds += time.perf_counter() - started

    def read_block(self, block_id: int) -> bytes:
        if block_id in self.forbidden_block_ids:
            raise AssertionError(f"forbidden physically omitted block {block_id} was read")
        metadata = self.get_metadata(block_id)
        started = time.perf_counter()
        with self.path.open("rb") as handle:
            handle.seek(metadata.offset)
            payload = handle.read(metadata.byte_length)
        self.stats.payload_io_seconds += time.perf_counter() - started
        if len(payload) != metadata.byte_length:
            raise ValueError(f"payload for block {block_id} is truncated")
        self.stats.payload_read_calls += 1
        self.stats.payload_bytes_read += len(payload)
        self.stats.blocks_loaded += 1
        return payload

    def decode_block(self, block_id: int, *, key_dim: int, value_dim: int, decoder=None):
        started = time.perf_counter()
        payload = self.read_block(block_id)
        block = decoder(payload) if decoder is not None else CompressedBlock.deserialize(payload, key_dim=key_dim, value_dim=value_dim)
        self.stats.decode_seconds += time.perf_counter() - started
        self.stats.blocks_decoded += 1
        return block


@dataclass(frozen=True)
class PhysicalExecutionResult:
    outputs_by_head: Mapping[int, np.ndarray]
    required_block_ids: tuple[int, ...]
    physically_omitted_block_ids: tuple[int, ...]
    fallback_reasons: tuple[str, ...]
    stats: PayloadIOStats

    @property
    def payload_bytes_avoided(self) -> int:
        return self.stats.full_payload_bytes - self.stats.payload_bytes_read  # type: ignore[attr-defined]


def _load_blocks(store: KVPayloadStore, block_ids: Sequence[int], *, key_dim: int, value_dim: int) -> dict[int, CompressedBlock]:
    return {block_id: store.decode_block(block_id, key_dim=key_dim, value_dim=value_dim) for block_id in block_ids}


def _outputs(
    *,
    queries_by_head: Mapping[int, np.ndarray],
    recent_keys_by_head: Mapping[int, np.ndarray],
    recent_values_by_head: Mapping[int, np.ndarray],
    loaded: Mapping[int, CompressedBlock],
    block_ids: Sequence[int],
    attention_scale: float | None,
) -> dict[int, np.ndarray]:
    decoded = {block_id: loaded[block_id].decode_block() for block_id in block_ids}
    outputs = {}
    for head, query in queries_by_head.items():
        keys = [np.asarray(recent_keys_by_head[head], dtype=np.float64)]
        values = [np.asarray(recent_values_by_head[head], dtype=np.float64)]
        for block_id in block_ids:
            keys.append(decoded[block_id][0])
            values.append(decoded[block_id][1])
        outputs[head] = reconstructed_attention_output(query, np.vstack(keys), np.vstack(values), attention_scale=attention_scale)
    return outputs


def execute_physical_gqa(
    *,
    store: KVPayloadStore,
    queries_by_head: Mapping[int, np.ndarray],
    recent_keys_by_head: Mapping[int, np.ndarray],
    recent_values_by_head: Mapping[int, np.ndarray],
    required_leaf_blocks: Sequence[int],
    physically_eligible_blocks: Sequence[int],
    complete_gqa_group: bool,
    all_heads_represented: bool,
    mpfr_authorized: bool,
    numerical_fallback: bool = False,
    key_dim: int,
    value_dim: int,
    attention_scale: float | None = None,
) -> PhysicalExecutionResult:
    """Load only the fail-closed required physical blocks and decode once each."""
    all_blocks = set(store.block_ids)
    required = set(required_leaf_blocks)
    eligible = set(physically_eligible_blocks)
    if not required.issubset(all_blocks) or not eligible.issubset(all_blocks):
        raise ValueError("physical block decision references an unknown block")
    reasons = []
    can_omit = True
    for condition, reason in (
        (complete_gqa_group, "incomplete_gqa_group"),
        (all_heads_represented, "missing_mapped_query_head"),
        (mpfr_authorized, "mpfr_not_authorized"),
        (not numerical_fallback, "numerical_fallback"),
    ):
        if not condition:
            can_omit = False
            reasons.append(reason)
    if can_omit:
        omitted = all_blocks - required
        if omitted != eligible:
            can_omit = False
            reasons.append("eligibility_required_set_mismatch")
    if not can_omit:
        required = all_blocks
        omitted = set()
    else:
        omitted = eligible
    loaded = _load_blocks(store, sorted(required), key_dim=key_dim, value_dim=value_dim)
    outputs = _outputs(
        queries_by_head=queries_by_head,
        recent_keys_by_head=recent_keys_by_head,
        recent_values_by_head=recent_values_by_head,
        loaded=loaded,
        block_ids=sorted(required),
        attention_scale=attention_scale,
    )
    # Attach full-workload counters without counting them as actual reads.
    store.stats.full_payload_bytes = store.full_payload_bytes  # type: ignore[attr-defined]
    store.stats.full_block_count = store.full_block_count  # type: ignore[attr-defined]
    return PhysicalExecutionResult(outputs, tuple(sorted(required)), tuple(sorted(omitted)), tuple(reasons), store.stats)


def execute_logical_gqa(
    *,
    store: KVPayloadStore,
    required_leaf_blocks: Sequence[int],
    queries_by_head: Mapping[int, np.ndarray],
    recent_keys_by_head: Mapping[int, np.ndarray],
    recent_values_by_head: Mapping[int, np.ndarray],
    key_dim: int,
    value_dim: int,
    attention_scale: float | None = None,
) -> dict[int, np.ndarray]:
    """Reference logical path: read/decode all blocks, then retain required ones."""
    loaded = _load_blocks(store, store.block_ids, key_dim=key_dim, value_dim=value_dim)
    return _outputs(
        queries_by_head=queries_by_head,
        recent_keys_by_head=recent_keys_by_head,
        recent_values_by_head=recent_values_by_head,
        loaded=loaded,
        block_ids=sorted(required_leaf_blocks),
        attention_scale=attention_scale,
    )
