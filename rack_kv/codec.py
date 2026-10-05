from __future__ import annotations

from dataclasses import dataclass
import struct
import sys
from typing import Sequence

import gmpy2
import numpy as np

from .accounting import compressed_block_bytes
from .ieee import mpfr_to_proven_float32_upper
from .rigorous import (
    DEFAULT_PRECISION,
    exact_mpfr,
    exact_vector,
    max_exact,
    norm_upper,
    rounded_add,
    rounded_mul,
    rounded_sqrt,
    rounded_sub,
)

_BLOCK_CONTAINER_MAGIC = b"RACKKVH1"
_BLOCK_CONTAINER_HEADER = struct.Struct("<8sIHH")
_BLOCK_CONTAINER_INDEX_ENTRY = struct.Struct("<II")


@dataclass(frozen=True)
class BlockHeader:
    block_start: int
    block_len: int
    anchor_key: np.ndarray
    anchor_value: np.ndarray
    key_scale: np.float16
    value_scale: np.float16
    rho_upper: np.float32
    nu_upper: np.float32

    def __post_init__(self) -> None:
        if self.block_start < 0:
            raise ValueError("block_start must be nonnegative.")
        if self.block_len <= 0:
            raise ValueError("block_len must be positive.")
        if self.anchor_key.ndim != 1 or self.anchor_value.ndim != 1:
            raise ValueError("Anchor tensors must be rank-1 arrays.")
        if self.anchor_key.size == 0 or self.anchor_value.size == 0:
            raise ValueError("Anchor tensors must be non-empty.")
        if not np.all(np.isfinite(self.anchor_key)):
            raise ValueError("Anchor keys must be finite.")
        if not np.all(np.isfinite(self.anchor_value)):
            raise ValueError("Anchor values must be finite.")
        if not np.isfinite(self.key_scale) or float(self.key_scale) == 0.0:
            raise ValueError("key_scale must be finite and nonzero.")
        if not np.isfinite(self.value_scale) or float(self.value_scale) == 0.0:
            raise ValueError("value_scale must be finite and nonzero.")
        if not np.isfinite(self.rho_upper) or float(self.rho_upper) < 0.0:
            raise ValueError("rho_upper must be finite and nonnegative.")
        if not np.isfinite(self.nu_upper) or float(self.nu_upper) < 0.0:
            raise ValueError("nu_upper must be finite and nonnegative.")

    def metadata_bytes(self) -> int:
        return compressed_block_bytes(self.block_len, int(self.anchor_key.shape[0]), int(self.anchor_value.shape[0])) - (
            (self.block_len - 1) * (int(self.anchor_key.shape[0]) + int(self.anchor_value.shape[0]))
        )


@dataclass(frozen=True)
class CompressedBlock:
    header: BlockHeader
    key_residuals: np.ndarray
    value_residuals: np.ndarray

    def __post_init__(self) -> None:
        expected_key_shape = (self.header.block_len - 1, int(self.header.anchor_key.shape[0]))
        expected_value_shape = (self.header.block_len - 1, int(self.header.anchor_value.shape[0]))
        if self.key_residuals.shape != expected_key_shape:
            raise ValueError(f"Key residual shape mismatch: expected {expected_key_shape}, got {self.key_residuals.shape}.")
        if self.value_residuals.shape != expected_value_shape:
            raise ValueError(
                f"Value residual shape mismatch: expected {expected_value_shape}, got {self.value_residuals.shape}."
            )
        if self.key_residuals.dtype != np.int8 or self.value_residuals.dtype != np.int8:
            raise ValueError("Residual payloads must be int8 arrays.")

    @property
    def block_len(self) -> int:
        return self.header.block_len

    def decode_key_token(self, local_index: int) -> np.ndarray:
        if local_index < 0 or local_index >= self.block_len:
            raise IndexError("Token index out of range for compressed block.")
        anchor_key = self.header.anchor_key.astype(np.float64)
        if local_index == 0:
            return anchor_key
        return anchor_key + self.key_residuals[local_index - 1].astype(np.float64) * float(self.header.key_scale)

    def decode_value_token(self, local_index: int) -> np.ndarray:
        if local_index < 0 or local_index >= self.block_len:
            raise IndexError("Token index out of range for compressed block.")
        anchor_value = self.header.anchor_value.astype(np.float64)
        if local_index == 0:
            return anchor_value
        return anchor_value + self.value_residuals[local_index - 1].astype(np.float64) * float(self.header.value_scale)

    def decode_token(self, local_index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.decode_key_token(local_index), self.decode_value_token(local_index)

    def decode_key_block(self) -> np.ndarray:
        return np.vstack([self.decode_key_token(local_index) for local_index in range(self.block_len)])

    def decode_value_block(self) -> np.ndarray:
        return np.vstack([self.decode_value_token(local_index) for local_index in range(self.block_len)])

    def decode_block(self) -> tuple[np.ndarray, np.ndarray]:
        return self.decode_key_block(), self.decode_value_block()

    def certificate_metadata_bytes(self) -> int:
        return self.header.metadata_bytes()

    def estimated_packed_bytes(self) -> int:
        return compressed_block_bytes(self.block_len, int(self.header.anchor_key.shape[0]), int(self.header.anchor_value.shape[0]))

    def serialize(self) -> bytes:
        flags_pad = 0
        chunks = [
            struct.pack("<IHH", self.header.block_start, self.header.block_len, flags_pad),
            np.asarray(self.header.anchor_key, dtype=np.float16).tobytes(order="C"),
            np.asarray(self.header.anchor_value, dtype=np.float16).tobytes(order="C"),
            np.asarray([self.header.key_scale], dtype=np.float16).tobytes(order="C"),
            np.asarray([self.header.value_scale], dtype=np.float16).tobytes(order="C"),
            np.asarray([self.header.rho_upper], dtype=np.float32).tobytes(order="C"),
            np.asarray([self.header.nu_upper], dtype=np.float32).tobytes(order="C"),
            np.asarray(self.key_residuals, dtype=np.int8).tobytes(order="C"),
            np.asarray(self.value_residuals, dtype=np.int8).tobytes(order="C"),
        ]
        blob = b"".join(chunks)
        expected = self.estimated_packed_bytes()
        if len(blob) != expected:
            raise RuntimeError(f"Serialized block size {len(blob)} does not match expected packed bytes {expected}.")
        return blob

    @classmethod
    def deserialize(cls, payload: bytes, *, key_dim: int, value_dim: int) -> CompressedBlock:
        header_len = 8
        if len(payload) < header_len:
            raise ValueError("Serialized payload is too short for a block header.")
        block_start, block_len, _flags = struct.unpack("<IHH", payload[:header_len])
        cursor = header_len

        anchor_key_bytes = 2 * key_dim
        anchor_value_bytes = 2 * value_dim
        residual_tokens = block_len - 1
        key_residual_bytes = residual_tokens * key_dim
        value_residual_bytes = residual_tokens * value_dim

        anchor_key = np.frombuffer(payload[cursor : cursor + anchor_key_bytes], dtype=np.float16).copy()
        cursor += anchor_key_bytes
        anchor_value = np.frombuffer(payload[cursor : cursor + anchor_value_bytes], dtype=np.float16).copy()
        cursor += anchor_value_bytes
        key_scale = np.frombuffer(payload[cursor : cursor + 2], dtype=np.float16).copy()[0]
        cursor += 2
        value_scale = np.frombuffer(payload[cursor : cursor + 2], dtype=np.float16).copy()[0]
        cursor += 2
        rho_upper = np.frombuffer(payload[cursor : cursor + 4], dtype=np.float32).copy()[0]
        cursor += 4
        nu_upper = np.frombuffer(payload[cursor : cursor + 4], dtype=np.float32).copy()[0]
        cursor += 4
        key_residuals = (
            np.frombuffer(payload[cursor : cursor + key_residual_bytes], dtype=np.int8).copy().reshape(residual_tokens, key_dim)
        )
        cursor += key_residual_bytes
        value_residuals = (
            np.frombuffer(payload[cursor : cursor + value_residual_bytes], dtype=np.int8).copy().reshape(residual_tokens, value_dim)
        )
        cursor += value_residual_bytes
        if cursor != len(payload):
            raise ValueError("Serialized payload length does not match encoded block dimensions.")
        return cls(
            header=BlockHeader(
                block_start=block_start,
                block_len=block_len,
                anchor_key=anchor_key,
                anchor_value=anchor_value,
                key_scale=key_scale,
                value_scale=value_scale,
                rho_upper=rho_upper,
                nu_upper=nu_upper,
            ),
            key_residuals=key_residuals,
            value_residuals=value_residuals,
        )

    def actual_serialized_bytes(self) -> int:
        return len(self.serialize())

    def actual_python_object_bytes(self) -> int:
        return (
            sys.getsizeof(self)
            + sys.getsizeof(self.header)
            + sys.getsizeof(self.header.anchor_key)
            + sys.getsizeof(self.header.anchor_value)
            + sys.getsizeof(self.key_residuals)
            + sys.getsizeof(self.value_residuals)
        )

    def total_bytes(self) -> int:
        return self.actual_serialized_bytes()


@dataclass(frozen=True)
class SerializedBlockContainer:
    key_dim: int
    value_dim: int
    buffer: bytes
    block_offsets: tuple[int, ...]
    block_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.block_count == 0:
            raise ValueError("Empty serialized block containers are not supported.")
        if self.key_dim <= 0 or self.value_dim <= 0:
            raise ValueError("Container dimensions must be positive for non-empty containers.")
        if len(self.block_offsets) != len(self.block_lengths):
            raise ValueError("Container offsets and lengths must have matching lengths.")
        if len(self.buffer) < _BLOCK_CONTAINER_HEADER.size:
            raise ValueError("Serialized block container is too short for its header.")

    @property
    def block_count(self) -> int:
        return len(self.block_offsets)

    @property
    def header_bytes(self) -> int:
        return _BLOCK_CONTAINER_HEADER.size

    @property
    def index_bytes(self) -> int:
        return self.block_count * _BLOCK_CONTAINER_INDEX_ENTRY.size

    @property
    def payload_bytes(self) -> int:
        return sum(self.block_lengths)

    @property
    def padding_bytes(self) -> int:
        return 0

    @property
    def total_bytes(self) -> int:
        return len(self.buffer)

    def payload_for_block(self, block_index: int) -> bytes:
        if block_index < 0 or block_index >= self.block_count:
            raise IndexError("Block index is out of range for the serialized container.")
        offset = self.block_offsets[block_index]
        block_length = self.block_lengths[block_index]
        return self.buffer[offset : offset + block_length]

    def deserialize_block(self, block_index: int) -> CompressedBlock:
        return CompressedBlock.deserialize(
            self.payload_for_block(block_index),
            key_dim=self.key_dim,
            value_dim=self.value_dim,
        )


def serialize_block_container(blocks: Sequence[CompressedBlock]) -> SerializedBlockContainer:
    if not blocks:
        raise ValueError("Cannot serialize an empty block container.")

    key_dim = int(blocks[0].header.anchor_key.shape[0])
    value_dim = int(blocks[0].header.anchor_value.shape[0])
    for block in blocks:
        if int(block.header.anchor_key.shape[0]) != key_dim:
            raise ValueError("All serialized blocks must share the same key dimension.")
        if int(block.header.anchor_value.shape[0]) != value_dim:
            raise ValueError("All serialized blocks must share the same value dimension.")

    payloads = [block.serialize() for block in blocks]
    header = _BLOCK_CONTAINER_HEADER.pack(_BLOCK_CONTAINER_MAGIC, len(payloads), key_dim, value_dim)
    payload_start = _BLOCK_CONTAINER_HEADER.size + len(payloads) * _BLOCK_CONTAINER_INDEX_ENTRY.size
    offsets: list[int] = []
    lengths: list[int] = []
    cursor = payload_start
    index_entries = []
    for payload in payloads:
        offsets.append(cursor)
        lengths.append(len(payload))
        index_entries.append(_BLOCK_CONTAINER_INDEX_ENTRY.pack(cursor, len(payload)))
        cursor += len(payload)
    buffer = b"".join([header, *index_entries, *payloads])
    if len(buffer) != cursor:
        raise RuntimeError("Serialized block container length does not match the computed payload cursor.")
    return SerializedBlockContainer(
        key_dim=key_dim,
        value_dim=value_dim,
        buffer=buffer,
        block_offsets=tuple(offsets),
        block_lengths=tuple(lengths),
    )


def deserialize_block_container(payload: bytes) -> SerializedBlockContainer:
    if len(payload) < _BLOCK_CONTAINER_HEADER.size:
        raise ValueError("Serialized block container is too short.")
    magic, block_count, key_dim, value_dim = _BLOCK_CONTAINER_HEADER.unpack(payload[: _BLOCK_CONTAINER_HEADER.size])
    if magic != _BLOCK_CONTAINER_MAGIC:
        raise ValueError("Serialized block container magic does not match the expected format.")
    if block_count <= 0:
        raise ValueError("Serialized block container must contain at least one block.")
    if key_dim <= 0 or value_dim <= 0:
        raise ValueError("Serialized block container dimensions must be positive.")
    cursor = _BLOCK_CONTAINER_HEADER.size
    index_end = cursor + block_count * _BLOCK_CONTAINER_INDEX_ENTRY.size
    if index_end > len(payload):
        raise ValueError("Serialized block container is truncated in its index table.")
    offsets: list[int] = []
    lengths: list[int] = []
    previous_end = index_end
    for _ in range(block_count):
        next_cursor = cursor + _BLOCK_CONTAINER_INDEX_ENTRY.size
        offset, block_length = _BLOCK_CONTAINER_INDEX_ENTRY.unpack(payload[cursor:next_cursor])
        if block_length <= 0:
            raise ValueError("Serialized block container block lengths must be positive.")
        if offset < index_end:
            raise ValueError("Serialized block container payload offsets must begin after the full index table.")
        if offset < previous_end:
            raise ValueError("Serialized block container payload offsets must be monotonic and non-overlapping.")
        if offset + block_length > len(payload):
            raise ValueError("Serialized block container index points outside the buffer.")
        offsets.append(offset)
        lengths.append(block_length)
        previous_end = offset + block_length
        cursor = next_cursor
    if previous_end != len(payload):
        raise ValueError("Serialized block container has trailing or unreachable bytes.")
    return SerializedBlockContainer(
        key_dim=key_dim,
        value_dim=value_dim,
        buffer=payload,
        block_offsets=tuple(offsets),
        block_lengths=tuple(lengths),
    )


def _quantize_residuals(residuals: np.ndarray) -> tuple[np.float16, np.ndarray]:
    if residuals.size == 0:
        return np.float16(1.0), residuals.astype(np.int8)
    max_abs = float(np.max(np.abs(residuals)))
    if max_abs == 0.0:
        return np.float16(1.0), np.zeros_like(residuals, dtype=np.int8)
    raw_scale = max_abs / 127.0
    if not np.isfinite(raw_scale):
        raise OverflowError("Residual scale overflowed to a non-finite value.")
    if raw_scale > float(np.finfo(np.float16).max):
        raise OverflowError("Residual scale exceeds the representable float16 range.")
    scale = np.float16(raw_scale)
    if float(scale) == 0.0:
        scale = np.nextafter(np.float16(0.0), np.float16(np.inf), dtype=np.float16)
    quantized = np.rint(residuals / float(scale)).clip(-127, 127).astype(np.int8)
    return scale, quantized


def _rigorous_block_metadata(
    block: CompressedBlock,
    *,
    precision: int,
) -> tuple[np.float32, np.float32]:
    reconstructed_keys, reconstructed_values = block.decode_block()
    anchor_exact = exact_vector(block.header.anchor_key, precision=precision)

    key_norm_bounds: list[gmpy2.mpfr] = []
    for row in reconstructed_keys:
        row_exact = exact_vector(row, precision=precision)
        squared_sum = exact_mpfr(0, precision=precision)
        for component, anchor_component in zip(row_exact, anchor_exact):
            diff_lower = rounded_sub(component, anchor_component, precision=precision, round_mode=gmpy2.RoundDown)
            diff_upper = rounded_sub(component, anchor_component, precision=precision, round_mode=gmpy2.RoundUp)
            abs_lower = -diff_lower if diff_lower < 0 else diff_lower
            abs_upper = -diff_upper if diff_upper < 0 else diff_upper
            abs_diff_upper = max_exact([abs_lower, abs_upper], precision=precision)
            squared = rounded_mul(abs_diff_upper, abs_diff_upper, precision=precision, round_mode=gmpy2.RoundUp)
            squared_sum = rounded_add(squared_sum, squared, precision=precision, round_mode=gmpy2.RoundUp)
        key_norm_bounds.append(rounded_sqrt(squared_sum, precision=precision, round_mode=gmpy2.RoundUp))

    value_norm_bounds: list[gmpy2.mpfr] = []
    for row in reconstructed_values:
        row_exact = exact_vector(row, precision=precision)
        value_norm_bounds.append(norm_upper(row_exact, precision=precision))

    rho_upper = max_exact(key_norm_bounds, precision=precision)
    nu_upper = max_exact(value_norm_bounds, precision=precision)
    return (
        mpfr_to_proven_float32_upper(rho_upper),
        mpfr_to_proven_float32_upper(nu_upper),
    )


def encode_block(keys: np.ndarray, values: np.ndarray, *, block_start: int, precision: int = DEFAULT_PRECISION) -> CompressedBlock:
    if keys.ndim != 2 or values.ndim != 2:
        raise ValueError("Keys and values must be rank-2 arrays.")
    if keys.shape[0] != values.shape[0]:
        raise ValueError("Keys and values must have the same block length.")
    if keys.shape[0] == 0:
        raise ValueError("Cannot encode an empty block.")
    if block_start < 0:
        raise ValueError("block_start must be nonnegative.")

    key_data = np.asarray(keys, dtype=np.float64)
    value_data = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(key_data)) or not np.all(np.isfinite(value_data)):
        raise ValueError("Keys and values must be finite.")
    anchor_key = key_data[0].astype(np.float16)
    anchor_value = value_data[0].astype(np.float16)

    key_residuals = key_data[1:] - anchor_key.astype(np.float64)
    value_residuals = value_data[1:] - anchor_value.astype(np.float64)

    key_scale, q_keys = _quantize_residuals(key_residuals)
    value_scale, q_values = _quantize_residuals(value_residuals)

    provisional = CompressedBlock(
        header=BlockHeader(
            block_start=block_start,
            block_len=int(key_data.shape[0]),
            anchor_key=anchor_key,
            anchor_value=anchor_value,
            key_scale=key_scale,
            value_scale=value_scale,
            rho_upper=np.float32(0.0),
            nu_upper=np.float32(0.0),
        ),
        key_residuals=q_keys,
        value_residuals=q_values,
    )

    rho_upper, nu_upper = _rigorous_block_metadata(provisional, precision=precision)

    finalized_header = BlockHeader(
        block_start=block_start,
        block_len=int(key_data.shape[0]),
        anchor_key=anchor_key,
        anchor_value=anchor_value,
        key_scale=key_scale,
        value_scale=value_scale,
        rho_upper=rho_upper,
        nu_upper=nu_upper,
    )

    return CompressedBlock(header=finalized_header, key_residuals=q_keys, value_residuals=q_values)
