"""Fixed-transform, independently decodable RACK-KV V2 codecs.

V1 remains in :mod:`rack_kv.codec` unchanged.  This module stores all state
needed to decode one block locally and never uses a neighboring block.
"""
from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Mapping

import numpy as np


_MAGIC = b"RKV2"
_HEADER = struct.Struct("<4sIHHHHBBHIIII")
_CODECS = {"groupwise_int8": 1, "hadamard_int8": 2, "hadamard_int4": 3}
_CODECS_INV = {value: key for key, value in _CODECS.items()}


def hadamard_transform(values: np.ndarray) -> np.ndarray:
    """Apply the normalized orthogonal Walsh-Hadamard transform."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1] <= 0 or array.shape[-1] & (array.shape[-1] - 1):
        raise ValueError("Hadamard dimension must be a positive power of two")
    if array.shape[0] == 0:
        return array.copy()
    result = array.copy()
    width = 1
    while width < result.shape[-1]:
        reshaped = result.reshape(*result.shape[:-1], -1, 2 * width)
        left = reshaped[..., :width].copy()
        right = reshaped[..., width : 2 * width].copy()
        reshaped[..., :width] = left + right
        reshaped[..., width : 2 * width] = left - right
        width *= 2
    return result / np.sqrt(result.shape[-1])


def _pack_int4(values: np.ndarray) -> bytes:
    integers = np.asarray(values, dtype=np.int8).reshape(-1)
    if np.any(integers < -8) or np.any(integers > 7):
        raise ValueError("INT4 values must lie in [-8, 7]")
    unsigned = integers.astype(np.int16) & 0x0F
    if len(unsigned) % 2:
        unsigned = np.concatenate([unsigned, np.zeros(1, dtype=np.int16)])
    packed = unsigned[::2] | (unsigned[1::2] << 4)
    return packed.astype(np.uint8).tobytes()


def _unpack_int4(payload: bytes, count: int) -> np.ndarray:
    packed = np.frombuffer(payload, dtype=np.uint8)
    values = np.empty(len(packed) * 2, dtype=np.int8)
    values[0::2] = (packed & 0x0F).astype(np.int8)
    values[1::2] = ((packed >> 4) & 0x0F).astype(np.int8)
    values[values >= 8] -= 16
    return values[:count]


def pack_int4(values: np.ndarray) -> bytes:
    return _pack_int4(values)


def unpack_int4(payload: bytes, count: int) -> np.ndarray:
    return _unpack_int4(payload, count)


def _quantize(values: np.ndarray, *, bits: int, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    if bits not in (4, 8):
        raise ValueError("only INT4 and INT8 are supported")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    rows, dim = values.shape
    group_count = (dim + group_size - 1) // group_size
    limit = 7 if bits == 4 else 127
    scales = np.ones(group_count, dtype=np.float16)
    quantized = np.zeros_like(values, dtype=np.int8)
    for group in range(group_count):
        start, end = group * group_size, min(dim, (group + 1) * group_size)
        maximum = float(np.max(np.abs(values[:, start:end]))) if rows else 0.0
        raw_scale = maximum / limit
        scale = np.float16(raw_scale if raw_scale > 0.0 else 1.0)
        if float(scale) == 0.0:
            scale = np.nextafter(np.float16(0.0), np.float16(np.inf), dtype=np.float16)
        scales[group] = scale
        quantized[:, start:end] = np.rint(values[:, start:end] / float(scale)).clip(-limit, limit).astype(np.int8)
    return scales, quantized


def _dequantize(scales: np.ndarray, quantized: np.ndarray, *, group_size: int) -> np.ndarray:
    result = np.empty_like(quantized, dtype=np.float64)
    for group, scale in enumerate(np.asarray(scales, dtype=np.float64)):
        start, end = group * group_size, min(result.shape[1], (group + 1) * group_size)
        result[:, start:end] = quantized[:, start:end].astype(np.float64) * float(scale)
    return result


def _encode_residual(residuals: np.ndarray, codec: str, group_size: int) -> tuple[np.ndarray, bytes]:
    transformed = hadamard_transform(residuals) if codec.startswith("hadamard_") else residuals
    bits = 4 if codec.endswith("int4") else 8
    scales, quantized = _quantize(transformed, bits=bits, group_size=group_size)
    payload = _pack_int4(quantized) if bits == 4 else quantized.tobytes(order="C")
    return scales, payload


@dataclass(frozen=True)
class V2CompressedBlock:
    block_start: int
    block_len: int
    key_dim: int
    value_dim: int
    key_codec: str
    value_codec: str
    group_size: int
    key_anchor: np.ndarray
    value_anchor: np.ndarray
    key_scales: np.ndarray
    value_scales: np.ndarray
    key_payload: bytes
    value_payload: bytes

    def __post_init__(self) -> None:
        if self.key_codec not in _CODECS or self.value_codec not in _CODECS:
            raise ValueError("unsupported V2 codec tag")
        if self.block_len <= 0 or self.key_dim <= 0 or self.value_dim <= 0:
            raise ValueError("invalid V2 block dimensions")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")

    @property
    def key_residual_count(self) -> int:
        return (self.block_len - 1) * self.key_dim

    @property
    def value_residual_count(self) -> int:
        return (self.block_len - 1) * self.value_dim

    def _decode_residual(self, scales: np.ndarray, payload: bytes, *, dim: int, codec: str) -> np.ndarray:
        count = (self.block_len - 1) * dim
        if codec.endswith("int4"):
            quantized = _unpack_int4(payload, count).reshape(self.block_len - 1, dim)
        else:
            quantized = np.frombuffer(payload, dtype=np.int8).copy().reshape(self.block_len - 1, dim)
        transformed = _dequantize(scales, quantized, group_size=self.group_size)
        return hadamard_transform(transformed) if codec.startswith("hadamard_") else transformed

    def decode_block(self) -> tuple[np.ndarray, np.ndarray]:
        key_residuals = self._decode_residual(self.key_scales, self.key_payload, dim=self.key_dim, codec=self.key_codec)
        value_residuals = self._decode_residual(self.value_scales, self.value_payload, dim=self.value_dim, codec=self.value_codec)
        keys = np.vstack([self.key_anchor.astype(np.float64), self.key_anchor.astype(np.float64) + key_residuals])
        values = np.vstack([self.value_anchor.astype(np.float64), self.value_anchor.astype(np.float64) + value_residuals])
        return keys, values

    def storage_breakdown(self) -> Mapping[str, int]:
        header = _HEADER.size
        anchors = 2 * (self.key_dim + self.value_dim)
        scales = 2 * (len(self.key_scales) + len(self.value_scales))
        return {"headers": header, "anchors": anchors, "scales": scales, "packed_residual_payload": len(self.key_payload) + len(self.value_payload), "total": len(self.serialize())}

    def serialize(self) -> bytes:
        header = _HEADER.pack(_MAGIC, self.block_start, self.block_len, self.key_dim, self.value_dim,
                              self.group_size, _CODECS[self.key_codec], _CODECS[self.value_codec], 0,
                              len(self.key_scales), len(self.value_scales), len(self.key_payload), len(self.value_payload))
        return b"".join([
            header, self.key_anchor.astype(np.float16).tobytes(), self.value_anchor.astype(np.float16).tobytes(),
            self.key_scales.astype(np.float16).tobytes(), self.value_scales.astype(np.float16).tobytes(),
            self.key_payload, self.value_payload,
        ])

    @classmethod
    def deserialize(cls, payload: bytes) -> "V2CompressedBlock":
        if len(payload) < _HEADER.size:
            raise ValueError("V2 block payload is truncated")
        (magic, start, block_len, key_dim, value_dim, group_size, key_tag, value_tag, _reserved,
         key_scale_count, value_scale_count, key_payload_len, value_payload_len) = _HEADER.unpack(payload[:_HEADER.size])
        if magic != _MAGIC or key_tag not in _CODECS_INV or value_tag not in _CODECS_INV:
            raise ValueError("invalid V2 block header")
        cursor = _HEADER.size
        key_anchor = np.frombuffer(payload[cursor:cursor + 2 * key_dim], dtype=np.float16).copy(); cursor += 2 * key_dim
        value_anchor = np.frombuffer(payload[cursor:cursor + 2 * value_dim], dtype=np.float16).copy(); cursor += 2 * value_dim
        key_scales = np.frombuffer(payload[cursor:cursor + 2 * key_scale_count], dtype=np.float16).copy(); cursor += 2 * key_scale_count
        value_scales = np.frombuffer(payload[cursor:cursor + 2 * value_scale_count], dtype=np.float16).copy(); cursor += 2 * value_scale_count
        key_payload = bytes(payload[cursor:cursor + key_payload_len]); cursor += key_payload_len
        value_payload = bytes(payload[cursor:cursor + value_payload_len]); cursor += value_payload_len
        if cursor != len(payload):
            raise ValueError("V2 block payload length mismatch")
        return cls(start, block_len, key_dim, value_dim, _CODECS_INV[key_tag], _CODECS_INV[value_tag], group_size, key_anchor, value_anchor, key_scales, value_scales, key_payload, value_payload)


def encode_v2_block(keys: np.ndarray, values: np.ndarray, *, key_codec: str, value_codec: str, group_size: int = 16, block_start: int = 0) -> V2CompressedBlock:
    keys = np.asarray(keys, dtype=np.float64); values = np.asarray(values, dtype=np.float64)
    if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0] or keys.shape[0] == 0:
        raise ValueError("keys and values must be matching nonempty matrices")
    if keys.shape[1] & (keys.shape[1] - 1):
        raise ValueError("key dimension must be a power of two for Hadamard codecs")
    key_anchor = keys[0].astype(np.float16); value_anchor = values[0].astype(np.float16)
    key_scales, key_payload = _encode_residual(keys[1:] - key_anchor.astype(np.float64), key_codec, group_size)
    value_scales, value_payload = _encode_residual(values[1:] - value_anchor.astype(np.float64), value_codec, group_size)
    return V2CompressedBlock(int(block_start), int(keys.shape[0]), keys.shape[1], values.shape[1], key_codec, value_codec, group_size, key_anchor, value_anchor, key_scales, value_scales, key_payload, value_payload)


def choose_adaptive_codecs(keys: np.ndarray, values: np.ndarray, *, group_size: int = 16, quality_multiplier: float = 1.25) -> tuple[str, str]:
    """Choose K/V tags independently using a deterministic V1-relative envelope."""
    from .codec import encode_block
    reference = encode_block(keys, values, block_start=0).decode_block()
    choices = []
    for component, (original, reference_values) in enumerate(zip((keys, values), reference)):
        v1_error = float(np.max(np.linalg.norm(original - reference_values, axis=1)))
        envelope = max(v1_error * quality_multiplier, 1e-6)
        selected = "hadamard_int8"
        for candidate in ("hadamard_int4", "hadamard_int8", "groupwise_int8"):
            encoded = encode_v2_block(keys, values, key_codec=candidate if component == 0 else "hadamard_int8", value_codec=candidate if component == 1 else "hadamard_int8", group_size=group_size)
            decoded = encoded.decode_block()[component]
            if float(np.max(np.linalg.norm(original - decoded, axis=1))) <= envelope:
                selected = candidate
                break
        choices.append(selected)
    return choices[0], choices[1]
