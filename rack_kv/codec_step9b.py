"""Independent best-anchor, per-token, mixed-bit RACK-KV Step-9B codec."""
from __future__ import annotations

from dataclasses import dataclass
import struct
import numpy as np

from .codec_v2 import pack_int4, unpack_int4


_MAGIC = b"R9B1"
_HEADER = struct.Struct("<4sIHHHHBBHHIIII")
_POLICY = {"per_token_int4": 4, "per_token_int8": 8, "mixed": 0}
_POLICY_INV = {v: k for k, v in _POLICY.items()}


def _anchor_index(values: np.ndarray, mode: str, *, quantization_bits: int = 4) -> int:
    if mode == "first_token":
        return 0
    distances = np.linalg.norm(values - values[:, None, :], axis=2)
    if mode == "medoid":
        scores = np.sum(distances * distances, axis=1)
    elif mode == "minimax":
        scores = np.max(distances, axis=1)
    elif mode == "quantization_aware":
        scores = []
        limit = 7 if quantization_bits == 4 else 127
        for i in range(values.shape[0]):
            residual = values - values[i]
            scale = np.maximum(np.max(np.abs(residual), axis=1) / limit, 1e-7)
            q = np.rint(residual / scale[:, None]).clip(-limit, limit)
            scores.append(float(np.mean(np.linalg.norm(residual - q * scale[:, None], axis=1))))
        scores = np.asarray(scores)
    else:
        raise ValueError(f"unsupported anchor mode: {mode}")
    return int(np.argmin(scores))


def _quantize_row(row: np.ndarray, bits: int, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    limit = 7 if bits == 4 else 127
    scales = []
    q = np.empty(row.shape, dtype=np.int8)
    for start in range(0, row.shape[0], group_size):
        end = min(row.shape[0], start + group_size)
        scale = max(float(np.max(np.abs(row[start:end]))) / limit, 1e-7)
        scales.append(np.float16(scale))
        q[start:end] = np.rint(row[start:end] / float(scales[-1])).clip(-limit, limit).astype(np.int8)
    return np.asarray(scales, dtype=np.float16), q


def _decode_row(scales: np.ndarray, q: np.ndarray, group_size: int) -> np.ndarray:
    row = np.empty(q.shape, dtype=np.float64)
    for group, scale in enumerate(scales):
        start, end = group * group_size, min(row.shape[0], (group + 1) * group_size)
        row[start:end] = q[start:end].astype(np.float64) * float(scale)
    return row


@dataclass(frozen=True)
class Step9BBlock:
    block_start: int
    block_len: int
    key_dim: int
    value_dim: int
    key_policy: str
    value_policy: str
    key_anchor_index: int
    value_anchor_index: int
    group_size: int
    key_anchor: np.ndarray
    value_anchor: np.ndarray
    key_tags: np.ndarray
    value_tags: np.ndarray
    key_scales: tuple[np.ndarray, ...]
    value_scales: tuple[np.ndarray, ...]
    key_payload: bytes
    value_payload: bytes

    def _decode_component(self, anchor, anchor_index, tags, scales, payload, dim):
        output = np.empty((self.block_len, dim), dtype=np.float64)
        anchor = np.asarray(anchor, dtype=np.float64)
        output[anchor_index] = anchor
        cursor = 0
        scale_index = 0
        for row in range(self.block_len):
            if row == anchor_index:
                continue
            bits = int(tags[scale_index])
            count = dim if bits == 8 else (dim + 1) // 2
            raw = payload[cursor:cursor + count]
            cursor += count
            if bits == 8:
                q = np.frombuffer(raw, dtype=np.int8).copy()
            else:
                q = unpack_int4(raw, dim)
            output[row] = anchor + _decode_row(scales[scale_index], q, self.group_size)
            scale_index += 1
        return output

    def decode_block(self) -> tuple[np.ndarray, np.ndarray]:
        return (self._decode_component(self.key_anchor, self.key_anchor_index, self.key_tags, self.key_scales, self.key_payload, self.key_dim),
                self._decode_component(self.value_anchor, self.value_anchor_index, self.value_tags, self.value_scales, self.value_payload, self.value_dim))

    def storage_breakdown(self) -> dict[str, int]:
        header = _HEADER.size
        anchors = 2 * (self.key_dim + self.value_dim)
        anchor_indices = 2
        scales = 2 * sum(len(x) for x in self.key_scales + self.value_scales)
        tags = len(self.key_tags) + len(self.value_tags)
        payload = len(self.key_payload) + len(self.value_payload)
        return {"headers": header, "anchors": anchors, "anchor_indices": anchor_indices, "scales": scales, "bitwidth_tags": tags, "packed_residual_payload": payload, "total": len(self.serialize())}

    def serialize(self) -> bytes:
        header = _HEADER.pack(_MAGIC, self.block_start, self.block_len, self.key_dim, self.value_dim, self.group_size,
                              _POLICY[self.key_policy], _POLICY[self.value_policy], self.key_anchor_index,
                              self.value_anchor_index, len(self.key_payload), len(self.value_payload),
                              sum(len(x) for x in self.key_scales), sum(len(x) for x in self.value_scales))
        parts = [header, self.key_anchor.astype(np.float16).tobytes(), self.value_anchor.astype(np.float16).tobytes(),
                 self.key_tags.tobytes(), self.value_tags.tobytes()]
        parts += [b"".join(x.astype(np.float16).tobytes() for x in self.key_scales), b"".join(x.astype(np.float16).tobytes() for x in self.value_scales), self.key_payload, self.value_payload]
        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> "Step9BBlock":
        fields = _HEADER.unpack(data[:_HEADER.size])
        magic, start, length, kd, vd, group, kt, vt, kai, vai, kpl, vpl, ksc, vsc = fields
        if magic != _MAGIC:
            raise ValueError("invalid Step-9B magic")
        cursor = _HEADER.size
        ka = np.frombuffer(data[cursor:cursor + 2 * kd], dtype=np.float16).copy(); cursor += 2 * kd
        va = np.frombuffer(data[cursor:cursor + 2 * vd], dtype=np.float16).copy(); cursor += 2 * vd
        key_tags = np.frombuffer(data[cursor:cursor + max(length - 1, 0)], dtype=np.uint8).copy(); cursor += max(length - 1, 0)
        value_tags = np.frombuffer(data[cursor:cursor + max(length - 1, 0)], dtype=np.uint8).copy(); cursor += max(length - 1, 0)
        def scales(count, rows):
            nonlocal cursor
            flat = np.frombuffer(data[cursor:cursor + 2 * count], dtype=np.float16).copy(); cursor += 2 * count
            per_row = int(rows)
            return tuple(flat[i * per_row:(i + 1) * per_row] for i in range(max(length - 1, 0)))
        key_per_row = (kd + group - 1) // group
        value_per_row = (vd + group - 1) // group
        key_scales = scales(ksc, key_per_row)
        value_scales = scales(vsc, value_per_row)
        kp = bytes(data[cursor:cursor + kpl]); cursor += kpl
        vp = bytes(data[cursor:cursor + vpl]); cursor += vpl
        if cursor != len(data):
            raise ValueError("Step-9B payload length mismatch")
        return cls(start, length, kd, vd, _POLICY_INV[kt], _POLICY_INV[vt], kai, vai, group, ka, va, key_tags, value_tags, key_scales, value_scales, kp, vp)


def encode_step9b(keys: np.ndarray, values: np.ndarray, *, key_policy: str, value_policy: str, key_anchor_mode: str = "first_token", value_anchor_mode: str = "first_token", group_size: int = 128, mixed_threshold: float = 0.01, block_start: int = 0) -> Step9BBlock:
    keys, values = np.asarray(keys, dtype=np.float64), np.asarray(values, dtype=np.float64)
    if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]:
        raise ValueError("keys and values must be matching matrices")
    def encode_component(data, policy, anchor_mode):
        anchor_index = _anchor_index(data, anchor_mode, quantization_bits=4)
        anchor = data[anchor_index].astype(np.float16)
        tags, scales, payload = [], [], bytearray()
        for row in range(data.shape[0]):
            if row == anchor_index:
                continue
            residual = data[row] - anchor.astype(np.float64)
            bits = 4
            if policy == "per_token_int8":
                bits = 8
            elif policy == "mixed":
                s4, q4 = _quantize_row(residual, 4, group_size)
                err4 = np.linalg.norm(residual - _decode_row(s4, q4, group_size))
                s8, _ = _quantize_row(residual, 8, group_size)
                baseline = max(np.linalg.norm(residual), 1e-12)
                bits = 4 if err4 / baseline <= mixed_threshold else 8
                if bits == 8:
                    s4, q4 = s8, _quantize_row(residual, 8, group_size)[1]
            s, q = _quantize_row(residual, bits, group_size)
            tags.append(bits); scales.append(s); payload.extend(pack_int4(q) if bits == 4 else q.tobytes())
        return anchor_index, anchor, np.asarray(tags, dtype=np.uint8), tuple(scales), bytes(payload)
    kai, ka, kt, ks, kp = encode_component(keys, key_policy, key_anchor_mode)
    vai, va, vt, vs, vp = encode_component(values, value_policy, value_anchor_mode)
    return Step9BBlock(int(block_start), int(keys.shape[0]), keys.shape[1], values.shape[1], key_policy, value_policy, kai, vai, group_size, ka, va, kt, vt, ks, vs, kp, vp)
