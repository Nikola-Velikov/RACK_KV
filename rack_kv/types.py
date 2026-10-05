from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DecodeSchedule(str, Enum):
    LARGEST_U = "largest_u"
    LARGEST_U_TIMES_NU = "largest_u_times_nu"


class CertificateMode(str, Enum):
    RIGOROUS_REFERENCE = "rigorous_reference"
    FAST_FP64 = "fast_fp64"


@dataclass(frozen=True)
class ModelGeometry:
    num_layers: int = 32
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    key_dim: int = 128
    value_dim: int = 128

    @property
    def kv_group_size(self) -> int:
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads.")
        return self.num_attention_heads // self.num_kv_heads

    @property
    def original_bytes_per_token_per_head(self) -> int:
        return 2 * self.key_dim + 2 * self.value_dim

    @property
    def original_bytes_per_token_per_layer(self) -> int:
        return self.num_kv_heads * self.original_bytes_per_token_per_head

    @property
    def original_bytes_per_token_total(self) -> int:
        return self.num_layers * self.original_bytes_per_token_per_layer


LLAMA_3_1_8B_GEOMETRY = ModelGeometry()
