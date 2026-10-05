from __future__ import annotations

from dataclasses import dataclass

from .types import ModelGeometry


def original_block_bytes(block_len: int, key_dim: int, value_dim: int) -> int:
    return block_len * (2 * key_dim + 2 * value_dim)


def compressed_block_bytes(block_len: int, key_dim: int, value_dim: int) -> int:
    if block_len <= 0:
        return 0
    anchor_bytes = 2 * key_dim + 2 * value_dim
    residual_bytes = (block_len - 1) * (key_dim + value_dim)
    scale_bytes = 2 + 2
    certificate_metadata_bytes = 4 + 4
    index_alignment_bytes = 4 + 2 + 2
    return anchor_bytes + residual_bytes + scale_bytes + certificate_metadata_bytes + index_alignment_bytes


@dataclass(frozen=True)
class HistoricalMemoryReport:
    geometry: ModelGeometry
    block_size: int
    recent_window: int
    total_context_tokens: int
    historical_tokens: int
    full_blocks: int
    remainder_tokens: int
    recent_exact_bytes: int
    original_historical_bytes: int
    estimated_packed_historical_bytes: int
    estimated_packed_total_bytes: int

    @property
    def compression_ratio(self) -> float:
        if self.estimated_packed_historical_bytes == 0:
            return 1.0
        return self.original_historical_bytes / self.estimated_packed_historical_bytes

    @property
    def compressed_historical_bytes(self) -> int:
        return self.estimated_packed_historical_bytes


def historical_memory_report(
    *,
    geometry: ModelGeometry,
    total_context_tokens: int,
    recent_window: int,
    block_size: int,
) -> HistoricalMemoryReport:
    historical_tokens = max(total_context_tokens - recent_window, 0)
    full_blocks = historical_tokens // block_size if block_size > 0 else 0
    remainder_tokens = historical_tokens % block_size if block_size > 0 else 0

    original_per_block_per_head = original_block_bytes(block_size, geometry.key_dim, geometry.value_dim)
    compressed_per_block_per_head = compressed_block_bytes(block_size, geometry.key_dim, geometry.value_dim)

    original_total = full_blocks * original_per_block_per_head
    compressed_total = full_blocks * compressed_per_block_per_head

    if remainder_tokens:
        original_total += original_block_bytes(remainder_tokens, geometry.key_dim, geometry.value_dim)
        compressed_total += compressed_block_bytes(remainder_tokens, geometry.key_dim, geometry.value_dim)

    multiplier = geometry.num_layers * geometry.num_kv_heads
    recent_exact_tokens = min(total_context_tokens, recent_window)
    recent_exact_bytes = recent_exact_tokens * geometry.original_bytes_per_token_total
    estimated_historical = compressed_total * multiplier
    return HistoricalMemoryReport(
        geometry=geometry,
        block_size=block_size,
        recent_window=recent_window,
        total_context_tokens=total_context_tokens,
        historical_tokens=historical_tokens,
        full_blocks=full_blocks,
        remainder_tokens=remainder_tokens,
        recent_exact_bytes=recent_exact_bytes,
        original_historical_bytes=original_total * multiplier,
        estimated_packed_historical_bytes=estimated_historical,
        estimated_packed_total_bytes=recent_exact_bytes + estimated_historical,
    )
