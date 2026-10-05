"""RACK-KV MVP prototype."""

from .accounting import HistoricalMemoryReport, compressed_block_bytes, original_block_bytes
from .certificate import (
    CertificateMode,
    CertificationResult,
    certify_progressive_skipping,
    exact_reference_output,
)
from .codec import BlockHeader, CompressedBlock, encode_block
from .llama_trace import (
    DEFAULT_LLAMA31_BASE_REPO,
    DEFAULT_PROMPT,
    MinimalTraceCaptureResult,
    load_attention_trace,
    query_head_to_kv_head,
    replay_compact_trace_outputs,
    run_minimal_llama31_capture,
)
from .types import DecodeSchedule, ModelGeometry

__all__ = [
    "BlockHeader",
    "CertificateMode",
    "CertificationResult",
    "CompressedBlock",
    "DecodeSchedule",
    "HistoricalMemoryReport",
    "DEFAULT_LLAMA31_BASE_REPO",
    "DEFAULT_PROMPT",
    "ModelGeometry",
    "MinimalTraceCaptureResult",
    "certify_progressive_skipping",
    "compressed_block_bytes",
    "encode_block",
    "exact_reference_output",
    "load_attention_trace",
    "original_block_bytes",
    "query_head_to_kv_head",
    "replay_compact_trace_outputs",
    "run_minimal_llama31_capture",
]
