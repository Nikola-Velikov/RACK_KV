# RACK-KV V2 Physical Payload Omission

## Architecture

`KVPayloadStore` is a new V2 file-backed container. Its JSON index is read at
open time and contains block IDs, token ranges, byte offsets, byte lengths, and
optional per-block certification metadata. It never contains compressed K/V
payload bytes.
Compressed leaf payloads remain in a separate contiguous region and are fetched
only by `read_block`/`decode_block`. The V1 serializer and container are
unchanged.

## Fail-Closed Eligibility

Physical omission is accepted only when the caller supplies a complete GQA
group, every mapped query head is represented, MPFR authorization succeeded,
no numerical fallback occurred, and the eligible set exactly equals the
complement of the required leaf set. Any failed condition loads all historical
blocks and records a fallback reason.

## Accounting

The store counts metadata bytes separately from payload reads. Payload counters
increment only after an indexed file read succeeds. Decode counters increment
only after `CompressedBlock.deserialize` runs. Blocks required by multiple
mapped query heads are decoded once and reused.

## Smoke Test

The model-free smoke test used three serialized blocks and a complete four-head
GQA group. One block was required and two were marked physically eligible; the
eligible block IDs were forbidden by the instrumented store. The physical path
read 45 of 135 payload bytes, avoided 90 bytes and two decodes, and produced a
maximum logical/physical output difference of zero. The forbidden blocks were
read zero times and decoded zero times. Peak RSS was approximately 430 MB.

These are implementation-smoke values, not scientific benchmark results.

## Limitations

The current production hierarchy and GQA replay still require the unfinished
complete-GQA validation evidence. This step does not claim end-to-end latency,
throughput, or final model-memory improvement. Physical omission remains
disabled for incomplete or uncertain GQA evidence.
