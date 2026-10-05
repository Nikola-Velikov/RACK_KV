# RACK-KV 2.0 Limitations

- The planned physical validation target was 22 representative positions, 5 layers, and 8 KV groups per position, for 880 complete GQA groups. The full sweep was not completed because of local compute/runtime/resource constraints. Physical-system results are therefore reported only on the fully completed subset, with the exact coverage stated.
- Zero-vote physical regions are not present in the worker union files; P(4/4) uses recorded candidate regions only.
- Hierarchy evidence is limited to 20 cases and avoided zero leaf checks.
- The compression certificate is formally safe but conservative.
- Results use five representative layers and 256-token traces.
- No complete end-to-end throughput or long-context claim is made.
