# Provenance

Extracted on 2026-11-13 from a completed TraceLoom optimization investigation.
The intervention is a stock vLLM configuration, not a newly authored engine
algorithm. The source result SHA256 is recorded in `evidence/result.json`.
Source profiles, research archives and manuscript files remain in their
original custody; they are not redistributed in this code repository.

The installed API modifies only long_prefill_token_threshold.
All other kwargs are preserved. The baseline and candidate must use identical
workloads and engine binaries. The installer is inert and rollback uses the
caller's original kwargs.
