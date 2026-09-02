# Repository agent instructions

## Mandatory performance-evidence policy

Before proposing, implementing, reviewing, or benchmarking a performance or cost
change, read and follow `docs/performance-evidence-policy.md`.

Validate the premise before the implementation: classify inputs, establish a
retained production baseline, calculate the maximum possible whole-pipeline value,
pre-register the cheapest representative discriminator and stop rule, and run that
discriminator before broad implementation or adversarial hardening. Keep local,
synthetic, sampled, and production claims separate.

Every cloud experiment requires a valid `LST_EVIDENCE_CONTRACT` and explicit
operator approval. The contract never grants permission to spend money. Retain code
identity, phase/CPU/RSS/I/O measurements, output equivalence, Coiled lifecycle and
billing evidence, hashes of manually exported metrics, and optional Frisky spans.
Use `landsat-lst evidence collect` for the canonical bundle.

Why this is mandatory: issue #124's 4.4x local result was null in production, and
issue #108's modeled 52% idle premise measured near 9%. In both cases the cheap
production discriminator existed before the long implementation and review.
