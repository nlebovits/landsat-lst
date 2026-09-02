# Performance evidence policy

Performance work in this repository is evidence work. A plausible mechanism, a
green local benchmark, or a detailed cost model is not enough to justify an
implementation effort or a production claim.

This rule exists because two expensive investigations validated implementations
before validating their premises. Issue #124 found a 4.4x local composite
improvement that became a null result on one representative Coiled shard. Issue
#108 initially attributed 52% of cost to fleet idle from modeled inputs; retained
Coiled logs put the addressable share near 9%. In both cases the cheap production
discriminator was available before the long review.

## Required order

Before implementing or substantially reviewing a performance change:

1. Classify every load-bearing input as `measured`, `derived`, `assumed`,
   `user_reported`, or `unknown`, with an artifact for every measured value.
2. State the production baseline, target metric, minimum worthwhile effect, and
   Amdahl ceiling. If the production fraction is unknown, say so; do not convert
   a component benchmark into a pipeline speedup.
3. Pre-register the cheapest representative discriminator and a stop rule in a
   contract based on `docs/templates/performance-experiment-contract.json`.
4. Run only the minimum correctness check needed to make that discriminator
   interpretable, then run the discriminator immediately.
5. Stop when the stop rule fires. Adversarial hardening, broad test matrices, and
   production implementation follow only after transfer to the target environment
   is demonstrated.

For cloud work, the contract must also cap AWS dollars and Coiled credits, name
the exact workload and launch command, require a retained baseline artifact and
full Git revisions, require code-identity evidence, and define output equivalence.
Experiment contracts are limited to $100 and 400 Coiled credits; smaller limits
should be used whenever the discriminator allows them.
The contract must name the output comparison method, its exact acceptance
criterion, and the path where the post-run result will be retained. Evidence
collection copies and hashes that result and the baseline artifact into the bundle.
Collection refuses an equivalence report unless it explicitly records `passed: true`.
The contract is a prerequisite, **not authorization**: the operator must still
explicitly approve every cloud launch.

## Evidence that every measured run must retain

- Git commit and package identity seen by the worker.
- Exact command, configuration, input/run identity, instance type, and timestamps.
- Wall and process CPU time, peak RSS, I/O bytes, phase timings, and task/profile
  summaries when the relevant execution path exposes them.
- Output-content checksums or a stronger scientific-equivalence comparison.
- Coiled cluster lifecycle, worker census/status, and cluster-scoped billing.
- Any console metric exports or screenshots, copied and hashed as attachments.
- Measurement limitations and all quantities still assumed or unknown.

Use `landsat-lst evidence collect` to create the canonical JSON bundle. Frisky is
the preferred optional lens for bounded diagnostic runs: its agent-readable spans
separate execution, scheduling, serialization, disk, and network work that ordinary
phase timing collapses together. `frisky.hijack(cluster.get_client())` works with a
live Coiled Dask cluster. The production pipeline currently uses Coiled Batch tasks,
however, and those plain processes do not register their work with a Dask scheduler;
Frisky therefore requires an explicit diagnostic Dask execution path and is not a
drop-in observer for the current production path.

Capture Frisky spans before the diagnostic cluster stops, then retain them with
`landsat-lst evidence capture-frisky`. Historical Coiled lifecycle logs, billing,
and host CPU/memory/network series are downloadable with `evidence collect`, but
they cannot be converted retroactively into Frisky task and transfer spans. Old
clusters that emitted no spans remain span-less. Frisky is not the production
scheduler, and its measurements must pass the same production-transfer gate.

## Review and reporting rules

- Report local, synthetic, sampled, and production measurements separately.
- Never describe a modeled value as measured or quote an all-in cost when any
  price term is unknown.
- Compare treatment with the exact retained baseline; interleave noisy local A/B
  runs and run cloud arms sequentially with identical inputs.
- Prefer one implementer and one independent reviewer. Permit one bounded
  correction round; unresolved findings become explicit follow-up work rather
  than an expanding investigation.
- A passing unit suite proves correctness only for what the tests observe. It is
  not performance evidence.

## Enforcement

`AGENTS.md` and `CLAUDE.md` carry the mandatory summary. Claude receives the same
policy at session start, and its Bash hook denies recognized cloud experiment
launches without a valid `LST_EVIDENCE_CONTRACT`. The hook does not approve the
launch. CI runs `scripts/check_evidence_policy.py` so removing this guidance,
hook registration, contract template, or composite profiling fails visibly.

The Cloud/Coiled evidence collector is read-only. Actual launches remain subject
to the repository's normal operator confirmation and quota controls.
