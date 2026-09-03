# Performance evidence policy

Performance work in this repository is evidence work. This policy applies to any
claim or change involving runtime, latency, throughput, memory, I/O, concurrency,
scaling, cloud cost, or resource efficiency.

A plausible mechanism, detailed model, synthetic benchmark, local microbenchmark,
green test suite, or user-reported number may justify only a pre-registered
bounded discriminator and the smallest experimental treatment needed to run it.
None may justify production implementation, broad hardening, or a performance
claim.

This rule exists because expensive investigations repeatedly validated
implementations before validating their premises. Issue #124 found a 4.4x local
composite improvement that became a null result on one representative Coiled
shard. Issue #108 initially attributed 52% of cost to fleet idle from modeled
inputs; retained Coiled logs put the addressable share near 9%. In both cases the
cheap representative discriminator was available before the long review.

## Required lifecycle

### 1. Premise and instrumentation

Before changing optimization behavior:

1. Classify every load-bearing input as `measured`, `derived`, `assumed`,
   `user_reported`, or `unknown`. Every measured value must point to a retained
   machine-readable artifact.
2. State the baseline, target metric, minimum worthwhile effect, Amdahl ceiling,
   and uncertainty. Unknown production fractions must remain unknown.
3. Pre-register a contract based on
   `docs/templates/performance-experiment-contract.json`.
4. Name the exact representative real input, target environment, phases, metrics,
   raw artifacts, profiling method, repetitions, aggregation, output comparison,
   cheapest discriminator, and stop rule.
5. Add only the instrumentation, minimum correctness support, and smallest
   reversible experimental treatment needed to run that discriminator. Do not
   broaden or harden the treatment before the result.

Synthetic and modeled work is useful for graph shape, lower bounds, rejection of
impossible designs, and instrumentation development. It cannot satisfy the
real-data or target-environment gate.

### 2. Representative measurement

Run the discriminator before implementing or hardening the optimization.

The input must be real and immutably identified. The environment must be
production or explicitly production-representative. Record every known difference
from production. Profiling must address the claimed mechanism, and its overhead
must either be measured or held identical between baseline and treatment.

Write a result based on `docs/templates/performance-result.json`. It must retain:

- raw baseline and treatment observations, exactly as many as the contract
  pre-registered per arm, every one positive and finite;
- full baseline and treatment revisions, distinct from each other;
- exact input and environment identity;
- the pre-registered profiling or phase-timing artifact;
- what the run spent (`observed_cloud_cost_usd`, `observed_coiled_credits`),
  checked against the contract's caps: a run that exceeded its own cap is an
  invalid result, not a footnote;
- the recomputed effect and minimum-effect comparison (the pre-registered
  minimum effect is at least 5%, below which the stop rule cannot separate a
  change from placement noise);
- scientific output equivalence, whose report records `max_abs_diff` and a
  `passed` that must equal `max_abs_diff <= tolerance` from the contract;
- limitations and contrary observations;
- an unambiguous `stop` or `proceed` decision.

Use `landsat-lst evidence collect` to copy and hash the contract, result, raw
observations, profile, baseline, equivalence report, worker records, and requested
Coiled telemetry into a self-contained bundle.

### 3. Decision

Optimization implementation and implementation-focused adversarial review may
begin only after a validated evidence bundle records `decision: proceed`.

When the bundle records `decision: stop`, stop the optimization. Preserve and
merge the negative result as measurement evidence; do not rescue the idea by
changing the metric, threshold, input, or explanation after seeing the result.
A different claim requires a new contract.

A passing unit suite proves only the behavior observed by those tests. It is
never performance evidence.

## Truth and provenance rules

- Never enter a value from memory when a retained artifact is required.
- Never relabel modeled, derived, synthetic, or user-reported data as measured.
- Never report a selected successful run while omitting failed or contrary runs.
- Never convert a component result into an end-to-end claim without a measured
  production fraction and stated Amdahl ceiling.
- Report local, synthetic, sampled, production-representative, and production
  measurements separately.
- Baseline and treatment must use the same immutable inputs and measurement
  method. Interleave noisy local arms; run cloud arms sequentially.
- A result manifest is validated arithmetically against its retained observations.
  The evidence bundle hashes every supporting file so later edits fail validation.

## Cloud boundary

For cloud work, the contract must cap AWS dollars and Coiled credits, name the
exact launch command, retain a measured baseline, use full Git revisions, require
worker code identity, and define output equivalence. Contracts are limited to
$100 and 400 Coiled credits; use smaller limits whenever possible.

A contract is a prerequisite, **not authorization**. The operator must explicitly
approve every cloud launch.

The Claude Bash hook denies recognized cloud launches without a valid
`LST_EVIDENCE_CONTRACT`: direct; wrapped (`env`, `uv run`, `exec`, `nohup`,
`setsid`, `timeout`, `nice`, `stdbuf`, `sudo`); nested (`bash -c`, `eval`,
`ssh`, `xargs`); the Python API (`python -c`, a heredoc, or a script that names
`submit_batch`, `drive_tile`, `batch_run`, and their siblings); and `make`,
`just`, or `tox` recipes that do any of these. `landsat-lst benchmark` without
`--distributed` is the local CI tier and is allowed. The guard binds the
treatment revision to the checkout the command runs *from*, which must be clean
of tracked changes and of untracked files under `src/`, and both contract
revisions must exist as commits there. Workers emit the bound revision,
installed package version, and instrumentation-module digest; an operator's own
`LST_CODE_REVISION` reaches the VM only when no contract is set. Evidence
collection rejects missing or mismatched worker revisions.

The guard reads the command text. It cannot see a script it is not given, a
`Makefile` in a directory other than the command's, or a launch that hides
behind an alias. It is a seat belt for an agent, not a sandbox.

## Profiling boundary

Profiling is experiment-controlled because instrumentation can perturb the system
being measured. A production Batch submission enables Dask profiling only when
`LST_PROFILE_DASK` says so or the run passes `--max-scenes` (a sample exists to be
measured). A contract or explicit diagnostic command must request it and state
how observer effects are controlled.

Frisky can inspect a live Coiled Dask cluster through
`frisky.hijack(cluster.get_client())`. The production pipeline uses Coiled Batch
plain processes, so Frisky is not a drop-in observer for that path. Capture spans
before shutdown. Historical lifecycle logs and host metrics cannot be converted
retroactively into task spans.

## Pull request gate

Every human-authored PR must retain this declaration from the repository template:

```text
<!-- performance-evidence
stage: none
contract: n/a
evidence: n/a
-->
```

Valid stages are:

- `none`: no performance or cost claim;
- `governance`: evidence policy or tooling only;
- `instrumentation`: measurement-only work with a valid committed contract;
- `measurement`: a valid contract and evidence bundle, including negative results;
- `optimization`: implementation backed by a bundle whose validated decision is
  `proceed`.

CI rejects missing declarations, performance language declared as `none`,
invalid contracts, altered or incomplete bundles, and optimization PRs backed by
a `stop` decision.

## Agent enforcement

`AGENTS.override.md` is the durable Codex instruction surface and coexists with
the untracked `claude-mem` `AGENTS.md`. `CLAUDE.md` and the Claude SessionStart
hook carry the same mandatory gate for Claude/Fable. CI verifies these surfaces,
the PR template, validators, templates, and hooks remain wired.

No instruction or contract grants spending authority. The Coiled collector is
read-only; normal operator confirmation and quota controls still apply.
