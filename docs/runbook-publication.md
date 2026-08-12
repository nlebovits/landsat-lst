# Publication runbook

**Status: NOT YET EXECUTED.** Every step below is written down in advance. Nothing in this
sequence has been run against Source Cooperative, and nothing should be until the costing sweeps
finish. The tooling the steps call is built, tested, and offline-safe. The operation is deferred,
not the code.

**Target:** `s3://us-west-2.opendata.source.coop/nlebovits/landsat-lst/`, readable at
`https://data.source.coop/nlebovits/landsat-lst/`.

The two prerequisites that once blocked S3 and S5 are resolved and recorded under
[Resolved prerequisites](#resolved-prerequisites). Nothing blocks the sequence except running it.

---

## The sequence at a glance

| Step | Does what | Reaches the network |
|---|---|---|
| S1 | Rehearse on two or three Coiled-processed tiles | Coiled, our own S3 |
| S2 | Global run with resume | Coiled, our own S3 |
| S3 | Build the catalog from the finished COGs | S3 reads |
| S4 | Offline `rashid` check on the staging tree | No |
| S5 | Sync the staging tree to Source Cooperative | S3 writes |
| S6 | Live `rashid` pass against the read URL | HTTPS probes |
| S7 | Registry pull request | GitHub |

---

## Resolved prerequisites

Two gaps blocked S3 and S5 when this runbook was first written. Both are closed in code.

### The processing layout is the catalog layout

`StorageBackend.cog_key` writes `lst-p95-{window}/{tile}/{product}_{window}_{tile}.tif`, the
exact path the published item declares. The leading segment duplicates
`catalog.spec.spec_for_window(window).collection_id` rather than importing it, so workers do not
pay for the catalog stack; a unit test (`TestCatalogLayoutContract`) pins the two together and
fails the build if either side drifts. Point S2's processing at the destination bucket and
prefix, and the COGs land at their published paths. S5 then syncs metadata only.

### `catalog build --metadata-only` writes a JSON-only staging tree

The builder still reads every COG header from the source, so all the numbers the items carry
come from the files. It does not copy or download the COGs beside their items. Use it for S3
against the `s3://` source. A validator run over the resulting tree silently skips the COG byte
checks, which is the intended S4 behavior. Sample real tiles into the tree for byte coverage.

---

## S1. Rehearsal on two or three tiles

Prove the whole path on a handful of tiles before spending a global run on it. Pick tiles that
disagree with each other, for example one dense urban tile, one humid tropical tile, and one
sparse high-latitude tile.

```bash
landsat-lst process --tile N40W075
landsat-lst process --tile S05E035
landsat-lst process --tile N55E010
```

Pull those tiles down and build a catalog from the local copies, then run the full offline
validator with the bytes present. This is the one point in the sequence where every byte check
actually runs, so it is where a malformed COG gets caught.

```bash
landsat-lst catalog build \
  --source ./rehearsal-cogs \
  --out ./rehearsal-catalog \
  --thumbnail ./thumbnail.png
landsat-lst catalog validate ./rehearsal-catalog
```

Expect zero errors and exactly the accepted warning set, `PTL-AST-003` and `PTL-DAT-010`. Any
other warning id fails the gate and the command exits non-zero. Open one COG in
QGIS and confirm the composite reads in Celsius and lands in the range the tile deserves.

Go back to the pipeline, not forward to S2, if the rehearsal turns up anything.

### Cost check

S1 is also the measurement that sizes S2. Record wall-clock time per tile, worker memory, and
scene counts, and multiply out. The costing sweeps this runbook waits on are that multiplication.

## S2. Global run with resume

```python
from landsat_lst.job import generate_jobs, run_distributed
from landsat_lst.storage import get_storage

storage = get_storage()
done = storage.list_completed("2021-2025")
jobs = [job for job in generate_jobs() if job.tile.name not in done]
run_distributed(jobs)
```

`list_completed` reads one paginated listing and returns only the tiles that carry both assets, so
a tile that died between its two uploads is reprocessed rather than published half-finished. Run
the same three lines again after any interruption. It is the resume mechanism, and it is cheap
enough to be the normal way to start.

Expect failures at the tail. A tile that fails twice is a data problem, not a scheduling problem,
and it belongs in an issue rather than in a third retry.

## S3. Build the catalog

Build against the finished COGs, into a staging tree that holds metadata only. The
`--metadata-only` flag reads every COG header without downloading a single asset.

```bash
landsat-lst catalog build \
  --source s3://us-west-2.opendata.source.coop/nlebovits/landsat-lst \
  --out ./staging \
  --window 2021-2025 \
  --metadata-only
```

The thumbnail renders from each COG's coarsest overview over `/vsis3`, a few kilobytes per tile.
Pass `--thumbnail` instead to reuse one rendered earlier.

The builder reads each COG's header for its footprint, shape, and band statistics, which is a
range read of a few kilobytes per file. Those reads are unavoidable. The catalog states what is in
the data, so it has to look.

## S4. Offline validation of the staging tree

```bash
landsat-lst catalog validate ./staging --json > staging-report.json
```

Every pass here is offline. The schemas ship inside the `rashid` wheel and the data pass resolves
relative hrefs inside the tree.

**A missing asset skips its byte checks silently.** On a staging tree with no `.tif` files the
report comes back with zero errors and only `PTL-AST-003`. `PTL-DAT-010` disappears, and its
absence is the tell: the byte pass had nothing to read. Do not read that clean report as evidence
the COGs are sound. It is evidence they were not examined.

Dial the coverage up by placing a sample of real COGs into the staging tree at the paths the items
declare, then validating again. Every tile whose bytes are present gets checksummed, sized, and
opened, and `PTL-DAT-010` comes back for each `qa_count` among them. Sample enough tiles to cover
the shapes the run produced, including at least one tile that was reprocessed after a failure.

## S5. Sync to Source Cooperative

```bash
landsat-lst catalog publish ./staging \
  --remote s3://us-west-2.opendata.source.coop/nlebovits/landsat-lst/ \
  --profile source-coop \
  --dry-run
```

Read the plan. Every line names the key, the byte count, and the `Content-Type` the object will be
served under. Confirm that the COGs carry
`image/tiff; application=geotiff; profile=cloud-optimized`, because a client reads the header off
the response and will not treat any other string as a COG.

Drop `--dry-run` when the plan is right. The command re-sends JSON and markdown on every run and
skips an asset whose remote size already matches, so a republish after a metadata fix moves
kilobytes.

The COGs are already in place from the copy described in
[Open prerequisites](#open-prerequisites). If the staging tree holds sample COGs from S4, the
publish will send them, which is harmless duplication of bytes that are already correct. Remove
them first if the transfer is large enough to care about.

## S6. Live pass against the read URL

```bash
landsat-lst catalog validate ./staging --live \
  --live-base-url https://data.source.coop/nlebovits/landsat-lst/
```

This is the only step that probes the hosting server. It checks three things per host, whether a
ranged GET returns `206 Partial Content` with `Accept-Ranges: bytes`, whether CORS is configured
for browser reads, and whether a preflight allows the methods and request headers a range reader
needs. It also issues one HEAD per asset and compares `Content-Length` against the `file:size` the
item declares.

That last check is the reason to run this step at all. It verifies the published bytes against the
metadata for every asset in the catalog without transferring any of them.

**A CORS finding may be a 500 wearing a disguise.** `data.source.coop` returns its 5xx responses
without CORS headers, so a transient server error and a genuinely missing CORS policy produce the
same `PTL-LIV-003`. Re-run the pass before believing it. Treat a finding as real only when it
survives a second run minutes later.

## S7. Registry pull request

Add one file to `portolan-sdi/portolan-registry` at `catalogs/landsat-lst.yaml`:

```yaml
url: https://data.source.coop/nlebovits/landsat-lst/catalog.json
submitter_email: nlebovits@pm.me
```

Those are the only two fields to write. The entry schema sets `additionalProperties: false` and
every other field in it is filled in by the registry's own crawler from the catalog. A title or a
description added by hand will fail schema validation on the pull request.

The registry crawls the URL, so S6 has to pass first. Registering a catalog whose host fails the
live checks records a failure against the entry.
