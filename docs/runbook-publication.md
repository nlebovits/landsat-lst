# Publication runbook

**Status: NOT YET EXECUTED.** Every step below is written down in advance. Nothing in this
sequence has been run against Source Cooperative, and nothing should be until the costing sweeps
finish. The tooling the steps call is built, tested, and offline-safe. The operation is deferred,
not the code.

**Target:** `s3://us-west-2.opendata.source.coop/nlebovits/landsat-lst/`, readable at
`https://data.source.coop/nlebovits/landsat-lst/`.

**Two decisions are still open.** Both are described under [Open prerequisites](#open-prerequisites)
and both block S3 and S5. Read that section before scheduling anything.

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

## Open prerequisites

### The processing layout and the catalog layout are not the same

`StorageBackend.cog_key` writes `{window}/{tile}/{product}_{window}_{tile}.tif` under
`settings.s3_prefix`. The catalog places the same file at
`{collection_id}/{tile}/{product}_{window}_{tile}.tif`, and `collection_id` is
`lst-p95-{window}`, not `{window}`. A tile S2 writes to
`landsat-lst/2021-2025/N40W075/lst_p95_2021-2025_N40W075.tif` has to appear at
`nlebovits/landsat-lst/lst-p95-2021-2025/N40W075/lst_p95_2021-2025_N40W075.tif` for the published
item to resolve.

One recursive server-side copy maps the whole tree, because the two layouts agree below their
first path segment:

```bash
aws s3 cp --recursive \
  s3://source-coop-radiant-earth/landsat-lst/2021-2025/ \
  s3://us-west-2.opendata.source.coop/nlebovits/landsat-lst/lst-p95-2021-2025/
```

Both buckets are in `us-west-2` and the copy is server side, so the bytes never reach a local
machine and there is no egress charge. Decide before S2 whether to run that copy or to point
processing at the destination bucket directly. Pointing processing at the destination still needs
the prefix rewrite, because `cog_key` inserts the bare window label.

### `catalog build` materialises the COGs beside the items

`catalog.scan.place_file` downloads each source COG into the tree it is building. Against an
`s3://` source that means the full dataset lands on the machine running the build, which is the
egress the whole design avoids. A JSON-only staging tree is not reachable through the current
builder without paying for that download.

Resolve it one of two ways before S3. Give the builder a mode that records assets it does not
copy, or run S3 against a working directory that already holds the COGs. Do not work around it by
building the full tree and deleting the `.tif` files afterwards. The download has already been
paid for by then.

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

Build against the finished COGs, into a staging tree that holds metadata only. Read
[Open prerequisites](#open-prerequisites) first, because the builder does not yet do this without
downloading the assets.

```bash
landsat-lst catalog build \
  --source s3://source-coop-radiant-earth/landsat-lst \
  --out ./staging \
  --window 2021-2025 \
  --thumbnail ./thumbnail.png
```

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
