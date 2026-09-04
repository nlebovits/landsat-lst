"""VM body for the #125 staging throughput probe: write ~34 GB, read it back, delete it.

Measures the one number the #125 break-even turns on: sustained per-VM S3
throughput, write and read reported separately, on the offsets VM class, in
the production region and bucket, at the object size the staging design would
use.

Object size is not a free parameter. Phase A's block edge is
``normalization._io_block_edge`` at production geometry, which is 1024, and a
scene batch is one ``pipeline.TIME_CHUNK`` of 10. One staged object is
therefore ``1024 x 1024 x 10`` uint16, or 21.0 MB.

Two arms, because the production precedent and the staging design differ in
one respect that the decision depends on:

- ``memory``: ``put_object``/``get_object`` against RAM. What a staging
  implementation would do, and the arm the decision rule reads.
- ``file``: ``upload_file``/``download_file`` through local scratch, which is
  the path ``shard_tasks._assemble_ref`` already takes. Small, and present
  only to tie this probe to the 12-17 s climatology exchange it is checked
  against, and to price the EBS hop the memory arm skips.

Every arm deletes its own keys. The ``finally`` sweep deletes anything the
arms left, because an abandoned object under the production prefix is an
object a later listing reads as finished work.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3
import numpy as np

BUCKET = os.environ["PROBE_BUCKET"]
PREFIX = os.environ["PROBE_PREFIX"].rstrip("/")
REGION = os.environ["PROBE_REGION"]
MAIN_GB = float(os.environ.get("PROBE_MAIN_GB", "34"))
ANCHOR_GB = float(os.environ.get("PROBE_ANCHOR_GB", "4"))
THREADS = int(os.environ.get("PROBE_THREADS", "16"))

BLOCK = 1024
SCENES = 10
OBJ_BYTES = BLOCK * BLOCK * SCENES * 2
DISTINCT_BUFFERS = 16

client = boto3.client("s3", region_name=REGION)


def emit(**row: object) -> None:
    print(json.dumps(row), flush=True)


def buffers() -> list[bytes]:
    """Distinct payloads, cycled, so no layer can serve a repeat from a cache."""
    rng = np.random.default_rng(20260904)
    return [
        rng.integers(0, 65535, size=BLOCK * BLOCK * SCENES, dtype=np.uint16).tobytes()
        for _ in range(DISTINCT_BUFFERS)
    ]


def keys(arm: str, n: int) -> list[str]:
    return [f"{PREFIX}/{arm}/obj{i:05d}.raw" for i in range(n)]


def run_pool(fn, items: list, threads: int) -> tuple[float, int]:
    """Run ``fn`` over ``items`` in a fixed pool. Returns (elapsed_s, bytes)."""
    total = 0
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = [ex.submit(fn, item) for item in items]
        for fut in as_completed(futures):
            total += fut.result()
    return time.monotonic() - start, total


def delete(all_keys: list[str]) -> int:
    removed = 0
    for i in range(0, len(all_keys), 1000):
        batch = [{"Key": k} for k in all_keys[i : i + 1000]]
        resp = client.delete_objects(Bucket=BUCKET, Delete={"Objects": batch, "Quiet": True})
        removed += len(batch) - len(resp.get("Errors", []))
    return removed


def arm_memory(bufs: list[bytes], n: int) -> list[str]:
    ks = keys("memory", n)

    def put(item: tuple[int, str]) -> int:
        i, k = item
        body = bufs[i % DISTINCT_BUFFERS]
        client.put_object(Bucket=BUCKET, Key=k, Body=body)
        return len(body)

    def get(k: str) -> int:
        return len(client.get_object(Bucket=BUCKET, Key=k)["Body"].read())

    w_s, w_b = run_pool(put, list(enumerate(ks)), THREADS)
    emit(
        phase="arm",
        arm="memory",
        op="write",
        threads=THREADS,
        objects=n,
        object_mb=round(OBJ_BYTES / 1e6, 1),
        gb=round(w_b / 1e9, 2),
        elapsed_s=round(w_s, 1),
        mb_s=round(w_b / 1e6 / w_s, 1),
    )
    r_s, r_b = run_pool(get, ks, THREADS)
    emit(
        phase="arm",
        arm="memory",
        op="read",
        threads=THREADS,
        objects=n,
        object_mb=round(OBJ_BYTES / 1e6, 1),
        gb=round(r_b / 1e9, 2),
        elapsed_s=round(r_s, 1),
        mb_s=round(r_b / 1e6 / r_s, 1),
    )
    emit(
        phase="cycle",
        arm="memory",
        gb=round(w_b / 1e9, 2),
        write_s=round(w_s, 1),
        read_s=round(r_s, 1),
        total_s=round(w_s + r_s, 1),
    )
    return ks


def arm_file(bufs: list[bytes], n: int, scratch: Path) -> list[str]:
    """The ``_assemble_ref`` path: through local scratch, transfer-manager multipart."""
    ks = keys("file", n)
    src = scratch / "src.raw"
    src.write_bytes(bufs[0])

    def put(k: str) -> int:
        client.upload_file(str(src), BUCKET, k)
        return OBJ_BYTES

    def get(item: tuple[int, str]) -> int:
        i, k = item
        local = scratch / f"d{i:05d}.raw"
        client.download_file(BUCKET, k, str(local))
        size = local.stat().st_size
        local.unlink()
        return size

    w_s, w_b = run_pool(put, ks, THREADS)
    emit(
        phase="arm",
        arm="file",
        op="write",
        threads=THREADS,
        objects=n,
        gb=round(w_b / 1e9, 2),
        elapsed_s=round(w_s, 1),
        mb_s=round(w_b / 1e6 / w_s, 1),
    )
    r_s, r_b = run_pool(get, list(enumerate(ks)), THREADS)
    emit(
        phase="arm",
        arm="file",
        op="read",
        threads=THREADS,
        objects=n,
        gb=round(r_b / 1e9, 2),
        elapsed_s=round(r_s, 1),
        mb_s=round(r_b / 1e6 / r_s, 1),
    )
    src.unlink()
    return ks


def main() -> int:
    started = time.monotonic()
    written: list[str] = []
    try:
        emit(
            phase="start",
            bucket=BUCKET,
            prefix=PREFIX,
            region=REGION,
            object_mb=round(OBJ_BYTES / 1e6, 1),
            main_gb=MAIN_GB,
            anchor_gb=ANCHOR_GB,
            threads=THREADS,
        )
        bufs = buffers()
        n_main = int(MAIN_GB * 1e9 // OBJ_BYTES)
        n_anchor = int(ANCHOR_GB * 1e9 // OBJ_BYTES)
        written += arm_memory(bufs, n_main)
        with TemporaryDirectory(prefix="lst_probe_") as td:
            written += arm_file(bufs, n_anchor, Path(td))
        emit(phase="done", total_elapsed_s=round(time.monotonic() - started, 1))
    except Exception as exc:
        emit(phase="error", error=repr(exc)[:400])
        return 1
    finally:
        try:
            emit(phase="cleanup", deleted=delete(written), requested=len(written))
        except Exception as exc:
            emit(phase="cleanup_failed", error=repr(exc)[:400], keys=len(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
