"""Demonstration 1 for issue #155: one inner task, seen through its trace.

Runs a bounded synthetic composite shard as one Frisky future on a local
two-worker cluster, with the shard's own graphs on the in-process Frisky
scheduler, then reads the persisted trace back and explains one inner task:
what it depended on, when and where it ran, and every part of the delay from
``dask.compute`` entry to its start. The same shard runs a second time on the
threaded scheduler and the two rasters are compared pixel for pixel.

No network, no credentials, no cloud. Output goes to ``--out`` (default
``results/inner-visibility/<stamp>/``): the retained explanation, the
comparison, and the trace artifacts themselves.

    uv run python scripts/inner_visibility_demo.py
    uv run python scripts/inner_visibility_demo.py --scenes 48 --shape 512 1024
"""

from __future__ import annotations

# ruff: noqa: PLC0415
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUN_ID = "inner-visibility-demo"
TILE = "S30W065"
INDEX = 16


def run_bounded_shard(
    root: Path, *, mode: str, shape: tuple[int, int], scenes: int, outer_key: str
):
    """The body one outer future runs: a bounded synthetic shard under a trace."""
    import numpy as np
    import rasterio

    from landsat_lst import shards
    from landsat_lst.cog import lst_product, qa_product, write_intermediates_bounded
    from landsat_lst.config import settings
    from landsat_lst.innertrace import (
        InnerTrace,
        ShardIdentity,
        active_inner_trace,
        inner_scheduler,
        new_trace_id,
        outer_binding,
    )
    from landsat_lst.job import _encode_native
    from landsat_lst.models import ProcessingJob
    from landsat_lst.pipeline import compute_annual_composite
    from landsat_lst.profiling import synthetic_dataset
    from landsat_lst.progress import TileHeartbeat, timed_section
    from landsat_lst.storage import LocalStorage
    from landsat_lst.tiling import parse_tile_name

    settings.destripe = False
    storage = LocalStorage(Path(root))
    sink = None
    sink_error = None
    if mode == "frisky":
        try:
            from landsat_lst.innertrace import FriskySink

            sink = FriskySink()
        except Exception as exc:  # the sink is optional; its absence is recorded
            sink_error = f"{type(exc).__name__}: {exc}"
    trace_id = new_trace_id()
    stem = shards.unit_trace_prefix(RUN_ID, "composite", TILE, INDEX)
    identity = ShardIdentity(RUN_ID, "composite", TILE, INDEX, 1, trace_id, outer_key)
    job = ProcessingJob(tile=parse_tile_name(TILE), year=2021, end_year=2025)
    started = time.perf_counter()
    scratch = Path(tempfile.mkdtemp(prefix=f"lst_demo_{mode}_"))
    with (
        outer_binding(
            outer_key=outer_key,
            trace_id=trace_id,
            sink=sink,
            inner_scheduler=mode,
            sink_error=sink_error,
        ),
        inner_scheduler(mode=mode, threads=4) as sched,
    ):
        trace = InnerTrace(
            identity=identity, storage=storage, stem=stem, scheduler=sched, sink=sink, flush_s=5
        )
        heartbeat = TileHeartbeat(
            run_id=RUN_ID,
            job=job,
            storage=storage,
            attempt=1,
            key=shards.shard_state_key(shards.shard_root(RUN_ID, TILE), "composite", INDEX, 1),
        )
        with heartbeat, trace:
            heartbeat.attach("inner", trace.heartbeat_fields)
            assert active_inner_trace() is trace
            data = synthetic_dataset(shape=shape, scenes=scenes, chunk_size=128)
            with timed_section("composite_graph"):
                composite = compute_annual_composite(data)
            with trace.section("encode"):
                native = _encode_native(composite)
            paths = {"lst_p95": scratch / "lst_p95.tif", "qa_count": scratch / "qa_count.tif"}
            products = [
                lst_product(native, paths["lst_p95"]),
                qa_product(native, paths["qa_count"]),
            ]
            with timed_section("exporting"), trace.observe_computes():
                write_intermediates_bounded(
                    [(p.da, path) for p, path in zip(products, paths.values(), strict=True)],
                    longitude_group=256,
                    observer=trace.group_observer(),
                )
            pixels = {name: rasterio.open(path).read() for name, path in paths.items()}
            digests = {name: int(np.asarray(v, dtype=np.int64).sum()) for name, v in pixels.items()}
    shutil.rmtree(scratch, ignore_errors=True)
    snapshot = trace.snapshot()
    return {
        "mode": mode,
        "pid": os.getpid(),
        "wall_s": round(time.perf_counter() - started, 2),
        "tasks_started": snapshot["tasks_started"],
        "spill_count": snapshot["spill_count"],
        "groups": len(snapshot["groups"]),
        "errors": snapshot["errors"],
        "pixel_sums": digests,
        "pixels": {k: v.tolist() for k, v in pixels.items()},
        "trace_id": hex(trace_id),
        "sink_error": sink_error,
        "final_key": f"{stem}.inner.a01.final.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scenes", type=int, default=24)
    parser.add_argument("--shape", type=int, nargs=2, default=(256, 512), metavar=("ROWS", "COLS"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--task", default=None, help="Inner task key or substring to explain")
    args = parser.parse_args()

    import frisky
    import numpy as np

    from landsat_lst.futures_tasks import task_key
    from landsat_lst.shard_explain import explain_group, explain_task, load_group, pick_output_key
    from landsat_lst.storage import LocalStorage

    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (ROOT / "results" / "inner-visibility" / stamp)
    out.mkdir(parents=True, exist_ok=True)
    bucket_frisky = out / "bucket-frisky"
    bucket_threads = out / "bucket-threads"

    cluster = frisky.LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=True,
        transport="tcp",
        dashboard_address="127.0.0.1:0",
        silence_summary=True,
    )
    client = cluster.get_client()
    url = client.dashboard_link
    print(f"outer frisky cluster: {url}")
    key_frisky = task_key(RUN_ID, TILE, "composite", INDEX, "frisky01")
    key_threads = task_key(RUN_ID, TILE, "composite", INDEX, "thread01")
    fut_frisky = client.submit(
        run_bounded_shard,
        str(bucket_frisky),
        key=key_frisky,
        outer_key=key_frisky,
        mode="frisky",
        shape=tuple(args.shape),
        scenes=args.scenes,
    )
    result_frisky = fut_frisky.result()
    fut_threads = client.submit(
        run_bounded_shard,
        str(bucket_threads),
        key=key_threads,
        outer_key=key_threads,
        mode="threads",
        shape=tuple(args.shape),
        scenes=args.scenes,
    )
    result_threads = fut_threads.result()
    time.sleep(1.0)
    outer_shard_spans = frisky.query_spans(name="shard.", limit=10000, dashboard_url=url)
    outer_exec = frisky.query_spans(name="worker.exec.call", limit=10000, dashboard_url=url)
    client.close()
    cluster.close()

    same = {
        name: bool(
            np.array_equal(
                np.asarray(result_frisky["pixels"][name]),
                np.asarray(result_threads["pixels"][name]),
            )
        )
        for name in ("lst_p95", "qa_count")
    }
    storage = LocalStorage(bucket_frisky)
    group0 = load_group(storage, RUN_ID, "composite", TILE, INDEX, attempt=1, group=0)
    assert group0 is not None, "group 0 artifacts are missing"
    key = args.task or pick_output_key(group0)
    explanation = explain_group(group0) + "\n" + explain_task(group0, key, depth=4)
    (out / "explain-task.txt").write_text(explanation)
    comparison = {
        "frisky": {k: v for k, v in result_frisky.items() if k != "pixels"},
        "threads": {k: v for k, v in result_threads.items() if k != "pixels"},
        "pixels_identical": same,
        "outer_scheduler": {
            "dashboard": url,
            "shard_spans": sorted(
                {(s["name"], tuple(s.get("keys") or ())) for s in outer_shard_spans}
            ),
            "outer_exec_keys": sorted({str(s["keys"][0]) for s in outer_exec if s.get("keys")}),
        },
    }
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2, default=list))
    print(explanation)
    print(json.dumps({k: v for k, v in comparison.items() if k != "outer_scheduler"}, indent=2))
    print(f"outer shard.* spans: {len(outer_shard_spans)}; artifacts under {out}")
    ok = (
        all(same.values())
        and result_frisky["tasks_started"] > 0
        and outer_shard_spans
        and not result_frisky["errors"]
    )
    print("DEMONSTRATION 1:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
