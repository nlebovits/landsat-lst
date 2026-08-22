"""One arm of the composite-packing probe: K concurrent shards on one VM.

Runs on the Coiled VM. The arm launches ``K`` composite row bands as *separate
processes* -- not threads in one interpreter -- because that is what packing
would actually look like, and because ``VmHWM`` is a per-``mm`` high-water mark
that only a fresh ``execve`` reports honestly (``landsat_lst.benchmarks._peak_mb``
carries the same note: Linux does not reset ``ru_maxrss`` across ``execve``, so a
child forked from a fat parent reports the *parent's* peak).

Every arm reads a **disjoint** set of row bands, so no arm is served from a
previous arm's page cache and no source byte is requested twice across the
ladder. Bands are assigned by the driver and passed in whole.

The probe plan
--------------
The arm writes its own :class:`~landsat_lst.shards.TilePlan` under
``_shards/{PROBE_RUN_ID}/{tile}/``, copying ``scene_ids``, ``scene_times``,
``window`` and ``offset_factor`` verbatim from a frozen production plan and
replacing only ``bands``. Two consequences, both deliberate:

- :func:`landsat_lst.shard_tasks._offset_key` is built from
  ``(tile, window, factor, scene_ids)`` and nothing else, so the probe's
  composite shards read the **canonical** ``_offsets/`` record for the tile --
  run-independent, already merged, no offset pass paid here.
- :meth:`~landsat_lst.shards.TilePlan.from_dict` checks a digest over the
  chunking settings, so the plan is written *after* ``apply_shard_settings()``
  under the same ``shard_composite_chunk`` every child will run at. A plan
  written under a different chunk would be refused by its own children.

``run_composite_shard`` is reused verbatim: its signature takes ``run_id`` as a
plain string and resolves everything else from the plan at that root, so a
foreign run id needs no changes in ``src/``.

Output: one JSON line tagged ``"phase": "arm"``, collected by the driver.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

TILE = os.environ.get("PROBE_TILE", "S30W065")
SOURCE_ROOT = os.environ["PROBE_SOURCE_ROOT"]
RUN_ID = os.environ["PROBE_RUN_ID"]
ARM = int(os.environ.get("PROBE_ARM", "0"))
VM_TYPE = os.environ.get("PROBE_VM_TYPE", "unknown")
VM_HOURLY = float(os.environ.get("PROBE_VM_HOURLY", "0"))
CHUNK = int(os.environ.get("PROBE_CHUNK", "512"))
THREADS = int(os.environ.get("PROBE_THREADS", "4"))
CHILD_TIMEOUT_S = int(os.environ.get("PROBE_CHILD_TIMEOUT_S", "3600"))
MEM_POLL_S = float(os.environ.get("PROBE_MEM_POLL_S", "2"))

#: ``"start:stop,start:stop,..."`` -- every probe band, identical in every arm
#: so all arms share one plan (and so one band index means one set of rows
#: everywhere). The arm runs only the indices in ``PROBE_INDICES``.
ALL_BANDS = [tuple(int(v) for v in b.split(":")) for b in os.environ["PROBE_BANDS"].split(",")]
INDICES = [int(i) for i in os.environ["PROBE_INDICES"].split(",")]


def _meminfo() -> dict[str, int]:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            out[key] = int(rest.split()[0]) * 1024
    return out


def _net_rx_tx() -> tuple[int, int]:
    rx = tx = 0
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        name, _, rest = line.partition(":")
        if name.strip() == "lo":
            continue
        fields = rest.split()
        rx += int(fields[0])
        tx += int(fields[8])
    return rx, tx


class MemSampler(threading.Thread):
    """Whole-VM headroom at a fixed cadence.

    The minimum of ``MemAvailable`` is the acceptance number: an arm that runs
    fast at 3% free has not shown that K shards fit, it has shown that K shards
    nearly did not. Sampling rather than a single end-of-arm read, because the
    trough is transient -- it lands when every child's rechunk is resident at
    once, which is nowhere near the end.
    """

    def __init__(self, period: float) -> None:
        super().__init__(daemon=True)
        self.period = period
        self.stop_flag = threading.Event()
        self.total = _meminfo()["MemTotal"]
        self.samples: list[int] = []

    def run(self) -> None:
        while not self.stop_flag.wait(self.period):
            try:
                self.samples.append(_meminfo()["MemAvailable"])
            except OSError:  # pragma: no cover - procfs is not optional on Linux
                return

    def report(self) -> dict[str, float | int]:
        if not self.samples:
            return {"mem_total_bytes": self.total, "mem_samples": 0}
        low = min(self.samples)
        return {
            "mem_total_bytes": self.total,
            "mem_samples": len(self.samples),
            "min_avail_bytes": low,
            "min_headroom_frac": round(low / self.total, 4),
            "mean_avail_bytes": int(sum(self.samples) / len(self.samples)),
        }


_CHILD = """
    import json, os, resource, time
    from pathlib import Path

    def peak_mb():
        # VmHWM, not ru_maxrss: Linux carries ru_maxrss across execve, so a
        # child of a fat parent reports the parent's mark. Same reasoning and
        # same fallback as landsat_lst.benchmarks._peak_mb.
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) / 1024
        except OSError:
            pass
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    index = {index}
    t0 = time.monotonic()
    from landsat_lst.config import settings
    settings.shard_composite_chunk = {chunk}
    settings.dask_max_threads = {threads}
    from landsat_lst.shard_tasks import apply_shard_settings, load_context, run_composite_shard
    apply_shard_settings()

    ctx = load_context({run_id!r}, {tile!r})
    start, stop = ctx.plan.bands[index]
    # Decoded volume the load materializes for this band: odc-stac allocates
    # the FULL time axis and fills non-intersecting blocks with nodata (see the
    # driver's docstring), so this is (times, rows, width) per band, both bands,
    # uint16. It counts nodata fills; wire bytes are measured separately.
    times = len(ctx.plan.scene_times)
    width = ctx.plan.native_shape[1]
    decoded = times * (stop - start) * width * 2 * 2

    keys = run_composite_shard({run_id!r}, {tile!r}, index)
    r = resource.getrusage(resource.RUSAGE_SELF)
    print("PROBECHILD" + json.dumps({{
        "index": index,
        "rows": [start, stop],
        "times": times,
        "wall_s": round(time.monotonic() - t0, 1),
        "decoded_bytes": decoded,
        "cpu_s": round(r.ru_utime + r.ru_stime, 1),
        "peak_vmhwm_mb": round(peak_mb(), 1),
        "keys": len(keys),
    }}))
"""


def _write_probe_plan() -> dict:
    """Copy a frozen plan, replacing only where the bands are cut.

    Read-only against the source run: ``read_text`` twice, nothing written back
    under it. The probe's own root is the only thing this process creates.
    """
    from landsat_lst import shards  # noqa: PLC0415
    from landsat_lst.config import settings  # noqa: PLC0415
    from landsat_lst.shard_tasks import apply_shard_settings  # noqa: PLC0415
    from landsat_lst.storage import get_storage  # noqa: PLC0415

    settings.shard_composite_chunk = CHUNK
    apply_shard_settings()

    storage = get_storage()
    source = json.loads(storage.read_text(f"{SOURCE_ROOT}/plan.json"))
    items = storage.read_text(f"{SOURCE_ROOT}/items.json")

    source["bands"] = [list(b) for b in ALL_BANDS]
    source["band_shards"] = len(ALL_BANDS)
    source.pop("digest", None)
    plan = shards.TilePlan.from_dict(source)  # re-stamps under this process's settings

    root = shards.shard_root(RUN_ID, TILE)
    storage.write_text(shards.plan_key(root), json.dumps(plan.to_dict()))
    storage.write_text(shards.items_key(root), items)
    return {"root": root, "digest": plan.digest, "times": len(plan.scene_times)}


def _launch(index: int) -> subprocess.Popen:
    src = textwrap.dedent(
        _CHILD.format(index=index, chunk=CHUNK, threads=THREADS, run_id=RUN_ID, tile=TILE)
    )
    return subprocess.Popen(
        [sys.executable, "-c", src],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    row: dict = {
        "phase": "arm",
        "arm": ARM,
        "vm_type": VM_TYPE,
        "k": len(INDICES),
        "chunk": CHUNK,
        "threads_per_child": THREADS,
        "indices": INDICES,
    }
    try:
        row.update(_write_probe_plan())
    except Exception as exc:
        row["error"] = f"plan: {exc!r}"
        print(json.dumps(row), flush=True)
        return 0

    sampler = MemSampler(MEM_POLL_S)
    sampler.start()
    rx0, tx0 = _net_rx_tx()
    t0 = time.monotonic()

    procs = {i: _launch(i) for i in INDICES}
    children: list[dict] = []
    for index, proc in procs.items():
        child: dict = {"index": index}
        try:
            out, err = proc.communicate(timeout=CHILD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            child["error"] = "timeout"
        rc = proc.returncode
        child["returncode"] = rc
        # A child killed by the OOM reaper leaves an error row and nothing else;
        # the arm reports partial results and the ladder continues to the next
        # arm rather than aborting on one dead shard.
        if rc in (-9, 137):
            child["error"] = "killed (signal 9 -- OOM reaper is the usual cause)"
        line = next((ln for ln in (out or "").splitlines() if ln.startswith("PROBECHILD")), None)
        if line:
            child.update(json.loads(line[len("PROBECHILD") :]))
        elif "error" not in child:
            child["error"] = (err or "")[-800:] or "no report line"
        children.append(child)

    wall = time.monotonic() - t0
    sampler.stop_flag.set()
    rx1, tx1 = _net_rx_tx()

    ok = [c for c in children if "decoded_bytes" in c and not c.get("error")]
    walls = sorted(c["wall_s"] for c in ok)
    decoded = sum(c["decoded_bytes"] for c in ok)
    cpu = sum(c.get("cpu_s", 0.0) for c in ok)
    cost = VM_HOURLY * wall / 3600

    row.update(
        {
            "wall_s": round(wall, 1),
            "children": children,
            "failures": len(children) - len(ok),
            "decoded_bytes": decoded,
            "decoded_mb_s": round(decoded / wall / 1e6, 2) if wall else None,
            "wire_rx_bytes": rx1 - rx0,
            "wire_rx_mb_s": round((rx1 - rx0) / wall / 1e6, 2) if wall else None,
            "wire_tx_bytes": tx1 - tx0,
            "cpu_s": round(cpu, 1),
            "cpu_cores_busy": round(cpu / wall, 2) if wall else None,
            "peak_child_vmhwm_gb": (
                round(max(c.get("peak_vmhwm_mb", 0) for c in children) / 1024, 2)
                if children
                else None
            ),
            "child_wall_min_s": walls[0] if walls else None,
            "child_wall_median_s": walls[len(walls) // 2] if walls else None,
            "child_wall_max_s": walls[-1] if walls else None,
            "child_wall_spread": (round(walls[-1] / walls[0], 2) if walls and walls[0] else None),
            "arm_cost_usd": round(cost, 4),
            "usd_per_gb_decoded": (round(cost / (decoded / 1e9), 5) if decoded else None),
            **sampler.report(),
        }
    )
    print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
