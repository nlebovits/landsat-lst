"""Reproduction: a Frisky client nested in a Frisky worker task kills the scheduler.

Measured on frisky 0.7.2 with dask 2026.7.1 on 2026-09-08. Two workers; each
outer task opens ``frisky.Client(scheduler_address)`` and submits 24 blocks
plus one reduction to the shared scheduler. The inner tasks execute (their
``worker.exec.call`` spans appear on both workers), then the scheduler closes
and every outstanding future fails with::

    ConnectionError: Connection to scheduler closed before red-0 was
    delivered (scheduler recv ended: io error: early eof)

The same happens on the unix-socket and the TCP transport. This is the
blocker that keeps a shard's inner graph off the shared scheduler in issue
#155; the in-process alternative is :mod:`landsat_lst.innertrace`. Rerun this
script on each Frisky release; ``tests/integration/test_frisky_nested_client.py``
does the same under ``xfail(strict=True)`` so a fix is noticed.

    uv run python scripts/experimental/frisky_nested_client_repro.py
    uv run python scripts/experimental/frisky_nested_client_repro.py --transport unix
"""

from __future__ import annotations

# ruff: noqa: PLC0415
import argparse
import json
import sys
import time


def outer(i: int, scheduler_address: str) -> tuple:
    import os

    import frisky
    import numpy as np

    def block(n: int) -> tuple:
        return (os.getpid(), float(np.ones((256, 256)).sum()) + n)

    def reduce(*parts: tuple) -> tuple:
        return (os.getpid(), sum(p[1] for p in parts), sorted({p[0] for p in parts}))

    inner = frisky.Client(scheduler_address)
    parts = [inner.submit(block, n, key=f"blk-{i}-{n}") for n in range(24)]
    red = inner.submit(reduce, *parts, key=f"red-{i}")
    out = red.result()
    inner.close()
    return (i, os.getpid(), out)


def run(transport: str) -> dict:
    import frisky

    cluster = frisky.LocalCluster(
        n_workers=2,
        threads_per_worker=4,
        processes=True,
        transport=transport,
        dashboard_address="127.0.0.1:0",
        silence_summary=True,
    )
    client = cluster.get_client()
    address = client.scheduler_address
    started = time.time()
    outcome: dict = {
        "transport": transport,
        "frisky": frisky.__version__,
        "results": [],
        "errors": [],
    }
    futures = [client.submit(outer, i, address, key=f"outer-{i}") for i in range(2)]
    for future in futures:
        try:
            outcome["results"].append(future.result())
        except Exception as exc:
            outcome["errors"].append(f"{type(exc).__name__}: {exc}"[:300])
    outcome["wall_s"] = round(time.time() - started, 2)
    try:
        client.close()
        cluster.close()
    except Exception as exc:  # the scheduler may already be gone
        outcome["close_error"] = f"{type(exc).__name__}: {exc}"[:200]
    outcome["reproduced"] = any(
        "scheduler recv ended" in e or "closed" in e for e in outcome["errors"]
    )
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--transport", default="tcp", choices=["tcp", "unix"])
    args = parser.parse_args()
    outcome = run(args.transport)
    print(json.dumps(outcome, indent=2))
    print(
        "REPRODUCED" if outcome["reproduced"] else "NOT REPRODUCED (a Frisky fix may have landed)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
