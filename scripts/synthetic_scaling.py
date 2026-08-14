"""Measure how tasks and peak memory grow with scene count, on synthetic data.

``scripts/measure_memory_scaling.py`` answered this question against a 0.25
degree AOI over Philadelphia, on the assumption that "the **ratios** are what
transfer". They do not. Below roughly one degree the whole time stack fits in
RAM and dask never streams; a five degree tile streams from the first block. The
old script measures a regime production never runs in, so its ratios describe
somebody else's pipeline. It is deprecated in favour of this one. See ADR-011.

What transfers is geometry. Peak memory during de-striping is set by the chunk
edge, the thread count, and the depth of the time axis, none of which need real
pixels behind them. So :mod:`landsat_lst.benchmarks` builds the stack out of
``dask.array.random`` at production chunking and runs the real graphs against
it: same shape, same memory curve, no STAC query and no egress.

**Run the real sweep on a VM.** The dev box carries less memory than a
production VM, so the ceiling under test is unreachable there, and the answer is
about production hardware. This script refuses more than 200 scenes locally for
the same reason ``plan`` carries ``--max-tasks``: building a graph allocates
Python objects whether or not you compute it, and an unbounded local build has
taken a 64 GB desktop down.

    uv run python scripts/synthetic_scaling.py --distributed   # the real sweep
    uv run python scripts/synthetic_scaling.py --fetch <run-id>
    uv run python scripts/synthetic_scaling.py --scenes 50 100 --blocks 4

Everything here is a thin wrapper over ``landsat-lst benchmark``, which is what
the VM runs: Coiled ships the installed package, not this directory, so the
sweep itself has to live in the package for both tiers to run the same code.
"""

from __future__ import annotations

import argparse
import sys

from landsat_lst.benchmarks import DEFAULT_SWEEP_SCENES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, nargs="+", default=list(DEFAULT_SWEEP_SCENES))
    parser.add_argument(
        "--blocks",
        type=int,
        default=8,
        help="Blocks per side. 36 is a real 5-degree tile at chunk 512.",
    )
    parser.add_argument("--chunk", type=int, default=512, help="Spatial chunk edge in px")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Run on one Coiled VM of the production instance type",
    )
    parser.add_argument("--fetch", default=None, help="Read a published sweep back by run id")
    parser.add_argument("--force-local", action="store_true", help="Override the local ceiling")
    parser.add_argument("--out", default=None, help="Where to write the JSON")
    args = parser.parse_args()

    argv = ["benchmark"]
    for n in args.scenes:
        argv += ["--scenes", str(n)]
    argv += [
        "--blocks",
        str(args.blocks),
        "--chunk",
        str(args.chunk),
        "--threads",
        str(args.threads),
    ]
    if args.distributed:
        argv.append("--distributed")
    if args.fetch:
        argv += ["--fetch", args.fetch]
    if args.force_local:
        argv.append("--force-local")
    if args.out:
        argv += ["--out", args.out]

    from landsat_lst.cli import main as cli  # noqa: PLC0415

    # standalone_mode=False so click's SystemExit does not swallow the traceback
    # of a genuine failure behind an exit code.
    sys.exit(cli(argv, standalone_mode=False) or 0)


if __name__ == "__main__":
    main()
