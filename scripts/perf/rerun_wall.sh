#!/usr/bin/env bash
# Recover ONLY the interleaved wall-clock comparison for candidate 1.
#
# The acceptance run of 2026-09-02 was taken on a laptop shared with seven or
# eight other agents at a load average near 13 of 16 cores, so its wall-clock
# rows carry a spread the contract invalidates and no speedup ratio was quoted
# from them. Everything else the experiment measured -- bit-exactness, the
# native-pass tally, the fused task count and peak VmHWM -- is robust to CPU
# contention and is already recorded.
#
# Run this on a quiet machine to recover the one number that is missing. It
# needs no other step: the offsets are already pinned in results/perf/ and the
# truth arrays rebuild themselves on the first run of each tile.
#
# Prerequisites: results/fixtures/{S30W065,N40W075}_2021-2025_n300_f8 present
# (see results/perf/REPRODUCE.md), and this worktree's commit checked out with
# candidate 1 applied. The script toggles pipeline.py between HEAD and the
# working tree itself, so run it with the candidate committed.
#
#   REPS=3 bash scripts/perf/rerun_wall.sh
#   uv run python scripts/perf/summarize.py
#
# Expect ~15 minutes on an idle 16-core machine. Check the reported spread: a
# row wider than 1.30x is invalid however quiet the machine looked.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "loadavg before: $(cat /proc/loadavg)"
REPS="${REPS:-3}" THREADS="${THREADS:-4}" bash scripts/perf/interleaved.sh
echo "loadavg after:  $(cat /proc/loadavg)"

uv run --frozen python scripts/perf/summarize.py
