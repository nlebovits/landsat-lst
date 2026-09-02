#!/usr/bin/env bash
# Alternate the two arms of the composite experiment, one rep at a time.
#
# The contract asks for a quiescent machine (section 2) and invalidates a row
# whose wall-clock spread exceeds 1.3x (stop condition 7). This laptop was not
# quiescent -- another agent held two full cores throughout -- so the arms are
# interleaved instead of run in blocks, which makes any drift common to both
# rather than an advantage for whichever ran second.
#
# The arm is a working-tree edit to one file, so it is toggled by swapping
# pipeline.py between two saved copies. Each run records that file's sha256.
set -euo pipefail

REPS="${REPS:-3}"
THREADS="${THREADS:-4}"
CHUNK="${CHUNK:-512}"
FIXTURES=("S30W065_2021-2025_n300_f8" "N40W075_2021-2025_n300_f8")

PIPE=src/landsat_lst/pipeline.py
BASE=$(mktemp)
TREAT=$(mktemp)
trap 'cp "$TREAT" "$PIPE"; rm -f "$BASE" "$TREAT"' EXIT

git show HEAD:"$PIPE" > "$BASE"
cp "$PIPE" "$TREAT"

for rep in $(seq 0 $((REPS - 1))); do
  for arm in baseline treatment; do
    if [ "$arm" = baseline ]; then cp "$BASE" "$PIPE"; else cp "$TREAT" "$PIPE"; fi
    for fixture in "${FIXTURES[@]}"; do
      tile="${fixture%%_*}"
      uv run --frozen --no-sync python scripts/perf/composite_experiment.py sweep \
        --fixture "$fixture" --chunk "$CHUNK" --threads "$THREADS" \
        --arm "$arm" --reps 1 --rep-offset "$rep"
    done
  done
done
