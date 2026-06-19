#!/usr/bin/env bash
# Phase 0 batch runner: export remaining validation tiles sequentially.
#
# Loops scripts/e2e_single_tile.py over the 5 remaining Phase 0 tiles,
# one at a time (4 workers x 10GB each = 40GB). Writes per-tile output to
# results/phase0/<tile>/ and a per-tile log. Continues on failure and
# records a pass/fail status line per tile in the combined batch log.
#
# Endpoint is pinned to Planetary Computer (free locally; AWS egress rule).
set -uo pipefail

cd "$(dirname "$0")/.."

# Remaining Phase 0 tiles (N40W075 done; S25E030/Durban and
# S35W070/Pergamino already inspected manually).
# All remaining phase-0 tiles, now that the refreshable /vsiaz auth removes the
# ~45min SAS-token wall. N25E080 (907 scenes, walled 3× before) is first so its
# crossing of 45min is the definitive proof. N60W150 already PASSED. See #31.
TILES=(
  N25E080
  N35W120
  N50E005
  S05W060
)

YEAR="${LST_YEAR:-2024}"
OUT_ROOT="results/phase0"
BATCH_LOG="${OUT_ROOT}/batch.log"

export LST_STAC_URL="https://planetarycomputer.microsoft.com/api/stac/v1"
export LST_YEAR="${YEAR}"
export LST_WORKERS=4
export LST_MEMORY=10GB
export LST_THREADS="${LST_THREADS:-2}"

mkdir -p "${OUT_ROOT}"
echo "=== Phase 0 batch start: ${#TILES[@]} tiles, year ${YEAR} ===" \
  | tee -a "${BATCH_LOG}"

# Retry each tile up to MAX_ATTEMPTS times, sleeping BACKOFF between
# failures. Survives transient Planetary Computer STAC timeouts and SAS
# token expiry: an overnight run keeps retrying until PC cooperates.
MAX_ATTEMPTS="${LST_MAX_ATTEMPTS:-12}"
BACKOFF="${LST_BACKOFF:-180}"

for tile in "${TILES[@]}"; do
  out_dir="${OUT_ROOT}/${tile}"
  mkdir -p "${out_dir}"

  attempt=1
  while [ "${attempt}" -le "${MAX_ATTEMPTS}" ]; do
    echo "--- ${tile} starting (attempt ${attempt}/${MAX_ATTEMPTS}) ---" \
      | tee -a "${BATCH_LOG}"
    start=$(date +%s)

    LST_TILE="${tile}" \
      uv run python scripts/e2e_single_tile.py \
        --output-dir "${out_dir}" \
        >"${out_dir}/run.log" 2>&1
    code=$?

    end=$(date +%s)
    elapsed=$((end - start))
    if [ "${code}" -eq 0 ]; then
      echo "--- ${tile} PASS in ${elapsed}s (attempt ${attempt}) ---" \
        | tee -a "${BATCH_LOG}"
      break
    fi

    echo "--- ${tile} FAIL(exit=${code}) in ${elapsed}s (attempt ${attempt}/${MAX_ATTEMPTS}, log: ${out_dir}/run.log) ---" \
      | tee -a "${BATCH_LOG}"
    cp "${out_dir}/run.log" "${out_dir}/run.attempt${attempt}.log" 2>/dev/null
    attempt=$((attempt + 1))
    if [ "${attempt}" -le "${MAX_ATTEMPTS}" ]; then
      sleep "${BACKOFF}"
    fi
  done
done

echo "=== Phase 0 batch complete ===" | tee -a "${BATCH_LOG}"
