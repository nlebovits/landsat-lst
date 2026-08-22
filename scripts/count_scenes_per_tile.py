"""Count scenes per production tile by paging Planetary Computer's STAC.

The cost projection needs real per-tile scene counts: the three-point
latitude interpolation predicted ~2,900 for S30W065 and the frozen plan
measured 4,403 -- scene density follows WRS-2 path geometry per tile, not
latitude. Earth Search returns ``numberMatched`` in one request but is
hook-blocked locally (egress rule); Planetary Computer returns no count,
so this pages with id-only fields, eight tiles at a time, and caches to
``results/probe/scene_counts.json`` so the sweep never reruns.

The filter mirrors ``pipeline.query_stac`` exactly: landsat-c2-l2, L8+L9,
eo:cloud_cover < settings.max_cloud_cover, the production window.

    python scripts/count_scenes_per_tile.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

OUT = HERE.parent / "results" / "probe" / "scene_counts.json"
PC = "https://planetarycomputer.microsoft.com/api/stac/v1"
WORKERS = 8
RETRIES = 4


def count_tile(bbox: tuple[float, float, float, float]) -> int:
    import pystac_client  # noqa: PLC0415

    from landsat_lst.config import settings  # noqa: PLC0415

    cat = pystac_client.Client.open(PC)
    for attempt in range(RETRIES):
        try:
            search = cat.search(
                collections=[settings.collection],
                bbox=list(bbox),
                datetime="2021-01-01/2025-12-31",
                query={
                    "eo:cloud_cover": {"lt": settings.max_cloud_cover},
                    "platform": {"in": ["landsat-8", "landsat-9"]},
                },
                limit=1000,
                fields={"include": ["id"], "exclude": ["geometry", "assets"]},
            )
            return sum(1 for _ in search.items_as_dicts())
        except Exception:
            time.sleep(2**attempt)
    return -1  # counted as failed; the summary names these


def main() -> int:
    from landsat_lst.job import generate_jobs  # noqa: PLC0415

    done: dict[str, int] = {}
    if OUT.exists():
        done = {k: v for k, v in json.loads(OUT.read_text()).items() if v > 0}

    tiles = [j.tile for j in generate_jobs() if j.tile.name not in done]
    print(f"{len(done)} cached, {len(tiles)} to count", flush=True)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(count_tile, t.bbox): t.name for t in tiles}
        for n, fut in enumerate(as_completed(futures), 1):
            name = futures[fut]
            done[name] = fut.result()
            if n % 25 == 0 or n == len(futures):
                rate = n / (time.monotonic() - t0)
                print(f"{n}/{len(futures)} ({rate:.1f} tiles/s)", flush=True)
                OUT.write_text(json.dumps(done, indent=0, sort_keys=True))
    OUT.write_text(json.dumps(done, indent=0, sort_keys=True))
    failed = [k for k, v in done.items() if v < 0]
    vals = [v for v in done.values() if v > 0]
    print(
        json.dumps(
            {
                "tiles": len(done),
                "failed": failed,
                "min": min(vals),
                "max": max(vals),
                "mean": round(sum(vals) / len(vals), 0),
                "total_scenes": sum(vals),
            }
        )
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
