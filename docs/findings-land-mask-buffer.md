# Land Mask Coastal Buffer: Fixing Under-Coverage of Coastal Features

**Date:** 2026-06-16
**Status:** Complete
**Analysis:** `scripts/land_mask_diagnostic.py`

---

## Summary

The original land mask using Natural Earth 10m polygons was **over-masking
coastal areas**, excluding barrier islands, marshes, estuaries, and other
inhabited coastal features. This was discovered during Phase 0 tile validation
when the N40W075 (New Jersey) tile showed missing LST data for populated areas
along the Jersey Shore and Chesapeake Bay region.

**Solution:** Add a 25km buffer to the Natural Earth land polygons before
rasterizing. This ensures all coastal features are included while still
excluding open ocean.

---

## Problem

Natural Earth 10m land polygons are generalized and optimized for cartographic
display, not pixel-level masking. They miss:

- **Barrier islands** (Jersey Shore, Outer Banks, etc.)
- **Salt marshes and estuaries** (Chesapeake Bay wetlands)
- **River deltas** (Mississippi, Ganges, etc.)
- **Small coastal features** that fall below the 10m generalization threshold

For a dataset targeting municipal decision-makers analyzing urban heat, these
are exactly the areas that matter — populated coastal zones are often the most
vulnerable to combined heat and flooding risks.

### Visual Evidence

N40W075 (New Jersey) with original mask showed clear gaps along:
- Atlantic City / barrier islands
- Delaware Bay shoreline
- Chesapeake Bay (in adjacent tiles)

These areas had LST data available from Landsat but were being masked out.

---

## Solution

Buffer the Natural Earth land polygons by **25km** before rasterizing. This:

1. **Captures all coastal features** — barrier islands, marshes, deltas
2. **Fills narrow straits and bays** — acceptable for our use case
3. **Still excludes open ocean** — the primary goal of the mask
4. **Adds ~83% more pixels** for coastal tiles (N40W075: 7.8M → 14.4M)

### Why 25km?

- **Too small (5-10km):** Still misses some barrier islands and wide estuaries
- **25km:** Captures all reasonable coastal features without excessive ocean
- **Too large (50km+):** Includes significant open water unnecessarily

### Edge Cases Considered

| Case | Behavior | Acceptable? |
|------|----------|-------------|
| Great Lakes | ~25km into lake from shore | Yes — lakefront cities want this |
| Narrow straits | Fully filled | Yes — inhabited areas |
| Small island nations | Halo around islands | Yes — errs on inclusion |
| Large bays (Chesapeake) | Fully included | Yes — populated coastline |

---

## Implementation

### Code Change

`src/landsat_lst/masks.py`:

```python
COASTAL_BUFFER_METERS = 25_000  # 25km buffer

def load_land_polygons(cache_dir=None, *, buffer_meters=COASTAL_BUFFER_METERS):
    land = gpd.read_file(NATURAL_EARTH_URL)
    land = land.to_crs("EPSG:4326")

    if buffer_meters > 0:
        # Buffer in projected CRS for accurate distance
        land_projected = land.to_crs("EPSG:3857")
        land_projected["geometry"] = land_projected.geometry.buffer(buffer_meters)
        land = land_projected.to_crs("EPSG:4326")

    return land
```

The buffer is applied in EPSG:3857 (Web Mercator) for accurate meter-based
distance calculation, then reprojected back to EPSG:4326.

### Diagnostic Script

Run `scripts/land_mask_diagnostic.py` to visualize mask coverage for any tile:

```bash
uv run python scripts/land_mask_diagnostic.py --tile N40W075 --output results/mask-diagnostic/
```

Outputs:
- `mask_original.tif` — original NE 10m mask
- `mask_buffered.tif` — with 25km buffer
- `mask_diff.tif` — pixels gained (value=1)

---

## Validation

### Coverage Comparison (N40W075)

| Mask | Pixels | Notes |
|------|--------|-------|
| Original NE 10m | 7,848,883 | Missing barrier islands, bays |
| 25km buffer | 14,361,163 | Full coastal coverage |
| Difference | +6,512,280 (+83%) | All coastal features included |

### Global Considerations

The 25km buffer works globally because:

1. **Latitude bounds (±60°)** exclude polar regions where buffer behavior is complex
2. **Ocean exclusion** is the goal, not precise coastline — 25km into ocean is acceptable
3. **Inhabited areas** are the priority — buffer ensures all administered land is included

---

## Alternatives Considered

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Larger NE buffer (50km+) | More coverage | Too much ocean | Rejected |
| ADM1 boundaries | Semantically meaningful | Different data source | Future option |
| Landsat QA water flag | No external data | Over-masks inland water | Previously tried, rejected |
| OSM coastline | Most detailed | Complex to manage | Future option if needed |
| ESA WorldCover | Satellite-derived | Additional dependency | Future option if needed |

The 25km buffer on existing NE data is the simplest solution that solves the
immediate problem without introducing new data dependencies.

---

## References

- Issue #31: Pipeline Performance Validation
- Natural Earth 10m Land: https://www.naturalearthdata.com/downloads/10m-physical-vectors/
- Phase 0 validation tile: N40W075 (New Jersey)
