"""Render the AGENTS.md and README.md that sit beside each STAC object.

Portolan requires both files in every catalog and collection directory, linked
from the sibling JSON. README.md addresses a person deciding whether the data
answers their question; AGENTS.md addresses a program deciding how to read it,
and states the contracts -- grid, encoding, band semantics -- that a reader
would otherwise have to infer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from landsat_lst.config import settings
from landsat_lst.encoding import LST_FILL_VALUE, LST_OFFSET, LST_SCALE

if TYPE_CHECKING:
    from landsat_lst.catalog.spec import CatalogSpec

_DECODE = f"celsius = dn * {LST_SCALE} - {abs(LST_OFFSET):g}"


def root_agents_md(spec: CatalogSpec) -> str:
    """Machine-consumer notes for the root catalog."""
    return f"""# AGENTS.md -- {spec.catalog_title}

Root catalog. One child collection per observation window; the published
window is `{spec.collection_id}`.

- Structural links are relative and there are no self links, so the tree can
  be copied or mirrored without rewriting metadata.
- Every catalog and collection declares the Portolan profile URI in
  `stac_extensions`. Items do not: they inherit conformance from their
  collection.
- Read base for the published copy: {spec.read_base_url}
- Per-collection contracts (grid, encoding, band semantics) are in that
  collection's own AGENTS.md. Read it before decoding any pixel.
"""


def root_readme_md(spec: CatalogSpec) -> str:
    """Human-facing overview for the root catalog."""
    return f"""# {spec.catalog_title}

{spec.catalog_description}

## Collections

- [{spec.collection_title}](./{spec.collection_id}/README.md) --
  `{spec.collection_id}`

## License

{spec.license}. The underlying Landsat Collection 2 Level-2 science products
are produced and distributed by the U.S. Geological Survey and carry no
restrictions on use.

## Source

Published to Source Cooperative as `{spec.source_coop_slug}`, readable at
{spec.read_base_url}
"""


def collection_agents_md(spec: CatalogSpec) -> str:
    """Machine-consumer notes for the collection: the contracts a reader needs."""
    return f"""# AGENTS.md -- {spec.collection_title}

One item per five-degree tile. Each item carries two assets, both
cloud-optimized GeoTIFFs with hrefs relative to the item JSON:
`lst_p95` (roles `data`) and `qa_count` (roles `data`, `quality`).

## Grid

- CRS EPSG:4326, one shared global grid of {settings.pixels_per_degree} pixels
  per degree (1/{settings.pixels_per_degree} degrees, about 30 m at the
  equator).
- Tiles are cut from that one grid, never anchored to their own bounding box,
  so neighbouring tiles register pixel for pixel. A five-degree tile is
  18000 x 18000 pixels.
- Item `bbox` and `geometry` are read from the COG header and checked against
  the grid at build time.

## Encoding contract (`lst_p95`)

- Storage: single-band `uint16`.
- Decode: `{_DECODE}`. The same numbers appear on the item as `lst:scale`,
  `lst:offset`, `lst:units`, and `lst:nodata`, and on the band as
  `raster:scale` / `raster:offset`.
- DN {LST_FILL_VALUE} is fill, not a temperature. Mask it before any
  arithmetic; GDAL band scale and offset are embedded, so viewers that honour
  them show Celsius directly.
- Band `statistics` in the item are already decoded to Celsius.

## `qa_count` semantics

- Twelve `uint8` bands, one per calendar month, January first.
- Band M = the number of valid observations in calendar month M, pooled across
  {spec.window}, counted after de-striping. It reports the evidence behind the
  percentile, not raw data availability.
- Value 0 means no valid observation in that month. It is a real value, so the
  asset declares no nodata and 0 must stay visible.

## Overviews

Both assets carry internal overviews built with average resampling. Averaging
`lst_p95` mixes fill (DN {LST_FILL_VALUE}) into coarse pixels along coastlines
and cloud gaps, and averaging `qa_count` spreads counts across nodata, so
overview levels are for display. Read level 0 for analysis.
"""


def collection_readme_md(spec: CatalogSpec) -> str:
    """Human-facing description of the collection, with a reading example."""
    return f"""# {spec.collection_title}

{spec.collection_description}

## License

{spec.license}. The input Landsat Collection 2 Level-2 science products are
produced and distributed by the U.S. Geological Survey and carry no
restrictions on use. Please cite this dataset and the USGS products together.

## Provenance

- Source: Landsat 8 and Landsat 9 Collection 2 Level-2 surface temperature
  (USGS), queried through a STAC API for the {spec.window} window.
- Quality masking: the Collection 2 QA_PIXEL band removes cloud, cloud shadow,
  snow, dilated cloud, and cirrus, and a physical-plausibility clamp drops
  retrievals outside a habitable temperature range.
- De-striping: each scene is shifted by a single scene-wide offset estimated
  against a per-pixel monthly climatology, so the seasonal cycle survives.
  Scenes whose offset cannot be trusted are discarded rather than corrected.
- Compositing: the 95th percentile of every surviving observation in the
  window, pooled -- never a percentile of per-year percentiles.
- Ocean pixels are masked with Natural Earth 10 m land polygons.

## Reading the data

Each item carries `lst_p95` (one `uint16` band) and `qa_count` (twelve `uint8`
bands, January first). Values in `lst_p95` are stored as digital numbers;
decode them with

    {_DECODE}

DN {LST_FILL_VALUE} is fill and must be masked before any arithmetic.

With rioxarray, which applies the embedded scale and offset for you:

```python
import rioxarray

url = (
    "{spec.read_base_url}"
    "{spec.collection_id}/N40W075/lst_p95_{spec.window}_N40W075.tif"
)
lst = rioxarray.open_rasterio(url, masked=True).squeeze()
print(float(lst.max()))  # degrees Celsius
```

With the GDAL command line, to inspect the same file without downloading it:

```bash
gdalinfo -stats \\
  /vsicurl/{spec.read_base_url}{spec.collection_id}/N40W075/lst_p95_{spec.window}_N40W075.tif
```

A `qa_count` value of 0 for a month means no valid observation that month, so
treat a low count as weak evidence for the percentile at that pixel rather
than as missing data.
"""
