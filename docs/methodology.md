# Methodology

Throughout the processing pipeline we favor simple, transparent methods over more
sophisticated approaches that require stronger assumptions or produce values that were never
directly observed. Wherever possible we exclude observations known to be unreliable rather than
replacing them with interpolated or synthesized values. The result is a reproducible dataset
whose strengths and limitations are easy to understand.

Read this alongside the architecture decision records in [`adr/`](adr/README.md), which carry
the supporting measurements.

## Five-year P95 composite

This dataset is a single five-year composite of the 95th percentile (P95) land surface
temperature, covering 2021 through 2025. The goal is to characterize the current spatial
distribution of extreme surface heat rather than temperatures on any particular day or changes
over time. A five-year window gives substantially better spatial coverage than a single year
while remaining representative of present-day conditions: a one-year composite leaves roughly
17% of pixels with no valid November observation at the Pergamino test site, and three years
closes that to near zero.

We use the 95th percentile because it captures persistent extreme conditions without requiring
us to define the hottest season separately for every climate around the world. The result is a
globally consistent indicator of where surfaces tend to become hottest under locally extreme
conditions.

The percentile is computed by pooling every scene in the window into one distribution. Averaging
per-year percentiles would be a percentile of percentiles, which is not the same quantity.

## Quality filtering

Quality control uses the standard Landsat Collection 2 Level-2 quality flags. Almost every scene
is retained and filtering happens at the pixel level. Pixels affected by cloud, cloud shadow,
cirrus, dilated cloud, or snow and ice are excluded before compositing.

Scene-level cloud filtering as a *quality* measure was tested and abandoned. Tightening the
threshold from 100% to 70% dropped 34 of 170 scenes and moved the composite mean by 0.01 C, with
a spatial correlation of 0.98 against the unfiltered result. It cost observations and improved
nothing, so pixel-level QA carries the work.

One filter does still apply. The query asks for `eo:cloud_cover` strictly below the threshold, so
the default of 100 excludes scenes reported at exactly 100% cloud: 154 of 2,912 for a five-year
N40W075 window. Those scenes have no unclouded pixel to contribute by their own metadata.

After conversion to land surface temperature, observations outside a conservative physically
plausible range (-50 to 80 C) are discarded. These primarily represent fill values, reprojection
artifacts, and other processing errors rather than real surface temperatures.

## No gap filling

We do not interpolate or gap-fill missing observations. Every pixel in the composite comes
directly from Landsat observations that passed quality screening. Interpolation would produce
more visually complete maps at the cost of introducing modeled values that cannot easily be
distinguished from measurements. For this first release we favor transparency and
reproducibility over complete spatial coverage.

## Scene normalization

Each Landsat scene is processed independently to convert measured thermal radiation into land
surface temperature using atmospheric information such as water vapor. Small uncertainties in
those atmospheric estimates can shift an entire scene slightly warmer or cooler than neighboring
scenes even when the underlying land surface is unchanged. When many scenes are combined into
one composite, these scene-wide offsets appear as visible seams that follow satellite footprint
boundaries rather than real features on the ground.

To reduce these acquisition artifacts, each scene is shifted by a single scene-wide offset
estimated from its deviation from the monthly climatological median. Because the same constant
applies uniformly to every pixel, the correction changes only the scene's overall baseline. It
does not alter the relative temperature differences within a scene, create or remove local hot
spots, sharpen or blur spatial features, or otherwise modify the spatial structure of the
observations. Its purpose is to place neighboring scenes on a common baseline before they are
combined.

The reference is monthly rather than annual, which is what keeps the seasonal cycle intact. An
annual reference was tested and failed: it absorbed the season itself, cooling the composite
from 40.6 C to 29.8 C.

Like any normalization procedure, this involves a tradeoff. The monthly climatology is estimated
from a limited number of observations, so the estimated offsets may capture some broad-scale
weather variability alongside the scene-wide acquisition bias they target. For that reason,
scenes requiring implausibly large corrections are discarded rather than adjusted. Bounding an
extreme offset instead of dropping the scene would leave most of the error in place while
presenting the scene as corrected.

One consequence deserves stating plainly. Because the composite reports the 95th percentile, it
draws on each location's hottest observations, and those come disproportionately from scenes the
correction identifies as warm-biased. Normalization therefore lowers the reported temperatures,
by about 4 °C at the site where it has been measured. The values describe how hot a surface
becomes relative to its own seasonal normal, rather than the hottest temperature ever recorded
there.

This reflects the intended use of the dataset: identifying persistent spatial patterns of
relative surface heat rather than estimating the exact land surface temperature at any
individual location. Land surface temperature is only one component of the human experience of
heat, and air temperature, humidity, wind, shade, and exposure all shape thermal comfort and
health risk. Users should read this product as a robust indicator of relative surface heat, not
as a source of precise absolute temperature measurements.

Details and the supporting measurements are in
[ADR-007](adr/007-scene-normalization.md).

## Observation counts

The dataset includes monthly quality-assured observation counts for every pixel, published as a
12-band raster. These counts let users assess both the amount and the seasonal distribution of
valid observations supporting each estimate. A location with abundant winter observations but
few summer ones may give less reliable estimates of extreme heat than a location with comparable
coverage year-round.

The counts describe the observations that actually reached the composite. Scenes dropped during
normalization are not counted, so the numbers reflect the evidence behind each value rather than
raw data availability. Publishing them makes data availability explicit rather than leaving
users to infer confidence from the temperature values alone.

## Missing observations

Clouds, cloud shadows, snow, and other observations that fail quality screening are excluded
from the composite.

Some regions also contain gaps from incomplete upstream ASTER coverage rather than any
limitation of this pipeline. Landsat Surface Temperature depends on emissivity from the ASTER
Global Emissivity Dataset, and where ASTER never caught clear sky between 2000 and 2008, USGS
produces no Surface Temperature in any year. These gaps survive every window length, because the
missing input is a static auxiliary dataset rather than a measurement. Across the global urban
domain, 2.66% of land has no emissivity, concentrated in the wet tropics: 12.07% of urban
Southeast Asia and 11.62% of urban Amazonia, against 0.00% for the Sahara and Sahel. They are
most noticeable in parts of southern Africa, including the Durban region. See
[ADR-006](adr/006-no-aster-gap-filling.md) and
[findings-aster-ged-gaps.md](findings-aster-ged-gaps.md).

A pixel with `qa_count == 0` inside the land mask is unambiguously a gap rather than ocean.

## Spatial extent

To reduce storage requirements and focus on the regions where the dataset is most relevant for
planning applications, the dataset covers land between plus and minus 60 degrees latitude. That
range includes all significant population centers, reaching northern Russia, Canada, and
Scandinavia at 60 N and Tierra del Fuego at 55 S, while excluding Antarctica. It reduces the
tile count from 1,728 to 700.

Ocean pixels are excluded using Natural Earth land polygons with a conservative 25 km coastal
buffer that preserves barrier islands, estuaries, and other coastal landforms. See
[findings-land-mask-buffer.md](findings-land-mask-buffer.md).

## Output grid

Every tile is cut from one global grid at 3,600 pixels per degree, roughly 30 m at the equator,
in EPSG:4326. The grid is 1,296,000 by 432,000 pixels, and a 5-degree tile is exactly 18,000 by
18,000. Pixel density is an integer rather than a rounded resolution, so tiles share pixel edges
exactly and adjacent tiles abut with no gap or overlap.

That property matters for the overviews. Coarsening blocks are cut from the global array rather
than from a tile, which keeps block boundaries aligned across what used to be tile edges. A tile
would not support this on its own: 18,000 divides by 4 and by 16 but not by 64, so a per-tile
64x overview would trim its edge and shift block phase against its neighbour. The global grid
divides cleanly by all three. See [ADR-008](adr/008-global-mosaic-topology.md).
