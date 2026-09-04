"""Configuration and settings for the LST pipeline."""

from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# STAC endpoint presets
STAC_PLANETARY_COMPUTER = "https://planetarycomputer.microsoft.com/api/stac/v1"
STAC_EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"


class Settings(BaseSettings):
    """Pipeline configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="LST_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    stac_url: str = Field(
        default=STAC_EARTH_SEARCH,
        description="STAC API endpoint. Earth Search is 4.6x faster on Coiled (same-region S3).",
    )
    collection: str = Field(
        default="landsat-c2-l2",
        description="STAC collection ID",
    )
    # pystac-client mounts its own adapter, but from a plain int, which leaves
    # status retries off entirely. These three drive an explicit policy that
    # covers 429 and 5xx on every verb. See pipeline._stac_retry.
    stac_retries: int = Field(
        default=5,
        ge=0,
        description="Retry attempts for a STAC request that returns 429 or 5xx.",
    )
    stac_retry_backoff_s: float = Field(
        default=1.0,
        ge=0.0,
        description="urllib3 backoff factor. At 1.0 five attempts span about 30s, "
        "so a real outage fails the tile long before coiled_job_timeout.",
    )
    stac_timeout_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Per-request timeout for STAC calls, in seconds.",
    )

    # Storage backend selection
    storage_backend: Literal["local", "s3"] = Field(
        default="local",
        description="Storage backend: 'local' for testing, 's3' for production",
    )

    # Local storage (testing)
    output_dir: Path = Field(
        default=Path("output"),
        description="Local output directory for COGs (used when storage_backend='local')",
    )

    # S3 storage (production). The defaults are the Source Coop publication
    # target that the runbook, the publish tests, and the June e2e run all
    # use; the old "source-coop-radiant-earth" bucket never existed.
    s3_bucket: str = Field(
        default="us-west-2.opendata.source.coop",
        description="S3 bucket for COG storage",
    )
    s3_prefix: str = Field(
        default="nlebovits/landsat-lst",
        description="S3 key prefix for COGs",
    )
    s3_region: str = Field(
        default="us-west-2",
        description="AWS region for S3 bucket",
    )

    tile_size_degrees: float = Field(
        default=5.0,
        description="Tile size in degrees (latitude and longitude)",
    )
    min_latitude: float = Field(
        default=-60.0,
        description="Minimum latitude bound",
    )
    max_latitude: float = Field(
        default=60.0,
        description="Maximum latitude bound",
    )

    max_cloud_cover: int = Field(
        default=100,
        description="Scene-level cloud filter, applied as eo:cloud_cover < this. "
        "The default is not the no-op it looks like: strict less-than still drops "
        "every scene reported at exactly 100% cloud, 154 of 2,912 for N40W075 over "
        "2021-2025. Use 101 for a true no-op. Below 100 the filter starts removing "
        "scenes that carry data, and eo:cloud_cover describes a whole ~185km "
        "footprint rather than the part of it a tile sees, so it is a weak proxy for "
        "what a tile actually gets. Score a candidate threshold with "
        "scripts/analyze_cloud_cover_filter.py before lowering it; see "
        "docs/findings-cloud-cover-filter.md.",
    )

    pixels_per_degree: int = Field(
        default=3600,
        description="Pixel density of the global grid, in pixels per degree (~30m at the "
        "equator). An integer rather than a resolution float so the global grid, and "
        "every tile cut from it, come out with a whole number of pixels. The earlier "
        "0.00027778 truncated 1/3600 and left each tile anchored to its own bbox, "
        "overshooting its eastern edge by ~0.5px and misregistering against its "
        "neighbour by ~0.14px. See ADR-008.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolution(self) -> float:
        """Output resolution in degrees, derived from :attr:`pixels_per_degree`."""
        return 1.0 / self.pixels_per_degree

    crs: str = Field(
        default="EPSG:4326",
        description="Output coordinate reference system",
    )

    nodata: float = Field(
        default=-9999.0,
        description="NoData value for output rasters",
    )

    lst_valid_min: float = Field(
        default=-50.0,
        description="Minimum physically-plausible land surface temperature (degC). "
        "Values below are treated as artifacts (e.g. ~-124degC DN=0/resampling junk) "
        "and dropped. Conservative: only removes clearly non-physical values.",
    )
    lst_valid_max: float = Field(
        default=80.0,
        description="Maximum physically-plausible land surface temperature (degC). "
        "Above the hottest observed land skin temps (~70-80degC deserts); drops "
        "high-DN saturation/fill artifacts without clipping real extreme heat.",
    )

    # ASTER GED emissivity-gap mask. See docs/findings-aster-ged-gaps.md and
    # the tracked per-pixel cross-tab in results/decision/ged_gap_s30w065.json,
    # regenerated by `landsat-lst ged-analyze`. (The older citation,
    # results/ged-mask-check/, was an agent scratch directory and is gone.)
    ged_gap_mask: bool = Field(
        default=True,
        description="Drop composite pixels whose ASTER GED v3 (AG100) cell has "
        "NumObs == 0, plus a buffer (ged_gap_buffer_cells). USGS interpolates "
        "GED emissivity over these cells, and the >= 70 degC artifact tail is "
        "strongly associated with them: on S30W065 they carry 79.9% of that "
        "tail at 369x the tile base rate, and 99.6% of the tile's missing "
        "pixels. That is a measured spatial association, not an "
        "observation-level trace of what produced any pixel. The rule removes "
        "0.8642% of valid pixels and 92.45% of the >= 70 degC tail (maximum "
        "77.87 degC). Applies to the LST output only, exactly like the land "
        "mask -- never to offset estimation, and never to qa_count (zero "
        "observations is data; the count layer stays the evidence).",
    )
    warp_exact_transform: bool = Field(
        default=True,
        description="Warp every source read with GDAL's exact transformer. This "
        "is the v1 scientific contract (2026-09-03): an output pixel must not "
        "depend on the processing window it was read through. rasterio's "
        "default is an approximate transformer at 0.125 px, linearised per "
        "destination window, under which a 512 x 1024 window moved 3,642 of 84M "
        "source pixels against 512 x 512 on 40 real S30W065 scenes and the P95 "
        "by up to 1,681 DN. Exact makes every window bit-identical; against the "
        "approximate product it moves 4.4% of P95 pixels, 97% of them by under "
        "1 C, and is why offsets.ALGORITHM_VERSION is 2. Off exists only to "
        "reproduce the pre-v1 product. Cost 5 to 32 ms of warp CPU per 512 x 512 "
        "read. Applied through pipeline._install_warp_tolerance to every "
        "load_scenes call in the process.",
    )
    ged_gap_buffer_cells: int = Field(
        default=1,
        ge=0,
        description="Dilation radius of the GED gap mask, in GED cells (~1 km "
        "each, 8-connected). 1 is the verified rule: on S30W065 the unbuffered "
        "gap cores catch the ST-fill blobs but leave the hot fringe, and one "
        "cell of buffer takes the >= 70 degC tail from 2,793 pixels to 211 "
        "(92.45% removed) for 0.8642% of valid pixels. The residue sits on "
        "NumObs 1-3 cells and is deliberately not chased: chasing it "
        "(NumObs <= 2 + buffer) costs 11.4% of valid pixels.",
    )
    ged_dir: Path = Field(
        default=Path("data/aster_ged"),
        description="Directory of local AG100 v3 granules "
        "(AG1km.v003.{lat}.{lon}.0010.h5). The fallback mask source when no "
        "artifact exists; a granule the mask needs but the directory lacks is "
        "an error naming the granule ids, never a silent skip.",
    )
    ged_artifact: Path = Field(
        default=Path("data/ged_gap_mask.npz"),
        description="Compact global gap-cell artifact written by "
        "scripts/build_ged_gap_mask.py. Preferred over ged_dir when present: "
        "a fleet VM ships this one file, not 8,776 granules. Verified to "
        "produce a mask identical to the granule path on S30W065.",
    )

    # Season-aware per-scene normalization (de-striping). See issue #46, ADR-007.
    destripe: bool = Field(
        default=True,
        description="Normalize each scene against a per-pixel monthly climatology "
        "before compositing, removing the WRS-footprint seams caused by per-scene "
        "atmospheric-correction bias. Disable to benchmark against raw composites.",
    )
    destripe_max_offset_c: float = Field(
        default=15.0,
        description="Discard a scene whose absolute offset exceeds this (degC) rather "
        "than adjusting it. Calibrated at Pergamino 2021-2025 (ADR-007): the offset "
        "distribution is a tight core (82.7% of scenes within +/-15, std 5.71) plus a "
        "one-sided cold tail from undetected cloud, so 15 sits near 2.6 core sigma and "
        "discards 21.8% of scenes, 63 cold against 1 warm. Re-run "
        "scripts/calibrate_destripe_cap.py on a humid tropical tile before the global "
        "build; the AOI behind this number is mid-latitude cropland.",
    )
    destripe_min_scene_pixels: int = Field(
        default=500,
        description="Coverage floor, in native-resolution pixels: discard a scene "
        "covering less valid land than this. Applies only when offsets are estimated "
        "at native resolution; a coarse grid uses destripe_min_offset_samples on its "
        "own pixels instead. The two floors replace each other rather than converting "
        "into each other, because a coarse count cannot be scaled back to a native "
        "one: GDAL's average ignores nodata, so one valid fine pixel still yields a "
        "valid coarse pixel (1 native pixel read as 13 at factor 8). See "
        "normalization.rejection_floor and docs/findings-offset-subsampling.md.",
    )
    destripe_min_offset_samples: int = Field(
        default=200,
        description="Reliability floor, in offset-grid pixels: discard a scene whose "
        "offset rests on fewer samples than this. Distinct from the coverage floor "
        "because a coarse grid can leave a well-covered scene with too few samples "
        "to place a median on.",
    )
    destripe_offset_resolution_factor: int = Field(
        default=2,
        description="Estimate per-scene offsets from a stack loaded at "
        "resolution * factor, served from the source COGs' internal overviews "
        "([2,4,8,16,32,64]). 1 keeps offsets at native resolution. Validated at "
        "Pergamino, 2026-08-14: factor 2 reproduces native offsets to a median of "
        "0.0017 degC (p99 0.072, max 0.219) with zero keep/reject flips, and is now "
        "the largest that passes. Factor 4 was tried for issue #81 and rejected: its "
        "max |delta| is 0.546 degC against a pre-registered bound of 0.5. It would "
        "have cut the offset pass from 613,240 tasks to 155,239. Offset error grows "
        "linearly in the factor, so do not raise this without re-running "
        "scripts/validate_offset_subsampling.py -- and re-run it rather than citing "
        "an older table, since the shipped grid and the offset code have both moved "
        "since the first sweep. The saving caps out near 2x regardless, because the "
        "P95 still needs a native pass.",
    )

    destripe_bounded_units: bool = Field(
        default=True,
        description="Estimate offsets as bounded work units rather than as one dask "
        "graph. The graph form holds two medians reducing along orthogonal axes -- a "
        "per-pixel median over time and a per-scene median over space -- so no "
        "chunking satisfies both and the scheduler materializes the stack. Measured "
        "2026-08-15: construction alone exceeds 26 GB above 2,000 scenes and the "
        "execution plateau is scene-independent at ~21 GB. The unit form computes the "
        "climatology one spatial block at a time and each scene's offset "
        "independently, which is bit-exact against the graph (E1: max |delta| = 0 on "
        "three variants, identical NaN patterns) while holding memory to one unit. "
        "Set False to run the graph form, which is retained as the equivalence oracle.",
    )
    destripe_unit_memory_gb: float = Field(
        default=13.0,
        ge=0.25,
        description="Resident budget for one phase-A spatial block. The block edge is "
        "chosen as the largest power of two whose stack (edge^2 * scenes * 4 bytes) "
        "fits, so unit memory stays flat as the window grows rather than scaling with "
        "it. Raising this enlarges the I/O unit and cuts the number of COG opens; it "
        "does not change any value. 13.0 admits a 1024 block edge at the 2,930-scene "
        "window (1024^2 * 2930 * 4 B = 12.3 GB), which load_chunk_size_offsets=1024 "
        "requires (_io_block_edge never goes below the spatial chunk edge). The "
        "aggregate in-flight bound lives in destripe_total_memory_gb, not here.",
    )
    destripe_total_memory_gb: float = Field(
        default=32.0,
        ge=1.0,
        description="Total in-flight budget across concurrent work units: the unit "
        "worker count is clamped to total // unit_bytes. At production offset "
        "geometry (12.3 GB units) that is 2 concurrent units; phase B's 3.2 GB "
        "batches get 8. Sized to leave the 3.9 GB climatology, the process "
        "baseline, and headroom on a 64 GiB VM.",
    )
    destripe_compute_panel: int = Field(
        default=256,
        ge=32,
        description="Edge of the panel the climatology kernel actually reduces over, "
        "within each I/O block. Decoupled from the block because the two want "
        "opposite things: I/O wants few large reads, the kernel wants a working set "
        "that stays in cache. Measured 2026-08-15 (E3, 300 scenes, 2250^2): 256 runs "
        "in 65.8 s against 116-122 s at 512, 1024, and 2048, which are flat within "
        "5% of each other. A 256^2 panel of one month is ~6.5 MB and L3-resident. "
        "The climatology checksum is identical at every panel size.",
    )
    destripe_scene_batch: int = Field(
        default=8,
        ge=1,
        description="Scenes per phase-B read. Each batch is one small dask graph over "
        "the full tile footprint, so the batch trades resident memory (batch * "
        "pixels * 4 bytes) against the number of graphs. Phase B costs 12.4 ms of CPU "
        "per scene (E2, constant across 50-300 scenes), so this setting is about read "
        "concurrency rather than compute.",
    )

    # COG output. Literal rather than str so an unsupported codec fails at
    # settings load instead of deep inside the GeoTIFF write.
    cog_compression: Literal["deflate", "zstd", "lzw"] = Field(
        default="deflate",
        description="GeoTIFF compression for exported COGs. Deflate is the widest-"
        "supported option; zstd is smaller and faster but needs a newer GDAL.",
    )
    cog_blocksize: int = Field(
        default=512,
        description="Internal tile size (px) for exported COGs. 512 keeps range "
        "requests coarse enough to amortize HTTP overhead without over-fetching.",
    )

    load_chunk_size: int = Field(
        default=512,
        description="Spatial (lat/lon) chunk size for odc-stac scene loading on "
        "the native/composite path. Chunk size sets bytes per S3 range request, "
        "and the 2026-08-21 concurrency-ladder probe (results/probe/, scripts/"
        "probe_io_ladder.py) found request size is THE throughput lever: chunk "
        "256 was flat at 12-24 MB/s across every thread count while 512 read "
        "70-85 MB/s warm. The old 256 default rested on the fused offset "
        "graph's threads*chunk**2*scenes memory term, which ADR-015's bounded "
        "units removed; the composite's own ceiling is the single-time-chunk "
        "rechunk, whose per-task working set is chunk**2 * scenes * 4 B -- "
        "3.1 GB at 512 over 2,930 scenes, and an infeasible 12.3 GB at 1024, "
        "which is why the composite stops at 512 while the offset pass "
        "(load_chunk_size_offsets) goes to 1024.",
    )

    load_chunk_size_offsets: int = Field(
        default=1024,
        description="Spatial chunk size for the coarse offset-pass load "
        "(resolution_factor > 1). Probe-measured 2026-08-21: (8 threads, "
        "chunk 1024) read 158.4 MB/s at 3.0 cores against 70-85 MB/s at 512 "
        "and 12-24 MB/s at the old 256 -- request size, not concurrency, is "
        "the lever, and the offset pass has no rechunk term to cap it. A 1024 "
        "chunk obliges _io_block_edge to a >= 1024 block edge (~12.3 GB per "
        "unit at 2,930 scenes), which destripe_unit_memory_gb accommodates "
        "and destripe_total_memory_gb bounds in aggregate.",
    )

    dask_max_threads: int | None = Field(
        default=None,
        description="Cap on dask's threaded scheduler for one tile; None uses "
        "dask's default (the VM's CPU count). Was 1 for the fused offset graph "
        "(one time series per thread, ~49 GB at 16 threads), then 4 from a "
        "2026-08-16 sweep (1776 / 1377 / 1235 / 1445 s at 1 / 2 / 4 / 8 "
        "threads, RSS flat within 3.6%). That sweep measured the wrong thing: "
        "its hot kernels were np.nanquantile's per-pixel apply_along_axis loop "
        "and np.nanmedian's masked-array small path, both of which hold the "
        "GIL, so past 4 threads contended rather than overlapped. Both kernels "
        "were replaced 2026-08-21 with GIL-releasing sort kernels "
        "(landsat_lst.kernels), which removes the curve's cause. Never "
        "re-calibrate this on a GIL-bound kernel. Unit-read concurrency is a "
        "separate knob (destripe_io_threads); this one governs composite and "
        "export graphs.",
    )
    destripe_unit_workers: int = Field(
        default=0,
        ge=0,
        description="Concurrent bounded work units (phase-A blocks, phase-B "
        "scene batches). 0 means auto: min(8, CPU count). Each worker holds at "
        "most one unit resident, so this is also the memory bound -- in-flight "
        "bytes stay within destripe_total_memory_gb, and the count shrinks "
        "when units are large (a 12.3 GB offset block at chunk 1024 admits 2). "
        "The serial form this replaces ran 324 independent blocks one at a "
        "time on ~1 core of 8; the loop, not the estimator, was the offset "
        "pass's wall clock (2026-08-21 investigation).",
    )
    destripe_io_threads: int = Field(
        default=8,
        ge=1,
        description="Threads in the shared pool that executes unit reads "
        "(dask threaded scheduler, pool= override). The 2026-08-21 "
        "concurrency-ladder probe overturned the raise-concurrency hypothesis: "
        "throughput fell monotonically with MORE threads at every chunk size "
        "(8 > 16 > 32 > 64 > 128), and the winning arm was 8 threads at chunk "
        "1024 (158.4 MB/s, 3.0 cores, zero S3 throttling). Request size, not "
        "request count, amortizes latency on this path. Do not raise this "
        "without re-running scripts/probe_io_ladder.py.",
    )
    dask_workers: int = Field(
        default=8,
        description="Number of Dask workers for local processing",
    )
    dask_threads_per_worker: int = Field(
        default=2,
        description="Threads per Dask worker",
    )
    dask_memory_limit: str = Field(
        default="6GB",
        description="Memory limit per Dask worker",
    )

    # Coiled distributed execution. Every knob is pinned here rather than left
    # to Coiled defaults: an unpinned region reads Landsat cross-region, and
    # the default adaptive scaling (n_workers=[0, 500]) has no cost ceiling.
    coiled_retries: int = Field(
        default=3,
        description="Number of retries per batch task. Covers spot preemption "
        "and transient object-store failures; a tile that fails deterministically "
        "burns all three and then reports.",
    )
    coiled_region: str = Field(
        default="us-west-2",
        description="Cloud region for Coiled workers. Must match the Landsat "
        "source bucket and the output bucket, or every read pays egress.",
    )
    coiled_vm_types: list[str] = Field(
        default=["r6i.2xlarge", "m6i.4xlarge"],
        description="Candidate VM types for batch tasks, in preference order. "
        "Both carry 64 GiB. The pipeline is I/O bound, so buy memory, not "
        "cores; 32 GiB (r6i.xlarge) OOMed a heavy tile at 28.77 GiB in run "
        "2021-2025-20260812T142408Z.",
    )
    coiled_spot_policy: Literal["on-demand", "spot", "spot_with_fallback"] = Field(
        default="spot_with_fallback",
        description="Instance purchase strategy. Spot preemption is safe: "
        "tile writes are idempotent via the two-asset existence check.",
    )
    shard_spot_policy: Literal["on-demand", "spot", "spot_with_fallback"] = Field(
        default="spot",
        description="Purchase strategy for shard-stage clusters, deliberately "
        "stricter than coiled_spot_policy: no per-VM on-demand fallback. The "
        "700-tile build's budget holds only at spot prices (measured-count "
        "projection 2026-08-22: $1.9-4.7k spot against $6.2k on-demand, gates "
        "<$3k target / $5k ceiling), and spot_with_fallback would let a "
        "capacity shortfall silently convert the run to the on-demand figure. "
        "Under 'spot' a shortfall surfaces as missing shards instead, which "
        "the driver's bounded resubmission rounds retry and then FAIL loudly. "
        "Preemption stays safe: shard outputs are idempotent by key. Raise to "
        "spot_with_fallback only as an explicit per-run decision with the "
        "exposure priced first.",
    )
    coiled_max_workers: int = Field(
        default=4,
        description="Ceiling on VMs running batch tasks at once. Coiled gives "
        "each task its own VM and queues the rest, so a 700-tile job never runs "
        "700 machines. A CONCURRENCY ceiling, not a cost cap: spend is "
        "concurrency integrated over time and nothing here bounds the time.",
    )
    coiled_job_timeout: str = Field(
        default="24 hours",
        description="Wall-clock budget per batch task. A tile that runs past "
        "this is stuck, not slow; the timeout stops it from billing all night. "
        "Was 6 hours, which no longer bounds a real tile and would have killed "
        "every one of them mid-climatology. A 2,930-scene offset pass reads "
        "949.3 GB twice and projects to 15.75 h from the rates measured at "
        "production geometry (26.7 MB/s phase A, 44.9 MB/s phase B, 2026-08-16); "
        "the native composite is a further 3,797 GB and is not yet measured. "
        "24 leaves ~50% headroom over the destriping projection while still "
        "catching a tile that has genuinely hung. Revisit once one end-to-end "
        "tile has completed and the composite has been characterized.",
    )
    aws_profile: str = Field(
        default="radiant-earth",
        description="AWS profile whose (SSO) credentials are frozen and "
        "forwarded to Coiled workers for S3 writes.",
    )

    # Sharded tile execution (ADR-016). One tile across many VMs, sequenced by a
    # local driver that polls S3 between stages. Coiled Batch has no dependency
    # mechanism, so every knob here is either a fleet width or a barrier bound.
    shard_climatology_vms: int = Field(
        default=0,
        ge=0,
        description="Processes the phase-A climatology is split across. 0 means "
        "auto, from landsat_lst.projection.tile_projection: the coarse pass "
        "divided by the offset phase's minute budget. Clamped to the number of "
        "blocks the tile actually has, since a shard with no block is a VM that "
        "boots to do nothing.",
    )
    shard_offset_vms: int = Field(
        default=0,
        ge=0,
        description="Processes the phase-B per-scene offsets are split across. "
        "0 means auto, as shard_climatology_vms. Clamped to the number of scene "
        "batches, which is what bounds it on a short window.",
    )
    shard_composite_vms: int = Field(
        default=0,
        ge=0,
        description="Row bands the native composite is split across. 0 means "
        "auto, from the composite phase's minute budget in "
        "landsat_lst.projection. Clamped to the number of whole COG block rows "
        "in the tile: a band must start on a block row for the merge to stay a "
        "windowed copy (landsat_lst.shards.band_edges).",
    )
    shard_composite_vm_type: str = Field(
        default="m6i.4xlarge",
        description="VM type for composite shards. The composite is the "
        "native-resolution read and wants cores against a 16-thread load; the "
        "offset stages keep the default preference list. Named as one type "
        "rather than a list so the chunk this stage runs at "
        "(shard_composite_chunk) describes a known core count.",
    )
    shard_composite_per_column: bool = Field(
        default=False,
        description="Experimental compatibility switch for loading a composite "
        "band through one stac_load call per longitude chunk. Production keeps "
        "this off: shard export bounds execution over one whole-band graph instead.",
    )
    shard_composite_chunk: int = Field(
        default=1024,
        ge=64,
        description="Spatial chunk edge a composite shard loads at, overriding "
        "load_chunk_size. 1024 is safe on two conditions that both hold now. "
        "Memory: the 2026-08-22 rejection (16 rechunks at 4.32 GB, 69 GB) was "
        "the all-columns-resident ordering; composite shards now compute and "
        "write two longitude chunks at a time from one whole-band graph. Pixels: "
        "the read window is also the warp window, and under rasterio's default "
        "approximate transformer a 512 x 1024 window moved 3,642 of 84M source "
        "pixels and the P95 by up to 1,681 DN; under warp_exact_transform (the "
        "v1 contract) 512 and 1024 are bit-identical in both products through "
        "the shard path (docs/evidence/issue-139/exact-baseline-local). The "
        "gain is the #139 read-rate lever: reads cost per request, not per byte, "
        "and a 512-row band reads 512 x 1024 windows, half the reads per item. "
        "Applied by every shard process AND by the planner, so the plan digest "
        "-- which covers load_chunk_size -- agrees across all of them.",
    )
    destripe_stage_coarse: bool = Field(
        default=True,
        description="Stage phase A's coarse observations as uint16 DN so phase B "
        "reuses them instead of reading the Landsat sources a second time "
        "(issue #125). Values are unchanged: phase B reconstructs the estimator's "
        "float32 input bit-identically. Off falls back to two source passes.",
    )
    shard_export_disk_gb: int = Field(
        default=100,
        ge=1,
        description="Scratch disk for the export-merge VM. It downloads every "
        "row band of both products, merges each into a full-tile intermediate, "
        "and runs cog_translate over it, so three full-tile rasters are on "
        "disk at once. The default VM disk does not hold them.",
    )
    shard_driver_poll_s: float = Field(
        default=20.0,
        gt=0,
        description="Seconds between the driver's storage listings while a "
        "stage barrier is open. One listing of the tile's shard prefix per "
        "poll, which is how the driver learns a shard finished: Coiled Batch "
        "has no dependency mechanism and an exit code is not completion.",
    )
    shard_barrier_timeout_s: int | None = Field(
        default=None,
        ge=0,
        description="Explicit override for every stage barrier deadline, in "
        "seconds. None (the default) derives each stage's deadline from "
        "landsat_lst.budgets: bytes over measured rates, per shard, times "
        "shard_budget_safety. It was a hand-entered 7200 for every stage, "
        "which is a guess that ages badly and that nobody recomputes when the "
        "window, the fleet width, or a measured rate moves. Set it only to "
        "reach for a stopwatch during an incident.",
    )
    coiled_credit_period_days: int = Field(
        default=30,
        ge=1,
        description="How far back the billing-activity fallback sums debits. "
        "An approximation: nothing observable says when the quota period "
        "resets. Too short under-counts spend and lets an unaffordable run "
        "start; too long over-counts and refuses an affordable one. 30 days is "
        "the conservative reading of a monthly quota.",
    )
    coiled_credit_quota: float | None = Field(
        default=None,
        gt=0,
        exclude=True,
        description="Deprecated compatibility setting. It is accepted so an existing "
        ".env does not make Settings fail to load, but quota preflight never trusts or "
        "uses this stored value; the current limit must be confirmed by an operator.",
    )
    coiled_billing_max_pages: int = Field(
        default=20,
        ge=1,
        description="Pages of billing activity the fallback will read. The "
        "observed history is ~1,294 events; this bounds a preflight so it "
        "cannot become slower than the run it precedes.",
    )
    coiled_credit_safety: float = Field(
        default=2.0,
        ge=1.0,
        description="Headroom a run must have beyond its credit estimate. The "
        "estimate is priced per vCPU-hour, calibrated on the S30W065 run that "
        "billed 268.11 credits; the per-cluster rates spread 0.6-1.25 because "
        "a fleet's VMs do not boot or finish together. 2.0 carries that band's "
        "width. Being killed mid-stage costs the whole tile, while refusing "
        "costs a re-check.",
    )
    ack_quota: bool = Field(
        default=False,
        description="Proceed when the credit balance cannot be read, on the "
        "strength of an operator's manual check. The driver refuses otherwise "
        "rather than guessing, because an exhausted quota kills a healthy "
        "fleet mid-stage.",
    )
    shard_budget_safety: float = Field(
        default=2.0,
        gt=0,
        description="The only slack in a barrier deadline: a stage may take "
        "this multiple of its projected work before the driver acts. One "
        "number rather than one per stage, so widening it is a conversation "
        "rather than a silent edit. 2.0 covers a spot replacement booting and "
        "redoing the slowest shard.",
    )
    shard_submit_retries: int = Field(
        default=3,
        ge=1,
        description="Attempts at one submission API call before it is treated "
        "as terminal. A transient failure must not kill the driver: on "
        "2026-08-22 an empty ServerError from a cluster create -- the Coiled "
        "credit quota, as it turned out -- ended the run outright rather than "
        "being retried or reported.",
    )
    shard_submit_backoff_s: float = Field(
        default=5.0,
        ge=0,
        description="First wait between submission retries; doubled each time. "
        "Short, because a submission is a control-plane call rather than the "
        "work, and the fleet is idle while it is retried.",
    )
    # Consolidation (ADR-016, "One fleet per side"). Boots and queueing, not
    # compute: offsets-side shards computed ~6 minutes each while their stages
    # held fleets ~30. These knobs bound the waits that replace those boots.
    shard_unit_poll_s: float = Field(
        default=10.0,
        gt=0,
        description="Seconds between S3 listings *inside* a task, while it "
        "waits at an in-process barrier: for the plan shard 0 writes, for the "
        "phase-A blocks its peers write, or for the merged offset record. "
        "Faster than the driver's poll because these waits are minutes rather "
        "than hours and the listings are small.",
    )
    shard_plan_wait_s: int = Field(
        default=1800,
        ge=0,
        description="How long a fused offsets task waits for shard 0 to "
        "publish plan.json before failing. Shard 0 runs one STAC query and "
        "builds two lazy graphs, which is minutes; the ceiling is generous "
        "because the alternative to waiting is a whole fleet's boot.",
    )
    shard_block_wait_s: int = Field(
        default=5400,
        ge=0,
        description="How long a fused offsets task waits at the in-process "
        "phase-A barrier for every peer's climatology blocks. Phase B cannot "
        "start without the whole climatology, and waiting in a booted process "
        "costs nothing but time already paid for.",
    )
    shard_offsets_record_wait_s: int = Field(
        default=5400,
        ge=0,
        description="How long a composite shard waits for the merged offset "
        "record before failing. Composite VMs are started while phase B is "
        "still running (shard_composite_overlap), so an early boot must wait "
        "rather than refuse -- a refusal here would burn the boot the overlap "
        "exists to save.",
    )
    shard_composite_overlap: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of phase-B partials that must exist before the "
        "driver starts the composite fleet, overlapping its boot with the tail "
        "of the offsets stage. 0.0 means the first partial: proof the offsets "
        "stage is producing, which is what distinguishes overlap from "
        "gambling. 1.0 disables the overlap (composite starts only once phase "
        "B is complete).",
    )
    shard_export_claim_fallback_s: int = Field(
        default=900,
        ge=0,
        description="How long the driver waits for the COGs after every band "
        "slab exists before submitting the export stage itself. A composite "
        "shard claims the export once it writes the last band, which saves a "
        "whole VM boot; this is the belt for a claim that is never executed "
        "because the claiming VM was preempted.",
    )
    shard_barrier_rounds: int = Field(
        default=2,
        ge=1,
        description="Submissions per stage, including the first. The second is "
        "the resubmission of only the missing indexes; after that the tile "
        "fails, naming the keys that never appeared. Unbounded retries would "
        "bill all night against a shard that is failing deterministically.",
    )

    # Fleet consolidation: many tiles through one work array per stage per
    # wave (ADR-018). A wave with more units than workers has Coiled queue the
    # surplus onto VMs that already booted, which is the whole saving.
    fleet_max_vms: int = Field(
        default=64,
        ge=1,
        description="Ceiling on CONCURRENT VMs across the whole consolidated "
        "run: the driver never submits a wave wider than the headroom left, so "
        "two stages cannot race each other past it. It is NOT a spending cap. "
        "Spend is the integral of concurrency over time, and nothing here "
        "bounds the time: a run at half this cap for twice as long costs the "
        "same, and a fleet that replaces reclaimed spot VMs churns instance "
        "launches at flat concurrency. The census measures what is running and "
        "so enforces this cap; it yields only a LOWER bound on the bill, "
        "because the substrate reports when a worker started and not when it "
        "stopped. See ADR-018.",
    )
    fleet_ghost_ttl_s: float = Field(
        default=300.0,
        ge=0.0,
        description="How long width released without an authoritative worker "
        "census keeps counting against the concurrency cap. Reached only in "
        "degraded mode -- no credentials, control plane down, or a backend that "
        "cannot be asked -- where the driver cannot tell a wave with a hung "
        "unit from a wave that was preempted. Holding forever is safe and "
        "deadlocks the run; releasing at once is live and doubles the bill. "
        "Charging the released width for a bounded interval is both, at a "
        "ceiling of twice the cap. One VM boot is the default because that is "
        "the interval a fleet can plausibly still be tearing down in.",
    )
    fleet_wave_window_s: float = Field(
        default=120.0,
        ge=0.0,
        description="How long buffered units wait for more tiles to join "
        "before a wave is submitted anyway. The batching window is what keeps "
        "the submission count independent of the tile count: without it, a "
        "stage whose tiles become ready one at a time would submit one array "
        "per tile and buy nothing. Paid at most once per wave, and skipped "
        "entirely when no other tile could still join.",
    )
    fleet_poll_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Seconds between fleet driver polls. One listing per "
        "shared prefix per poll serves every tile, so this is a request-rate "
        "knob rather than a latency one; the per-tile barriers are minutes "
        "wide. It also bounds the resolution of the per-wave completion "
        "timestamps, which are observations of the bucket rather than reports "
        "from a worker.",
    )

    # Live observability for batch tiles. A batch task is a plain process that
    # never registers with dask, so the cluster dashboard reports nothing about
    # it and its stdout stays on the VM until it exits. These knobs drive the
    # heartbeat objects and uploaded logs that replace both. See issue #68.
    heartbeat_interval_s: int = Field(
        default=60,
        description="Seconds between heartbeat writes from a running tile. "
        "One small PUT each, ~84k across a 700-tile run (about $0.42).",
    )
    watch_stale_after_s: int = Field(
        default=120,
        description="A tile whose last heartbeat is older than this is stale: "
        "killed, wedged, or preempted. Two heartbeat intervals, so a single "
        "missed write does not raise a false alarm.",
    )
    watch_poll_interval_s: int = Field(
        default=30,
        description="Seconds between storage polls in `landsat-lst watch`. "
        "Unchanged heartbeat objects are served from cache, so a poll costs "
        "one listing plus a read per tile that actually beat.",
    )
    task_log_max_bytes: int = Field(
        default=1_048_576,
        description="Ceiling on the uploaded task log. A longer log is "
        "uploaded as its tail, where the traceback is, under a truncation "
        "notice; the full log stays on the VM until it is torn down.",
    )
    manifest_dir: Path = Field(
        default=Path("results/runs"),
        description="Directory for per-run JSON manifests of distributed runs.",
    )

    # Per-key dask profiling. A heartbeat says a phase has run for an hour and
    # GraphProgress says how many tasks are left; neither says which tasks. See
    # issue #76 and landsat_lst.profiling.
    profile_dask: bool = Field(
        default=False,
        description="Wrap the de-striping compute in dask.diagnostics and dump "
        "a per-task-prefix summary beside the tile's heartbeat. Answers which "
        "operation owns the wall clock, and records the RSS curve we otherwise "
        "reconstruct by hand from heartbeat samples. Off by default: it is "
        "worth turning on for a sampled run, not for a 700-tile build.",
    )
    profile_dask_cache: bool = Field(
        default=False,
        description="Also run CacheProfiler, which reports the bytes dask held "
        "in memory and so answers why RSS is climbing. Gated separately from "
        "profile_dask because it retains one record per task: the de-striping "
        "graph reached 598,604 tasks on a 300-scene N40W075 sample, on a run "
        "already near its memory ceiling.",
    )
    profile_dask_interval_s: float = Field(
        default=1.0,
        description="Sampling interval for ResourceProfiler's RSS and CPU "
        "curve. One second over a two-hour phase is 7,200 samples, which the "
        "dump strides down before writing.",
    )
    exec_trace: bool = Field(
        default=False,
        description="Record a per-second composite-shard execution trace. Off by "
        "default and intended only for a bounded measurement run.",
    )
    exec_trace_interval_s: float = Field(
        default=1.0,
        gt=0.0,
        description="Seconds between host samples in an execution trace.",
    )
    exec_trace_read_sample: int = Field(
        default=20,
        ge=1,
        description="Record timing and source metadata for every Nth rio_read call.",
    )

    @model_validator(mode="after")
    def _grid_must_be_integral(self) -> "Settings":
        """Reject any grid where the globe or a tile lands on a fractional pixel.

        Every tile is cut from one global array (ADR-008), so a fractional pixel
        count anywhere means tiles stop sharing a grid and seams reappear. An
        integer ``pixels_per_degree`` already guarantees this for whole-degree
        spans; what remains is a fractional ``tile_size_degrees`` or latitude
        bound, which this catches at import rather than deep inside a write.
        """
        spans = {
            "global longitude span (360 deg)": 360.0,
            "global latitude span": self.max_latitude - self.min_latitude,
            "tile size": self.tile_size_degrees,
        }
        for label, degrees in spans.items():
            pixels = degrees * self.pixels_per_degree
            if pixels != int(pixels):
                msg = (
                    f"{label} of {degrees} deg is {pixels} pixels at "
                    f"{self.pixels_per_degree} px/deg, which is not a whole number. "
                    "Tiles would not share a grid."
                )
                raise ValueError(msg)
        return self


settings = Settings()
