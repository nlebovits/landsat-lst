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
        default=4.0,
        ge=0.25,
        description="Resident budget for one phase-A spatial block. The block edge is "
        "chosen as the largest power of two whose stack (edge^2 * scenes * 4 bytes) "
        "fits, so unit memory stays flat as the window grows rather than scaling with "
        "it. Raising this enlarges the I/O unit and cuts the number of COG opens; it "
        "does not change any value.",
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
        default=256,
        description="Spatial (lat/lon) chunk size for odc-stac scene loading. "
        "256, not 512, because 512 provably cannot finish a production tile. "
        "Measured on a production-type VM at 4096 squared px "
        "(`landsat-lst benchmark`): at 800 scenes, chunk 512 with 4 threads was "
        "SIGKILLed on 64 GiB while chunk 256 with 1 thread peaked at 16.2 GB. "
        "Across 50-400 scenes the cut is 3.1x to 4.1x, roughly ten times what "
        "the static floor predicts, because the unmodelled memory scales with "
        "threads * chunk**2 too. Costs ~23% more tasks and, on real data with "
        "real range requests, measured 2.9x slower at this size (see 7fda25c). "
        "See docs/findings-memory-model.md.",
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
        "bytes stay within destripe_unit_memory_gb x workers, and phase B "
        "shrinks the count when its batch spans a native-resolution footprint. "
        "The serial form this replaces ran 324 independent blocks one at a "
        "time on ~1 core of 8; the loop, not the estimator, was the offset "
        "pass's wall clock (2026-08-21 investigation).",
    )
    destripe_io_threads: int = Field(
        default=32,
        ge=1,
        description="Threads in the shared pool that executes unit reads "
        "(dask threaded scheduler, pool= override). This is the number of "
        "concurrent S3 range requests the offset pass can hold in flight, "
        "which is a latency lever, not a CPU one: at the previous effective "
        "concurrency of 4, a VM used ~1% of its NIC and ~1.2 of 8 cores while "
        "~84% of wall clock was request latency (batch4/scale, 2026-08). "
        "Threads here spend their time in GIL-released GDAL reads, so the "
        "count may exceed CPUs by a wide margin. Tune with the Stage-2 "
        "concurrency-ladder probe before raising past ~128: the S3 side has "
        "per-prefix request-rate ceilings.",
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
    coiled_max_workers: int = Field(
        default=4,
        description="Ceiling on VMs running batch tasks at once. Coiled gives "
        "each task its own VM and queues the rest, so this is the cost cap: a "
        "700-tile job never runs 700 machines.",
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
