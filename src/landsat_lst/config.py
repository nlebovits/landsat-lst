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
        description="Maximum cloud cover percentage for scene filtering. "
        "Set to 100 to disable scene-level filtering and rely on pixel-level QA.",
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
        "covering less valid land than this. Counts from a coarse offset grid are "
        "scaled up before comparison, so the threshold keeps one meaning whatever "
        "destripe_offset_resolution_factor is set to.",
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
        "Pergamino: factor 2 reproduces native offsets to a median of 0.002 degC "
        "(p99 0.063, max 0.188) with zero keep/reject flips, and 4 is the largest "
        "that passes. Offset error grows linearly in the factor, so do not raise "
        "this without re-running scripts/validate_offset_subsampling.py. The saving "
        "caps out near 2x regardless, since the P95 still needs a native pass.",
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
        description="Spatial (lat/lon) chunk size for odc-stac scene loading. "
        "Smaller values shrink each per-block time stack, cutting peak memory for "
        "the P95 quantile on multi-year/large-tile runs (e.g. 256 for a 4x drop).",
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
        default="6 hours",
        description="Wall-clock budget per batch task. A tile that runs past "
        "this is stuck, not slow; the timeout stops it from billing all night.",
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
