"""Configuration and settings for the LST pipeline."""

from pathlib import Path
from typing import Literal

from pydantic import Field
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
        description="Local output directory for Zarr stores (used when storage_backend='local')",
    )

    # S3 storage (production)
    s3_bucket: str = Field(
        default="source-coop-radiant-earth",
        description="S3 bucket for Zarr/Icechunk storage",
    )
    s3_prefix: str = Field(
        default="landsat-lst",
        description="S3 key prefix for Zarr stores",
    )
    s3_region: str = Field(
        default="us-west-2",
        description="AWS region for S3 bucket",
    )

    # Icechunk storage
    use_icechunk: bool = Field(
        default=False,
        description="Use Icechunk for versioned Zarr storage (enables time-travel)",
    )
    icechunk_prefix: str = Field(
        default="icechunk",
        description="Subdirectory/prefix for Icechunk store (under output_dir or s3_prefix)",
    )
    icechunk_max_retries: int = Field(
        default=5,
        description="Maximum retry attempts for Icechunk ConflictError",
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

    resolution: float = Field(
        default=0.00027778,
        description="Output resolution in degrees (~30m at equator)",
    )
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
        default=1,
        description="Estimate per-scene offsets from a stack loaded at "
        "resolution * factor, served from the source COGs' internal overviews "
        "([2,4,8,16,32,64]). 1 keeps offsets at native resolution. The offset is a "
        "single scalar per scene, so spatial detail buys nothing, and a coarse read "
        "cuts bytes fetched rather than merely compute. Validate a new value against "
        "scripts/validate_offset_subsampling.py before shipping it.",
    )

    # Multiscale overviews (GeoZarr multiscales convention)
    pyramid_factors: list[int] = Field(
        default=[4, 16, 64],
        description="Downsample factors (relative to native) for overview levels. "
        "Default is a sparse 4x pyramid (4x/16x/64x): ~6.7% storage overhead, "
        "tuned for mostly zoomed-out global viewing. Use [2, 4, 8, 16, 32, 64] for "
        "a full 2x pyramid (~33% overhead, smoother near-native zoom).",
    )

    # Compression (Zarr v3 codec)
    # Literal rather than str so an unsupported codec fails at settings load
    # instead of deep inside the Zarr write, and so the value satisfies the
    # codec name type BloscCodec expects.
    compression_codec: Literal["lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd"] = Field(
        default="zstd",
        description="Blosc compression codec name (e.g. 'zstd', 'lz4', 'blosclz').",
    )
    compression_level: int = Field(
        default=5,
        description="Blosc compression level (0-9). 0 disables compression.",
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

    # Coiled retry settings
    coiled_retries: int = Field(
        default=3,
        description="Number of retries for Coiled worker failures",
    )


settings = Settings()
