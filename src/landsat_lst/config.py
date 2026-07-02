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

    # Multiscale overviews (GeoZarr multiscales convention)
    pyramid_factors: list[int] = Field(
        default=[4, 16, 64],
        description="Downsample factors (relative to native) for overview levels. "
        "Default is a sparse 4x pyramid (4x/16x/64x): ~6.7% storage overhead, "
        "tuned for mostly zoomed-out global viewing. Use [2, 4, 8, 16, 32, 64] for "
        "a full 2x pyramid (~33% overhead, smoother near-native zoom).",
    )

    # Compression (Zarr v3 codec)
    compression_codec: str = Field(
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
