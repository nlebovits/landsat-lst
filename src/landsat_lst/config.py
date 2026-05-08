"""Configuration and settings for the LST pipeline."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pipeline configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="LST_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    stac_url: str = Field(
        default="https://planetarycomputer.microsoft.com/api/stac/v1",
        description="STAC API endpoint URL (Planetary Computer has free egress)",
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

    # S3 storage (production)
    s3_bucket: str = Field(
        default="source-coop-radiant-earth",
        description="S3 bucket for COG storage",
    )
    s3_prefix: str = Field(
        default="landsat-lst",
        description="S3 key prefix for COGs",
    )
    s3_region: str = Field(
        default="us-west-2",
        description="AWS region for S3 bucket",
    )

    # Icechunk storage
    icechunk_prefix: str = Field(
        default="icechunk",
        description="Subdirectory/prefix for Icechunk store (under output_dir or s3_prefix)",
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
        default=20,
        description="Maximum cloud cover percentage for scene filtering",
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
