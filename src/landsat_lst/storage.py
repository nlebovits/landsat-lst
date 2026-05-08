"""Storage abstraction for local and S3 Zarr backends.

Simplified from the original COG + Icechunk version per ADR-003.
Now only handles Zarr store existence checks and path generation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from landsat_lst.config import settings


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def zarr_exists(self, year: int, tile_name: str) -> bool:
        """Check if a Zarr store already exists for this tile/year."""

    @abstractmethod
    def zarr_path(self, year: int, tile_name: str) -> str:
        """Return the path/URL where a Zarr store should be written."""


class LocalStorage(StorageBackend):
    """Local filesystem storage for testing."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or settings.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def zarr_exists(self, year: int, tile_name: str) -> bool:
        return self._zarr_dir(year, tile_name).exists()

    def zarr_path(self, year: int, tile_name: str) -> str:
        path = self._zarr_dir(year, tile_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _zarr_dir(self, year: int, tile_name: str) -> Path:
        return self.output_dir / str(year) / f"{tile_name}.zarr"


class S3Storage(StorageBackend):
    """S3 storage for production."""

    def __init__(
        self,
        bucket: str | None = None,
        prefix: str | None = None,
        region: str | None = None,
    ):
        self.bucket = bucket or settings.s3_bucket
        self.prefix = prefix or settings.s3_prefix
        self.region = region or settings.s3_region
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3  # noqa: PLC0415

            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def zarr_exists(self, year: int, tile_name: str) -> bool:
        from botocore.exceptions import ClientError  # noqa: PLC0415

        try:
            # Check for .zmetadata or zarr.json to verify Zarr store exists
            self.client.head_object(
                Bucket=self.bucket,
                Key=f"{self._zarr_key(year, tile_name)}/.zmetadata",
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                # Try zarr.json for Zarr v3
                try:
                    self.client.head_object(
                        Bucket=self.bucket,
                        Key=f"{self._zarr_key(year, tile_name)}/zarr.json",
                    )
                    return True
                except ClientError as e2:
                    if e2.response["Error"]["Code"] == "404":
                        return False
                    raise
            raise

    def zarr_path(self, year: int, tile_name: str) -> str:
        return f"s3://{self.bucket}/{self._zarr_key(year, tile_name)}"

    def _zarr_key(self, year: int, tile_name: str) -> str:
        return f"{self.prefix}/{year}/{tile_name}.zarr"


def get_storage() -> StorageBackend:
    """Factory function to get the configured storage backend."""
    if settings.storage_backend == "s3":
        return S3Storage()
    return LocalStorage()
