"""Storage abstraction for local and S3 backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import icechunk

from landsat_lst.config import settings


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def cog_exists(self, year: int, tile_name: str) -> bool:
        """Check if a COG already exists for this tile/year."""

    @abstractmethod
    def cog_path(self, year: int, tile_name: str) -> str:
        """Return the path/key where a COG should be written."""

    @abstractmethod
    def icechunk_storage(self) -> icechunk.Storage:
        """Return Icechunk storage configuration."""

    @abstractmethod
    def virtual_chunk_store(self) -> Any:
        """Return object store for virtual chunk access."""

    @abstractmethod
    def virtual_chunk_store_and_prefix(self) -> tuple[Any, str]:
        """Return (store, prefix) tuple for ObjectStoreRegistry."""

    @abstractmethod
    def virtual_chunk_credentials(self) -> Any:
        """Return credentials for virtual chunk container (or None)."""

    @abstractmethod
    def cog_container_prefix(self) -> str:
        """Return URL prefix for COG container (e.g., 's3://bucket/')."""


class LocalStorage(StorageBackend):
    """Local filesystem storage for testing."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or settings.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def cog_exists(self, year: int, tile_name: str) -> bool:
        return self._cog_file(year, tile_name).exists()

    def cog_path(self, year: int, tile_name: str) -> str:
        path = self._cog_file(year, tile_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def icechunk_storage(self) -> icechunk.Storage:
        icechunk_path = self.output_dir / settings.icechunk_prefix
        icechunk_path.mkdir(parents=True, exist_ok=True)
        return icechunk.local_filesystem_storage(str(icechunk_path))

    def virtual_chunk_store(self) -> Any:
        import obstore  # noqa: PLC0415

        return obstore.store.LocalStore(str(self.output_dir))

    def virtual_chunk_store_and_prefix(self) -> tuple[Any, str]:
        return self.virtual_chunk_store(), f"file://{self.output_dir}/"

    def virtual_chunk_credentials(self) -> None:
        return None

    def cog_container_prefix(self) -> str:
        return f"file://{self.output_dir}/"

    def _cog_file(self, year: int, tile_name: str) -> Path:
        return self.output_dir / str(year) / f"{tile_name}.tif"


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
            import boto3  # noqa: PLC0415 - lazy import for optional dependency

            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def cog_exists(self, year: int, tile_name: str) -> bool:
        from botocore.exceptions import ClientError  # noqa: PLC0415

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._cog_key(year, tile_name))
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def cog_path(self, year: int, tile_name: str) -> str:
        return f"s3://{self.bucket}/{self._cog_key(year, tile_name)}"

    def icechunk_storage(self) -> icechunk.Storage:
        return icechunk.s3_storage(
            bucket=self.bucket,
            prefix=f"{self.prefix}/{settings.icechunk_prefix}",
            region=self.region,
        )

    def virtual_chunk_store(self) -> Any:
        import obstore  # noqa: PLC0415

        return obstore.store.S3Store(bucket=self.bucket, region=self.region)

    def virtual_chunk_store_and_prefix(self) -> tuple[Any, str]:
        return self.virtual_chunk_store(), f"s3://{self.bucket}/"

    def virtual_chunk_credentials(self) -> Any:
        return None

    def cog_container_prefix(self) -> str:
        return f"s3://{self.bucket}/"

    def _cog_key(self, year: int, tile_name: str) -> str:
        return f"{self.prefix}/{year}/{tile_name}.tif"


def get_storage() -> StorageBackend:
    """Factory function to get the configured storage backend."""
    if settings.storage_backend == "s3":
        return S3Storage()
    return LocalStorage()
