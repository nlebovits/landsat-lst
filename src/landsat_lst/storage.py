"""Storage abstraction for local, S3, and Icechunk Zarr backends.

Supports three storage modes:
- LocalStorage: Plain Zarr stores on local filesystem (testing)
- S3Storage: Plain Zarr stores on S3 (production without versioning)
- IcechunkStorage: Versioned Zarr in Icechunk repository (production with versioning)

See ADR-003 for architecture details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path  # noqa: TC003 - used at runtime in type hints
from typing import TYPE_CHECKING, Any

from landsat_lst.config import settings

if TYPE_CHECKING:
    import icechunk as ic


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def zarr_exists(self, year: int | str, tile_name: str) -> bool:
        """Check if a Zarr store already exists for this tile/window.

        ``year`` is a window label: an ``int`` year (``2024``) or a multi-year
        range string (``"2020-2024"``); it is the top-level group/prefix.
        """

    @abstractmethod
    def zarr_path(self, year: int | str, tile_name: str) -> str:
        """Return the path/URL where a Zarr store should be written."""


class LocalStorage(StorageBackend):
    """Local filesystem storage for testing."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or settings.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def zarr_exists(self, year: int | str, tile_name: str) -> bool:
        return self._zarr_dir(year, tile_name).exists()

    def zarr_path(self, year: int | str, tile_name: str) -> str:
        path = self._zarr_dir(year, tile_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _zarr_dir(self, year: int | str, tile_name: str) -> Path:
        return self.output_dir / str(year) / f"{tile_name}.zarr"


class S3Storage(StorageBackend):
    """S3 storage for production (plain Zarr, no versioning)."""

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

    def zarr_exists(self, year: int | str, tile_name: str) -> bool:
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

    def zarr_path(self, year: int | str, tile_name: str) -> str:
        return f"s3://{self.bucket}/{self._zarr_key(year, tile_name)}"

    def _zarr_key(self, year: int | str, tile_name: str) -> str:
        return f"{self.prefix}/{year}/{tile_name}.zarr"


class IcechunkStorage(StorageBackend):
    """Icechunk repository storage for versioned Zarr writes.

    All tiles are stored in a single repository as GeoZarr multiscale pyramids
    (see ADR-004). The tile group carries GeoZarr proj/spatial/multiscales metadata;
    native data lives in level subgroup ``0`` and overviews in ``1``/``2``/``3``:

        /{year}/{tile_name}/        # GeoZarr multiscales + proj/spatial attrs
            0/                      # native resolution
                lst_p95/
                qa_count/
            1/ 2/ 3/                # 4x / 16x / 64x overviews

    Each tile write is a single commit (native + all overviews), enabling
    time-travel and audit trail.
    """

    def __init__(self, repo: ic.Repository):
        self._repo = repo

    @classmethod
    def from_local(cls, path: Path) -> IcechunkStorage:
        """Create IcechunkStorage with local filesystem backend."""
        import icechunk as ic  # noqa: PLC0415

        path.mkdir(parents=True, exist_ok=True)
        storage = ic.local_filesystem_storage(str(path))
        repo = ic.Repository.open_or_create(storage)
        return cls(repo)

    @classmethod
    def from_s3(
        cls,
        bucket: str,
        prefix: str,
        region: str,
    ) -> IcechunkStorage:
        """Create IcechunkStorage with S3 backend."""
        import icechunk as ic  # noqa: PLC0415

        storage = ic.s3_storage(
            bucket=bucket,
            prefix=prefix,
            region=region,
            from_env=True,
        )
        repo = ic.Repository.open_or_create(storage)
        return cls(repo)

    def zarr_exists(self, year: int | str, tile_name: str) -> bool:
        """Check if tile-year group exists in repository."""
        import zarr  # noqa: PLC0415

        try:
            session = self._repo.readonly_session("main")
            group_path = f"{year}/{tile_name}"
            zarr.open_group(session.store, path=group_path, mode="r")
            return True
        except (KeyError, FileNotFoundError):
            return False

    def zarr_path(self, year: int | str, tile_name: str) -> str:
        """Return the group path within Icechunk (not a file path)."""
        return f"{year}/{tile_name}"

    def writable_session(self) -> ic.Session:
        """Get a writable session for the main branch."""
        return self._repo.writable_session("main")

    def readonly_session(self) -> ic.Session:
        """Get a readonly session for reading."""
        return self._repo.readonly_session("main")

    @property
    def repo(self) -> ic.Repository:
        """Access the underlying repository."""
        return self._repo


def get_storage() -> StorageBackend:
    """Factory function to get the configured storage backend."""
    if settings.use_icechunk:
        if settings.storage_backend == "s3":
            return IcechunkStorage.from_s3(
                bucket=settings.s3_bucket,
                prefix=f"{settings.s3_prefix}/{settings.icechunk_prefix}",
                region=settings.s3_region,
            )
        icechunk_path = settings.output_dir / settings.icechunk_prefix
        return IcechunkStorage.from_local(icechunk_path)

    if settings.storage_backend == "s3":
        return S3Storage()
    return LocalStorage()
