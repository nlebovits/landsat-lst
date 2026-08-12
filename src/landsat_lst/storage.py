"""Storage abstraction for COG outputs, on local disk or S3.

Two backends:

- :class:`LocalStorage`: writes under ``settings.output_dir`` (testing).
- :class:`S3Storage`: writes under ``s3://{bucket}/{prefix}`` (production).

Each tile-window produces **two** assets, ``lst_p95`` and ``qa_count``, laid out
as ``{window}/{tile}/{product}_{window}_{tile}.tif``. Both must be present for a
tile to count as done: a tile with one uploaded asset is a half-written tile and
has to be rebuilt, so :meth:`StorageBackend.cog_exists` always checks both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path  # noqa: TC003 - used at runtime in type hints
from typing import Any

from landsat_lst.config import settings

#: The two assets that together make one complete tile-window output.
PRODUCTS: tuple[str, str] = ("lst_p95", "qa_count")


class StorageBackend(ABC):
    """Abstract base class for COG storage backends."""

    def cog_key(self, window: str, tile: str, product: str) -> str:
        """Backend-relative key for one asset.

        Concrete rather than abstract because the layout is the contract shared
        by every backend: if local and S3 disagreed on it, a catalog built from
        one would not resolve against the other.

        Args:
            window: Window label (``"2024"`` or ``"2021-2025"``).
            tile: Tile name (``"N40W075"``).
            product: Asset name, one of :data:`PRODUCTS`.

        Returns:
            ``{window}/{tile}/{product}_{window}_{tile}.tif``
        """
        return f"{window}/{tile}/{product}_{window}_{tile}.tif"

    @abstractmethod
    def cog_exists(self, window: str, tile: str) -> bool:
        """Whether **both** assets for this tile-window are already stored."""

    @abstractmethod
    def upload(self, local: Path, key: str) -> None:
        """Store ``local`` at ``key`` (as returned by :meth:`cog_key`)."""

    @abstractmethod
    def list_completed(self, window: str) -> set[str]:
        """Tile names in ``window`` that have both assets stored."""


class LocalStorage(StorageBackend):
    """Local filesystem storage for testing."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or settings.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def cog_exists(self, window: str, tile: str) -> bool:
        return all(
            (self.output_dir / self.cog_key(window, tile, product)).exists() for product in PRODUCTS
        )

    def upload(self, local: Path, key: str) -> None:
        dest = self.output_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(local.read_bytes())

    def list_completed(self, window: str) -> set[str]:
        window_dir = self.output_dir / window
        if not window_dir.is_dir():
            return set()
        return {
            d.name for d in window_dir.iterdir() if d.is_dir() and self.cog_exists(window, d.name)
        }


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

    def _full_key(self, key: str) -> str:
        """Resolve a backend-relative key against the bucket prefix."""
        return f"{self.prefix}/{key}"

    def _head(self, key: str) -> bool:
        """Whether one object exists, distinguishing 404 from a real failure."""
        from botocore.exceptions import ClientError  # noqa: PLC0415

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._full_key(key))
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise
        return True

    def cog_exists(self, window: str, tile: str) -> bool:
        return all(self._head(self.cog_key(window, tile, product)) for product in PRODUCTS)

    def upload(self, local: Path, key: str) -> None:
        self.client.upload_file(str(local), self.bucket, self._full_key(key))

    def list_completed(self, window: str) -> set[str]:
        """One paginated listing of the window prefix, rather than 2N head requests."""
        prefix = self._full_key(f"{window}/")
        seen: set[str] = set()
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                seen.add(obj["Key"])
        return {
            tile
            for tile in {key[len(prefix) :].split("/")[0] for key in seen}
            if all(self._full_key(self.cog_key(window, tile, p)) in seen for p in PRODUCTS)
        }


def get_storage() -> StorageBackend:
    """Factory function to get the configured storage backend."""
    if settings.storage_backend == "s3":
        return S3Storage()
    return LocalStorage()
