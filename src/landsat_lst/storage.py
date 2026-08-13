"""Storage abstraction for COG outputs, on local disk or S3.

Two backends:

- :class:`LocalStorage`: writes under ``settings.output_dir`` (testing).
- :class:`S3Storage`: writes under ``s3://{bucket}/{prefix}`` (production).

Each tile-window produces **two** assets, ``lst_p95`` and ``qa_count``, laid out
as ``lst-p95-{window}/{tile}/{product}_{window}_{tile}.tif``. The leading
segment is the published collection id, so the pipeline writes every COG at the
exact path the STAC catalog will declare and publication syncs metadata only.
A test pins this prefix to ``catalog.spec.spec_for_window`` rather than an
import, which would drag the whole catalog stack into every worker.

Both assets must be present for a tile to count as done: a tile with one
uploaded asset is a half-written tile and has to be rebuilt, so
:meth:`StorageBackend.cog_exists` always checks both.

Each backend also stores small per-tile objects under ``_runs/{run_id}/``, the
only channel a batch VM has back to whoever is watching:

- ``{tile}.json`` -- the **run record**, written once when the tile finishes.
  A distributed run has no live driver to collect results, so the VM reports its
  own duration, scene count, peak memory, and error;
  :func:`landsat_lst.batch.reconcile_run` reads them back.
- ``{tile}.progress.json`` -- the **heartbeat**, rewritten every
  ``settings.heartbeat_interval_s`` while the tile runs, so a wedged tile is
  distinguishable from a busy one (:mod:`landsat_lst.progress`).
- ``{tile}.log`` -- the task's captured stdout and stderr, uploaded when it
  exits either way. Coiled's own logs never carry it and its exit code is the
  tee wrapper's, so a failed tile explains itself here or nowhere.

The prefix is a sibling of the collection directories and is invisible to the
catalog, which only ever reads ``lst-p95-*``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 - used at runtime in type hints
from typing import Any

from landsat_lst.config import settings

#: The two assets that together make one complete tile-window output.
PRODUCTS: tuple[str, str] = ("lst_p95", "qa_count")

#: Prefix for per-tile run records, a sibling of the collection directories.
RUN_RECORD_PREFIX = "_runs"


def collection_prefix(window: str) -> str:
    """The published collection id for one window, e.g. ``lst-p95-2021-2025``.

    Must match ``catalog.spec.spec_for_window(window).collection_id``; the
    contract is asserted by a unit test instead of an import so workers do not
    pay for the catalog stack.
    """
    return f"lst-p95-{window}"


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
            ``lst-p95-{window}/{tile}/{product}_{window}_{tile}.tif``
        """
        return f"{collection_prefix(window)}/{tile}/{product}_{window}_{tile}.tif"

    def run_prefix(self, run_id: str) -> str:
        """Backend-relative prefix holding everything one run reported.

        Returns:
            ``_runs/{run_id}/``
        """
        return f"{RUN_RECORD_PREFIX}/{run_id}/"

    def run_record_key(self, run_id: str, tile: str) -> str:
        """Backend-relative key for one tile's run record.

        Args:
            run_id: Run token the record belongs to.
            tile: Tile name (``"N40W075"``).

        Returns:
            ``_runs/{run_id}/{tile}.json``
        """
        return f"{RUN_RECORD_PREFIX}/{run_id}/{tile}.json"

    def progress_key(self, run_id: str, tile: str) -> str:
        """Backend-relative key for one tile's heartbeat.

        Distinct from :meth:`run_record_key` because the two have opposite
        lifetimes: the heartbeat is overwritten every minute while the tile
        runs, the record is written once at the end and never touched again.

        Returns:
            ``_runs/{run_id}/{tile}.progress.json``
        """
        return f"{RUN_RECORD_PREFIX}/{run_id}/{tile}.progress.json"

    def profile_key(self, run_id: str, tile: str, label: str) -> str:
        """Backend-relative key for one dask profile dump.

        Written at most once per labelled compute, and only when
        ``settings.profile_dask`` is on, so it sits beside the heartbeat and
        the log rather than replacing either.

        Args:
            run_id: Run token the profile belongs to.
            tile: Tile name (``"N40W075"``).
            label: Which compute was profiled (``"destripe_offsets"``).

        Returns:
            ``_runs/{run_id}/{tile}.{label}.profile.json``
        """
        return f"{RUN_RECORD_PREFIX}/{run_id}/{tile}.{label}.profile.json"

    def log_key(self, run_id: str, tile: str) -> str:
        """Backend-relative key for one tile's captured task log.

        Returns:
            ``_runs/{run_id}/{tile}.log``
        """
        return f"{RUN_RECORD_PREFIX}/{run_id}/{tile}.log"

    @abstractmethod
    def cog_exists(self, window: str, tile: str) -> bool:
        """Whether **both** assets for this tile-window are already stored."""

    @abstractmethod
    def upload(self, local: Path, key: str) -> None:
        """Store ``local`` at ``key`` (as returned by :meth:`cog_key`)."""

    @abstractmethod
    def list_completed(self, window: str) -> set[str]:
        """Tile names in ``window`` that have both assets stored."""

    @abstractmethod
    def write_text(self, key: str, text: str, *, content_type: str = "application/json") -> None:
        """Store ``text`` at ``key``, replacing whatever was there.

        Args:
            key: Backend-relative key.
            text: Content to store.
            content_type: Media type recorded with the object. Backends that
                have nowhere to put it ignore it.
        """

    @abstractmethod
    def read_text(self, key: str) -> str | None:
        """Read ``key``, or return ``None`` when it does not exist.

        A missing key is an ordinary outcome, not an error: a tile whose VM was
        killed before it could report leaves no record, and reconciliation has
        to distinguish that from a read failure.
        """

    @abstractmethod
    def list_prefix(self, prefix: str) -> dict[str, datetime]:
        """Every key under ``prefix``, mapped to when it was last written.

        The timestamp comes from the store rather than from the object's own
        contents, so a watcher measures heartbeat age against one clock instead
        of trusting a VM's. An absent prefix maps to an empty dict.
        """


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
        collection_dir = self.output_dir / collection_prefix(window)
        if not collection_dir.is_dir():
            return set()
        return {
            d.name
            for d in collection_dir.iterdir()
            if d.is_dir() and self.cog_exists(window, d.name)
        }

    def write_text(self, key: str, text: str, *, content_type: str = "application/json") -> None:
        del content_type  # a filesystem records the media type in the suffix
        dest = self.output_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)

    def read_text(self, key: str) -> str | None:
        path = self.output_dir / key
        return path.read_text() if path.is_file() else None

    def list_prefix(self, prefix: str) -> dict[str, datetime]:
        root = self.output_dir / prefix
        if not root.is_dir():
            return {}
        return {
            str(path.relative_to(self.output_dir)): datetime.fromtimestamp(
                path.stat().st_mtime, tz=UTC
            )
            for path in sorted(root.rglob("*"))
            if path.is_file()
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
        prefix = self._full_key(f"{collection_prefix(window)}/")
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

    def write_text(self, key: str, text: str, *, content_type: str = "application/json") -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._full_key(key),
            Body=text.encode(),
            ContentType=content_type,
        )

    def read_text(self, key: str) -> str | None:
        from botocore.exceptions import ClientError  # noqa: PLC0415

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._full_key(key))
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return None
            raise
        return response["Body"].read().decode()

    def list_prefix(self, prefix: str) -> dict[str, datetime]:
        full = self._full_key(prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        listed: dict[str, datetime] = {}
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for obj in page.get("Contents", []):
                listed[obj["Key"][len(self.prefix) + 1 :]] = obj["LastModified"]
        return listed


def get_storage() -> StorageBackend:
    """Factory function to get the configured storage backend."""
    if settings.storage_backend == "s3":
        return S3Storage()
    return LocalStorage()
