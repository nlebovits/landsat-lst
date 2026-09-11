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
only channel a batch VM has back to whoever is watching. Every one of them is
keyed by attempt, because ``settings.coiled_retries`` is 3 and a retry used to
overwrite the attempt before it:

- ``{tile}.{attempt}.json`` -- the **state object**, rewritten every
  ``settings.heartbeat_interval_s`` while the tile runs and once more when it
  settles. It carries both the live phase and the outcome, so a wedged tile is
  distinguishable from a busy one and a finished tile reports its own duration,
  scene count, peak memory, and error (:mod:`landsat_lst.progress`).
- ``{tile}.json`` -- a copy of the final state, written once when the tile
  stops. A reader that knows only this key finds a superset of what the old
  run record held, and its presence is what tells every reader the tile
  settled.
- ``{tile}.{attempt}.log`` -- the task's captured stdout and stderr, uploaded
  when it exits either way. Coiled's own logs never carry it and its exit code
  is the tee wrapper's, so a failed tile explains itself here or nowhere.
- ``{tile}.{attempt}.{label}.profile.json`` -- a dask profile dump, only when
  ``settings.profile_dask`` is on.

:mod:`landsat_lst.runs` owns the grammar of these names and is the one place
that parses them. :func:`landsat_lst.batch.reconcile_run` reads them back.

The prefix is a sibling of the collection directories and is invisible to the
catalog, which only ever reads ``lst-p95-*``.
"""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from landsat_lst.config import settings

#: The two assets that together make one complete tile-window output.
PRODUCTS: tuple[str, str] = ("lst_p95", "qa_count")

#: The pooled all-path P95, emitted beside ``lst_p95`` for comparison when
#: ``settings.wrs_emit_pooled_baseline`` is on. Deliberately NOT in
#: :data:`PRODUCTS`: that tuple is the completion contract every reader uses to
#: decide whether a tile is finished, and a diagnostic asset must not be able
#: to hold a tile incomplete.
POOLED_PRODUCT = "lst_p95_pooled"


def band_products() -> tuple[str, ...]:
    """Products a composite shard writes and the export merges.

    :data:`PRODUCTS` plus the pooled baseline when it is enabled.
    """
    from landsat_lst.config import settings  # noqa: PLC0415

    if settings.wrs_feather and settings.wrs_emit_pooled_baseline:
        return (*PRODUCTS, POOLED_PRODUCT)
    return PRODUCTS


#: Prefix for per-tile run records, a sibling of the collection directories.
RUN_RECORD_PREFIX = "_runs"

#: Prefix for cached per-scene de-striping offsets. Another sibling of the
#: collection directories, invisible to the catalog. See :mod:`landsat_lst.offsets`.
OFFSET_PREFIX = "_offsets"


def offset_cache_key(
    *, tile: str, window: str, factor: int, algorithm_version: int, digest: str
) -> str:
    """Backend-relative key for one tile-window's cached scene offsets.

    Every term that changes the offsets is in the path rather than only in the
    digest, so a bucket listing is readable: a human can see which factor and
    which algorithm version a record belongs to without opening it. The digest
    is what actually makes a stale record unreachable from a changed input.

    Args:
        tile: Tile name (``"N40W075"``).
        window: Window label (``"2021-2025"``, or ``"2021-2025-sample300"``).
        factor: ``destripe_offset_resolution_factor`` the offsets were estimated at.
        algorithm_version: :data:`landsat_lst.offsets.ALGORITHM_VERSION`.
        digest: Hash over the scene ids and the settings that shape the estimate.

    Returns:
        ``_offsets/{tile}/{window}/f{factor}/v{version}-{digest}.json``
    """
    return f"{OFFSET_PREFIX}/{tile}/{window}/f{factor}/v{algorithm_version}-{digest}.json"


def _attempt_segment(attempt: int | None) -> str:
    """``".2"`` for attempt 2, and nothing at all for the settled pointer."""
    return "" if attempt is None else f".{attempt}"


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

    def run_record_key(self, run_id: str, tile: str, attempt: int | None = None) -> str:
        """Backend-relative key for one tile's state object.

        ``attempt`` selects one attempt's own object. ``None`` gives the
        unsuffixed key, which a settled tile copies its final state to and
        which a run written before this scheme used for its only record.

        Every Coiled retry used to write the one unsuffixed key here, so the
        last attempt erased the ones before it. That is why run
        ``2021-2025-20260814T092642Z`` reported a 10-second failure against a
        33-minute wall clock. See issue #92.

        Args:
            run_id: Run token the object belongs to.
            tile: Tile name (``"N40W075"``).
            attempt: Attempt number, or ``None`` for the settled-state pointer.

        Returns:
            ``_runs/{run_id}/{tile}.{attempt}.json``, or
            ``_runs/{run_id}/{tile}.json`` when ``attempt`` is ``None``.
        """
        return f"{RUN_RECORD_PREFIX}/{run_id}/{tile}{_attempt_segment(attempt)}.json"

    def profile_key(self, run_id: str, tile: str, label: str, attempt: int | None = None) -> str:
        """Backend-relative key for one dask profile dump.

        Written at most once per labelled compute, and only when
        ``settings.profile_dask`` is on, so it sits beside the state object and
        the log rather than replacing either.

        Args:
            run_id: Run token the profile belongs to.
            tile: Tile name (``"N40W075"``).
            label: Which compute was profiled (``"destripe_offsets"``).
            attempt: Attempt number, or ``None`` for an unsuffixed key.

        Returns:
            ``_runs/{run_id}/{tile}.{attempt}.{label}.profile.json``
        """
        segment = _attempt_segment(attempt)
        return f"{RUN_RECORD_PREFIX}/{run_id}/{tile}{segment}.{label}.profile.json"

    def log_key(self, run_id: str, tile: str, attempt: int | None = None) -> str:
        """Backend-relative key for one tile's captured task log.

        Returns:
            ``_runs/{run_id}/{tile}.{attempt}.log``, or
            ``_runs/{run_id}/{tile}.log`` when ``attempt`` is ``None``.
        """
        return f"{RUN_RECORD_PREFIX}/{run_id}/{tile}{_attempt_segment(attempt)}.log"

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
    def download(self, key: str, local: Path) -> bool:
        """Fetch ``key`` to ``local``, returning whether it existed.

        The binary counterpart of :meth:`read_text`, and the inverse of
        :meth:`upload`. A sharded tile needs it twice: the phase-B shards
        reassemble the climatology from ``.npy`` blocks, and the export-merge
        stitches per-band GeoTIFFs. Neither is text and neither fits in a
        heartbeat-sized object, so ``read_text`` cannot serve them.

        A missing key returns ``False`` rather than raising, for the reason
        :meth:`read_text` documents: a shard that never published is an
        ordinary outcome the caller decides about.
        """

    @abstractmethod
    def list_prefix(self, prefix: str) -> dict[str, datetime]:
        """Every key under ``prefix``, mapped to when it was last written.

        ``prefix`` is matched as a key prefix, not as a directory path, so a
        partial final segment such as ``_runs/{run_id}/N40W075.`` selects one
        tile's artifacts. Both backends must honour that, because a caller
        holding a :class:`StorageBackend` cannot know which one it has.

        The timestamp comes from the store rather than from the object's own
        contents, so a watcher measures heartbeat age against one clock instead
        of trusting a VM's. An absent prefix maps to an empty dict.
        """

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int:
        """Delete every object under ``prefix``. Returns how many were removed.

        Staging (issue #125) is the only writer that needs this: a coarse
        stage is scratch that must not outlive the record it produced, because
        a later listing under the run prefix reads leftovers as finished work.
        Deleting nothing is success -- the caller cannot tell a swept prefix
        from one that never existed, and must not care.
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
        """Write ``text`` at ``key`` atomically, via a temporary file and a rename.

        A reader must never see half an object. S3 gives that for free, since a
        PUT is atomic and a failed one leaves no key; a plain ``write_text``
        does not, and a process killed mid-write would leave a truncated
        heartbeat or a truncated offset record that parses as far as it goes.
        ``Path.replace`` on the same filesystem is the local equivalent. See
        issue #77 item 1.
        """
        del content_type  # a filesystem records the media type in the suffix
        dest = self.output_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Same directory as the destination, so the rename never crosses a
        # filesystem boundary and so stays atomic.
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - renamed, not closed-and-read
            "w", dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp", delete=False
        )
        try:
            with handle:
                handle.write(text)
            Path(handle.name).replace(dest)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise

    def read_text(self, key: str) -> str | None:
        path = self.output_dir / key
        return path.read_text() if path.is_file() else None

    def download(self, key: str, local: Path) -> bool:
        path = self.output_dir / key
        if not path.is_file():
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(path.read_bytes())
        return True

    def list_prefix(self, prefix: str) -> dict[str, datetime]:
        """Every key under ``prefix``, mapped to when it was last written.

        ``prefix`` is a key prefix, not a directory path, so
        ``_runs/{run_id}/N40W075.`` lists one tile's artifacts here exactly as
        it does on S3. Reading it as a directory returned nothing locally for a
        partial final segment while returning the right answer in production,
        which would have made every test of a tile-scoped listing pass
        vacuously against the one fake backend this repo has.
        """
        head, _, _ = prefix.rpartition("/")
        root = self.output_dir / head if head else self.output_dir
        if not root.is_dir():
            return {}
        return {
            key: datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            for path in sorted(root.rglob("*"))
            if path.is_file() and (key := str(path.relative_to(self.output_dir))).startswith(prefix)
        }

    def delete_prefix(self, prefix: str) -> int:
        removed = 0
        for key in list(self.list_prefix(prefix)):
            path = self.output_dir / key
            if path.is_file():
                path.unlink()
                removed += 1
        return removed


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

    def download(self, key: str, local: Path) -> bool:
        from botocore.exceptions import ClientError  # noqa: PLC0415

        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(self.bucket, self._full_key(key), str(local))
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise
        return True

    def list_prefix(self, prefix: str) -> dict[str, datetime]:
        full = self._full_key(prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        listed: dict[str, datetime] = {}
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for obj in page.get("Contents", []):
                listed[obj["Key"][len(self.prefix) + 1 :]] = obj["LastModified"]
        return listed

    def delete_prefix(self, prefix: str) -> int:
        full = self._full_key(prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        removed = 0
        batch: list[dict] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for obj in page.get("Contents", []):
                batch.append({"Key": obj["Key"]})
                if len(batch) == 1000:
                    removed += self._delete_batch(batch)
                    batch = []
        if batch:
            removed += self._delete_batch(batch)
        return removed

    def _delete_batch(self, batch: list[dict]) -> int:
        response = self.client.delete_objects(
            Bucket=self.bucket, Delete={"Objects": batch, "Quiet": True}
        )
        return len(batch) - len(response.get("Errors", []))


def get_storage() -> StorageBackend:
    """Factory function to get the configured storage backend."""
    if settings.storage_backend == "s3":
        return S3Storage()
    return LocalStorage()
