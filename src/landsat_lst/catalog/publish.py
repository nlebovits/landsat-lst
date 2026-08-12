"""Copy a built catalog tree to the S3 prefix it is published under.

A built catalog is a directory of JSON, markdown, PNG, and COG files whose
structural links are all relative, so publishing it is a file copy that
preserves the tree shape. Nothing is rewritten on the way out.

Two things make this more than ``aws s3 sync``. Every object needs the media
type the catalog declares for it, because a Portolan client reads
``Content-Type`` off the response and a COG served as
``application/octet-stream`` is not recognised as one. And a republish must not
re-send the assets that have not changed, because the assets are the terabytes.

Freshness is decided by size, never by checksum. The catalog deliberately omits
``file:checksum`` (see :mod:`landsat_lst.catalog.validation`), and recomputing
one per object at publish time would reintroduce exactly the cost that decision
avoids. Size is a sound test for the COGs, whose bytes come out of a
deterministic export: a changed composite changes the compressed length. It is
not sound for metadata, where an edit can leave the byte count identical, so
JSON and markdown are re-sent unconditionally. They are kilobytes; the
correctness is worth more than the transfer.

    from landsat_lst.catalog.publish import publish_catalog

    summary = publish_catalog("./catalog", "s3://bucket/prefix/", dry_run=True)
    for upload in summary.planned:
        print(upload.key, upload.content_type)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from landsat_lst.catalog.spec import COG_MEDIA_TYPE

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["PublishSummary", "Upload", "content_type_for", "publish_catalog"]

#: The media type each published extension is served as. A COG carries the same
#: string the catalog's assets declare, so the header and the metadata agree.
CONTENT_TYPES: dict[str, str] = {
    ".json": "application/json",
    ".geojson": "application/geo+json",
    ".md": "text/markdown",
    ".png": "image/png",
    ".parquet": "application/vnd.apache.parquet",
    ".tif": COG_MEDIA_TYPE,
    ".tiff": COG_MEDIA_TYPE,
}

#: What an unrecognised extension is served as. Reaching this is a sign the
#: builder started emitting something this map has not been taught about.
DEFAULT_CONTENT_TYPE = "application/octet-stream"

#: Extensions re-sent on every publish rather than size-compared. See the module
#: docstring: an equal-sized metadata edit is realistic, and these files are
#: small enough that the check is not worth its risk.
_ALWAYS_UPLOAD = frozenset({".json", ".geojson", ".md"})


def content_type_for(path: Path | str) -> str:
    """The ``Content-Type`` one published file is served under."""
    return CONTENT_TYPES.get(Path(path).suffix.lower(), DEFAULT_CONTENT_TYPE)


@dataclass(frozen=True)
class Upload:
    """One local file and the object it becomes."""

    local: Path
    key: str
    size: int
    content_type: str

    def __str__(self) -> str:
        return f"{self.key}  {self.size} bytes  {self.content_type}"


@dataclass(frozen=True)
class PublishSummary:
    """What a publish did, or what a dry run would have done."""

    bucket: str
    prefix: str
    planned: tuple[Upload, ...] = ()
    skipped: tuple[Upload, ...] = ()
    dry_run: bool = False
    #: Set only on a real run, so a dry run cannot be mistaken for a transfer.
    uploaded: tuple[Upload, ...] = field(default=())

    @property
    def uploaded_count(self) -> int:
        return len(self.uploaded)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def uploaded_bytes(self) -> int:
        return sum(upload.size for upload in self.uploaded)

    @property
    def planned_bytes(self) -> int:
        return sum(upload.size for upload in self.planned)


def parse_s3_uri(remote: str) -> tuple[str, str]:
    """Split ``s3://bucket/prefix/`` into a bucket and a normalised prefix.

    The prefix comes back without a leading slash and with exactly one trailing
    slash, or empty for a bucket root, so joining a relative path onto it is a
    concatenation.
    """
    parsed = urlparse(remote)
    if parsed.scheme != "s3" or not parsed.netloc:
        msg = f"remote must be an s3://bucket/prefix URI, got: {remote!r}"
        raise ValueError(msg)
    prefix = parsed.path.strip("/")
    return parsed.netloc, f"{prefix}/" if prefix else ""


def _walk(root: Path) -> Iterator[Path]:
    """Every file in the tree, in a stable order, ignoring dotfiles.

    Sorted so a plan reads the same twice and a diff between two runs is
    meaningful. Dot-prefixed names are build scratch, never catalog content.
    """
    for path in sorted(root.rglob("*")):
        if path.is_file() and not any(
            part.startswith(".") for part in path.relative_to(root).parts
        ):
            yield path


def _remote_sizes(client: Any, bucket: str, prefix: str) -> dict[str, int]:
    """The size of every object already under the prefix, keyed by object key.

    One paginated listing rather than a HEAD per file: a global catalog is
    ~1,400 assets, and the listing costs a handful of requests.
    """
    sizes: dict[str, int] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", ()):
            sizes[obj["Key"]] = obj["Size"]
    return sizes


def plan_uploads(
    root: Path | str, prefix: str, remote_sizes: dict[str, int]
) -> tuple[list[Upload], list[Upload]]:
    """Split the tree into what must be sent and what is already in place.

    Args:
        root: The built catalog directory.
        prefix: The normalised key prefix, as :func:`parse_s3_uri` returns it.
        remote_sizes: Object key to size, for everything already under the
            prefix. An empty mapping plans a first publish.

    Returns:
        The uploads to perform and the uploads skipped as unchanged.
    """
    root = Path(root)
    planned: list[Upload] = []
    skipped: list[Upload] = []
    for path in _walk(root):
        relative = path.relative_to(root).as_posix()
        upload = Upload(
            local=path,
            key=f"{prefix}{relative}",
            size=path.stat().st_size,
            content_type=content_type_for(path),
        )
        unchanged = remote_sizes.get(upload.key) == upload.size
        if unchanged and path.suffix.lower() not in _ALWAYS_UPLOAD:
            skipped.append(upload)
        else:
            planned.append(upload)
    return planned, skipped


def _s3_client(profile: str | None) -> Any:
    """A boto3 S3 client, optionally bound to a named profile."""
    import boto3  # noqa: PLC0415 - only a real publish pays for the SDK import

    if profile is None:
        return boto3.client("s3")
    return boto3.Session(profile_name=profile).client("s3")


def publish_catalog(
    root: Path | str,
    remote: str,
    *,
    dry_run: bool = False,
    profile: str | None = None,
    client: Any | None = None,
) -> PublishSummary:
    """Sync a built catalog tree to an ``s3://`` prefix.

    Args:
        root: The catalog directory to publish, as ``build_catalog`` wrote it.
        remote: ``s3://bucket/prefix/`` to publish under.
        dry_run: Plan the publish without uploading anything. The listing of
            what is already published still runs, so the plan reflects the real
            remote state.
        profile: An AWS named profile to authenticate with. Ignored when
            ``client`` is given.
        client: An S3 client to use instead of constructing one. Tests pass a
            stub; callers reusing a session pass theirs.

    Returns:
        What was uploaded, what was skipped, and how many bytes moved.

    Raises:
        ValueError: ``remote`` is not an ``s3://`` URI.
        NotADirectoryError: ``root`` is not a directory.
    """
    root = Path(root)
    if not root.is_dir():
        msg = f"catalog root is not a directory: {root}"
        raise NotADirectoryError(msg)
    bucket, prefix = parse_s3_uri(remote)
    if client is None:
        client = _s3_client(profile)
    planned, skipped = plan_uploads(root, prefix, _remote_sizes(client, bucket, prefix))
    if dry_run:
        return PublishSummary(
            bucket=bucket,
            prefix=prefix,
            planned=tuple(planned),
            skipped=tuple(skipped),
            dry_run=True,
        )
    for upload in planned:
        client.upload_file(
            str(upload.local),
            bucket,
            upload.key,
            ExtraArgs={"ContentType": upload.content_type},
        )
    return PublishSummary(
        bucket=bucket,
        prefix=prefix,
        planned=tuple(planned),
        skipped=tuple(skipped),
        uploaded=tuple(planned),
    )
