"""The publish step, exercised against a stub S3 client.

Hermetic and offline. The catalog is a real tree built from synthetic 64x64
COGs, but the client is a ``MagicMock`` configured to answer the one listing
call the planner makes, so no credential is read and no request leaves the
process. The dry-run tests do not even reach an upload path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from landsat_lst.catalog import build_catalog
from landsat_lst.catalog.publish import (
    CONTENT_TYPES,
    DEFAULT_CONTENT_TYPE,
    content_type_for,
    parse_s3_uri,
    plan_uploads,
    publish_catalog,
)
from landsat_lst.catalog.spec import COG_MEDIA_TYPE, DEFAULT_SPEC
from tests.cog_fixtures import write_source_tree, write_thumbnail

if TYPE_CHECKING:
    from pathlib import Path

TILES = ("N40W075", "N35W120")
REMOTE = "s3://us-west-2.opendata.source.coop/nlebovits/landsat-lst/"
PREFIX = "nlebovits/landsat-lst/"


@pytest.fixture(scope="module")
def built_catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A complete catalog built from synthetic COGs, shared across the module."""
    root = tmp_path_factory.mktemp("publish")
    source = write_source_tree(root / "cogs", TILES)
    thumbnail = write_thumbnail(root / "thumbnail.png")
    return build_catalog(source, root / "catalog", thumbnail=thumbnail)


def stub_client(objects: dict[str, int] | None = None) -> MagicMock:
    """An S3 client whose listing reports ``objects`` as already published."""
    client = MagicMock()
    contents = [{"Key": key, "Size": size} for key, size in sorted((objects or {}).items())]
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": contents}] if contents else [{}]
    client.get_paginator.return_value = paginator
    return client


def remote_state(root: Path, prefix: str = PREFIX) -> dict[str, int]:
    """The listing a bucket would return after a complete publish of ``root``."""
    return {
        f"{prefix}{path.relative_to(root).as_posix()}": path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    }


class TestContentTypes:
    """Every extension the builder emits gets the media type it declares."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("catalog.json", "application/json"),
            ("N40W075/N40W075.json", "application/json"),
            ("items.geojson", "application/geo+json"),
            ("README.md", "text/markdown"),
            ("AGENTS.md", "text/markdown"),
            ("thumbnail.png", "image/png"),
            ("items.parquet", "application/vnd.apache.parquet"),
            ("lst_p95_2021-2025_N40W075.tif", COG_MEDIA_TYPE),
            ("legacy.TIFF", COG_MEDIA_TYPE),
        ],
    )
    def test_known_extensions(self, name: str, expected: str) -> None:
        assert content_type_for(name) == expected

    def test_unknown_extension_falls_back(self) -> None:
        assert content_type_for("notes.txt") == DEFAULT_CONTENT_TYPE

    def test_cog_type_matches_the_catalog_declaration(self) -> None:
        """A COG served under a different string is not recognised as a COG."""
        assert CONTENT_TYPES[".tif"] == COG_MEDIA_TYPE

    def test_every_file_in_a_built_catalog_is_recognised(self, built_catalog: Path) -> None:
        unknown = [
            path.name
            for path in built_catalog.rglob("*")
            if path.is_file() and content_type_for(path) == DEFAULT_CONTENT_TYPE
        ]
        assert unknown == []


class TestParseS3Uri:
    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("s3://bucket/a/b/", ("bucket", "a/b/")),
            ("s3://bucket/a/b", ("bucket", "a/b/")),
            ("s3://bucket", ("bucket", "")),
            ("s3://bucket/", ("bucket", "")),
        ],
    )
    def test_normalises_the_prefix(self, uri: str, expected: tuple[str, str]) -> None:
        assert parse_s3_uri(uri) == expected

    @pytest.mark.parametrize("uri", ["https://example.com/x", "/local/path", "s3:///nobucket"])
    def test_rejects_anything_that_is_not_an_s3_uri(self, uri: str) -> None:
        with pytest.raises(ValueError, match="s3://bucket/prefix"):
            parse_s3_uri(uri)


class TestPlan:
    def test_first_publish_plans_every_file(self, built_catalog: Path) -> None:
        planned, skipped = plan_uploads(built_catalog, PREFIX, {})
        expected = sorted(
            path.relative_to(built_catalog).as_posix()
            for path in built_catalog.rglob("*")
            if path.is_file()
        )
        assert [upload.key for upload in planned] == [f"{PREFIX}{name}" for name in expected]
        assert skipped == []

    def test_keys_preserve_the_tree_shape(self, built_catalog: Path) -> None:
        planned, _ = plan_uploads(built_catalog, PREFIX, {})
        keys = {upload.key for upload in planned}
        collection = DEFAULT_SPEC.collection_id
        assert f"{PREFIX}catalog.json" in keys
        assert f"{PREFIX}{collection}/collection.json" in keys
        assert f"{PREFIX}{collection}/N40W075/N40W075.json" in keys
        assert f"{PREFIX}{collection}/N40W075/lst_p95_2021-2025_N40W075.tif" in keys

    def test_plan_is_sorted_and_stable(self, built_catalog: Path) -> None:
        first, _ = plan_uploads(built_catalog, PREFIX, {})
        second, _ = plan_uploads(built_catalog, PREFIX, {})
        keys = [upload.key for upload in first]
        assert keys == sorted(keys)
        assert keys == [upload.key for upload in second]

    def test_sizes_and_types_come_from_the_files(self, built_catalog: Path) -> None:
        planned, _ = plan_uploads(built_catalog, PREFIX, {})
        for upload in planned:
            assert upload.size == upload.local.stat().st_size
            assert upload.content_type == content_type_for(upload.local)

    def test_bucket_root_prefix_yields_bare_keys(self, built_catalog: Path) -> None:
        planned, _ = plan_uploads(built_catalog, "", {})
        assert "catalog.json" in {upload.key for upload in planned}

    def test_dotfiles_are_not_published(self, tmp_path: Path) -> None:
        """Build scratch such as a partial COG must not reach the bucket."""
        root = tmp_path / "tree"
        (root / ".scratch").mkdir(parents=True)
        (root / ".scratch" / "partial.tif").write_bytes(b"x")
        (root / ".hidden.json").write_text("{}")
        (root / "catalog.json").write_text("{}")
        planned, skipped = plan_uploads(root, PREFIX, {})
        assert [upload.key for upload in planned] == [f"{PREFIX}catalog.json"]
        assert skipped == []


class TestSkipUnchanged:
    def test_matching_size_skips_every_asset(self, built_catalog: Path) -> None:
        planned, skipped = plan_uploads(built_catalog, PREFIX, remote_state(built_catalog))
        assert {upload.key for upload in skipped} == {
            f"{PREFIX}{path.relative_to(built_catalog).as_posix()}"
            for path in built_catalog.rglob("*")
            if path.is_file() and path.suffix.lower() in {".tif", ".png", ".parquet"}
        }
        assert not any(upload.key.endswith(".tif") for upload in planned)

    def test_differing_size_replans_a_cog(self, built_catalog: Path) -> None:
        state = remote_state(built_catalog)
        stale = next(key for key in state if key.endswith("lst_p95_2021-2025_N40W075.tif"))
        state[stale] = state[stale] + 1
        planned, skipped = plan_uploads(built_catalog, PREFIX, state)
        assert stale in {upload.key for upload in planned}
        assert stale not in {upload.key for upload in skipped}

    def test_absent_object_replans_a_cog(self, built_catalog: Path) -> None:
        state = remote_state(built_catalog)
        missing = next(key for key in state if key.endswith("qa_count_2021-2025_N35W120.tif"))
        del state[missing]
        planned, _ = plan_uploads(built_catalog, PREFIX, state)
        assert missing in {upload.key for upload in planned}

    def test_metadata_is_always_resent(self, built_catalog: Path) -> None:
        """Size is not a sound freshness test for text: an edit can preserve it."""
        planned, skipped = plan_uploads(built_catalog, PREFIX, remote_state(built_catalog))
        resent = {upload.key for upload in planned}
        assert f"{PREFIX}catalog.json" in resent
        assert f"{PREFIX}README.md" in resent
        assert not any(upload.key.endswith((".json", ".md")) for upload in skipped)

    def test_a_thumbnail_of_matching_size_is_skipped(self, built_catalog: Path) -> None:
        planned, skipped = plan_uploads(built_catalog, PREFIX, remote_state(built_catalog))
        assert any(upload.key.endswith("thumbnail.png") for upload in skipped)
        assert not any(upload.key.endswith("thumbnail.png") for upload in planned)

    def test_a_key_under_another_prefix_does_not_count(self, built_catalog: Path) -> None:
        """Freshness is per key, so republishing elsewhere is a full publish."""
        planned, skipped = plan_uploads(
            built_catalog, PREFIX, remote_state(built_catalog, "other/prefix/")
        )
        assert skipped == []
        assert planned


class TestPublishCatalog:
    def test_dry_run_uploads_nothing(self, built_catalog: Path) -> None:
        client = stub_client()
        summary = publish_catalog(built_catalog, REMOTE, dry_run=True, client=client)
        client.upload_file.assert_not_called()
        assert summary.dry_run
        assert summary.uploaded_count == 0
        assert summary.uploaded_bytes == 0
        assert summary.planned
        assert summary.planned_bytes == sum(
            upload.local.stat().st_size for upload in summary.planned
        )

    def test_dry_run_still_lists_the_remote(self, built_catalog: Path) -> None:
        """The plan has to reflect what is really there, or it is not a plan."""
        client = stub_client(remote_state(built_catalog))
        summary = publish_catalog(built_catalog, REMOTE, dry_run=True, client=client)
        client.get_paginator.return_value.paginate.assert_called_once_with(
            Bucket="us-west-2.opendata.source.coop", Prefix=PREFIX
        )
        assert summary.skipped_count > 0
        assert not any(upload.key.endswith(".tif") for upload in summary.planned)

    def test_upload_carries_the_content_type(self, built_catalog: Path) -> None:
        client = stub_client()
        publish_catalog(built_catalog, REMOTE, client=client)
        sent = {
            call.args[2]: call.kwargs["ExtraArgs"]["ContentType"]
            for call in client.upload_file.call_args_list
        }
        assert sent[f"{PREFIX}catalog.json"] == "application/json"
        assert sent[f"{PREFIX}README.md"] == "text/markdown"
        collection = DEFAULT_SPEC.collection_id
        cog = f"{PREFIX}{collection}/N40W075/lst_p95_2021-2025_N40W075.tif"
        assert sent[cog] == COG_MEDIA_TYPE

    def test_upload_targets_the_parsed_bucket(self, built_catalog: Path) -> None:
        client = stub_client()
        publish_catalog(built_catalog, REMOTE, client=client)
        buckets = {call.args[1] for call in client.upload_file.call_args_list}
        assert buckets == {"us-west-2.opendata.source.coop"}

    def test_summary_counts_what_moved(self, built_catalog: Path) -> None:
        client = stub_client()
        summary = publish_catalog(built_catalog, REMOTE, client=client)
        assert summary.uploaded_count == client.upload_file.call_count
        assert summary.skipped_count == 0
        assert summary.uploaded_bytes == sum(u.local.stat().st_size for u in summary.uploaded)
        assert not summary.dry_run

    def test_republish_sends_only_metadata(self, built_catalog: Path) -> None:
        client = stub_client(remote_state(built_catalog))
        summary = publish_catalog(built_catalog, REMOTE, client=client)
        sent = [call.args[2] for call in client.upload_file.call_args_list]
        assert sent
        assert all(key.endswith((".json", ".md")) for key in sent)
        assert summary.uploaded_count == len(sent)
        assert summary.skipped_count > 0

    def test_rejects_a_non_s3_remote(self, built_catalog: Path) -> None:
        with pytest.raises(ValueError, match="s3://bucket/prefix"):
            publish_catalog(built_catalog, "https://example.com/x", client=stub_client())

    def test_rejects_a_root_that_is_not_a_directory(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            publish_catalog(tmp_path / "absent", REMOTE, client=stub_client())


class TestCli:
    """The commands, with the S3 and validator calls patched out."""

    def test_publish_dry_run_prints_the_plan(self, built_catalog: Path) -> None:
        from click.testing import CliRunner

        from landsat_lst.cli import main

        with patch("landsat_lst.catalog.publish._s3_client", return_value=stub_client()) as factory:
            result = CliRunner().invoke(
                main, ["catalog", "publish", str(built_catalog), "--remote", REMOTE, "--dry-run"]
            )
        assert result.exit_code == 0, result.output
        assert "Dry run" in result.output
        assert "catalog.json" in result.output
        factory.assert_called_once_with(None)

    def test_publish_passes_the_profile_through(self, built_catalog: Path) -> None:
        from click.testing import CliRunner

        from landsat_lst.cli import main

        client = stub_client()
        with patch("landsat_lst.catalog.publish._s3_client", return_value=client) as factory:
            result = CliRunner().invoke(
                main,
                [
                    "catalog",
                    "publish",
                    str(built_catalog),
                    "--remote",
                    REMOTE,
                    "--profile",
                    "source-coop",
                ],
            )
        assert result.exit_code == 0, result.output
        factory.assert_called_once_with("source-coop")
        assert client.upload_file.call_count > 0
        assert "Uploaded:" in result.output

    def test_validate_live_requires_a_base_url(self, built_catalog: Path) -> None:
        """Relative hrefs give the live pass nothing to probe on their own."""
        from click.testing import CliRunner

        from landsat_lst.cli import main

        with patch("landsat_lst.catalog.validation.validate") as validator:
            result = CliRunner().invoke(main, ["catalog", "validate", str(built_catalog), "--live"])
        assert result.exit_code != 0
        assert "--live-base-url" in result.output
        validator.assert_not_called()

    @pytest.mark.parametrize("flags", [["--live"], []])
    def test_validate_live_base_url_reaches_rashid(
        self, built_catalog: Path, flags: list[str]
    ) -> None:
        """A named base URL turns the pass on, with or without the bare flag."""
        from click.testing import CliRunner

        from landsat_lst.cli import main

        base = "https://data.source.coop/nlebovits/landsat-lst/"
        report = MagicMock(errors=[], warnings=[])
        report.by_rule.return_value = []
        with patch("landsat_lst.catalog.validation.validate", return_value=report) as validator:
            result = CliRunner().invoke(
                main,
                ["catalog", "validate", str(built_catalog), "--live-base-url", base, *flags],
            )
        assert result.exit_code == 0, result.output
        assert validator.call_args.kwargs["live"] is True
        assert validator.call_args.kwargs["live_base_url"] == base

    def test_validate_stays_offline_by_default(self, built_catalog: Path) -> None:
        from click.testing import CliRunner

        from landsat_lst.cli import main

        report = MagicMock(errors=[], warnings=[])
        report.by_rule.return_value = []
        with patch("landsat_lst.catalog.validation.validate", return_value=report) as validator:
            result = CliRunner().invoke(main, ["catalog", "validate", str(built_catalog)])
        assert result.exit_code == 0, result.output
        assert validator.call_args.kwargs["live"] is False
        assert validator.call_args.kwargs["live_base_url"] is None
