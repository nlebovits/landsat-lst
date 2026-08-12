"""Unit tests for the COG storage abstraction."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from landsat_lst.storage import (
    PRODUCTS,
    LocalStorage,
    S3Storage,
    collection_prefix,
    get_storage,
)


def _write_asset(root: Path, window: str, tile: str, product: str) -> Path:
    """Create one asset file under ``root`` at the canonical key."""
    path = root / collection_prefix(window) / tile / f"{product}_{window}_{tile}.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"tif")
    return path


class TestCogKey:
    """The key layout is shared by every backend."""

    def test_key_format(self, tmp_path):
        storage = LocalStorage(output_dir=tmp_path)

        assert storage.cog_key("2021-2025", "N40W075", "lst_p95") == (
            "lst-p95-2021-2025/N40W075/lst_p95_2021-2025_N40W075.tif"
        )

    def test_backends_agree_on_layout(self, tmp_path):
        """A catalog built against one backend must resolve against the other."""
        local = LocalStorage(output_dir=tmp_path)
        s3 = S3Storage(bucket="b", prefix="p", region="r")

        assert local.cog_key("2024", "N40W075", "qa_count") == s3.cog_key(
            "2024", "N40W075", "qa_count"
        )


class TestLocalStorage:
    """Tests for local filesystem storage backend."""

    def test_cog_exists_true_when_both_assets_present(self, tmp_path):
        storage = LocalStorage(output_dir=tmp_path)
        for product in PRODUCTS:
            _write_asset(tmp_path, "2024", "N40W075", product)

        assert storage.cog_exists("2024", "N40W075") is True

    def test_cog_exists_false_when_only_one_asset(self, tmp_path):
        """A half-uploaded tile is not done -- it has to be rebuilt."""
        storage = LocalStorage(output_dir=tmp_path)
        _write_asset(tmp_path, "2024", "N40W075", "lst_p95")

        assert storage.cog_exists("2024", "N40W075") is False

    def test_cog_exists_false_when_nothing_written(self, tmp_path):
        storage = LocalStorage(output_dir=tmp_path)

        assert storage.cog_exists("2024", "N40W075") is False

    def test_upload_copies_to_key(self, tmp_path):
        storage = LocalStorage(output_dir=tmp_path / "out")
        src = tmp_path / "src.tif"
        src.write_bytes(b"payload")

        key = storage.cog_key("2024", "N40W075", "lst_p95")
        storage.upload(src, key)

        assert (tmp_path / "out" / key).read_bytes() == b"payload"

    def test_list_completed_only_returns_whole_tiles(self, tmp_path):
        storage = LocalStorage(output_dir=tmp_path)
        for product in PRODUCTS:
            _write_asset(tmp_path, "2024", "N40W075", product)
        _write_asset(tmp_path, "2024", "N45W075", "lst_p95")  # half-written

        assert storage.list_completed("2024") == {"N40W075"}

    def test_list_completed_empty_for_unknown_window(self, tmp_path):
        storage = LocalStorage(output_dir=tmp_path)

        assert storage.list_completed("1999") == set()


class TestS3Storage:
    """Tests for S3 storage backend."""

    def test_upload_prefixes_key(self):
        storage = S3Storage(bucket="b", prefix="p", region="r")
        storage._client = MagicMock()

        storage.upload(Path("/tmp/src.tif"), "2024/N40W075/lst_p95_2024_N40W075.tif")

        storage._client.upload_file.assert_called_once_with(
            "/tmp/src.tif",
            "b",
            "p/2024/N40W075/lst_p95_2024_N40W075.tif",
        )

    def test_cog_exists_true_when_both_heads_succeed(self):
        storage = S3Storage(bucket="b", prefix="p", region="r")
        mock_client = MagicMock()
        mock_client.head_object.return_value = {}
        storage._client = mock_client

        assert storage.cog_exists("2024", "N40W075") is True
        assert mock_client.head_object.call_count == len(PRODUCTS)

    def test_cog_exists_false_when_only_one_asset(self):
        """Missing qa_count means the tile is incomplete, whatever lst_p95 says."""
        from botocore.exceptions import ClientError

        storage = S3Storage(bucket="b", prefix="p", region="r")
        mock_client = MagicMock()
        missing = ClientError({"Error": {"Code": "404"}}, "HeadObject")
        mock_client.head_object.side_effect = [{}, missing]
        storage._client = mock_client

        assert storage.cog_exists("2024", "N40W075") is False

    def test_cog_exists_raises_other_errors(self):
        """A 403 is a broken credential, not an absent object."""
        from botocore.exceptions import ClientError

        storage = S3Storage(bucket="b", prefix="p", region="r")
        mock_client = MagicMock()
        mock_client.head_object.side_effect = ClientError({"Error": {"Code": "403"}}, "HeadObject")
        storage._client = mock_client

        with pytest.raises(ClientError):
            storage.cog_exists("2024", "N40W075")

    def test_list_completed_pages_once_and_filters_partials(self):
        storage = S3Storage(bucket="b", prefix="p", region="r")
        mock_client = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "p/lst-p95-2024/N40W075/lst_p95_2024_N40W075.tif"},
                    {"Key": "p/lst-p95-2024/N40W075/qa_count_2024_N40W075.tif"},
                ]
            },
            {"Contents": [{"Key": "p/lst-p95-2024/N45W075/lst_p95_2024_N45W075.tif"}]},
        ]
        mock_client.get_paginator.return_value = paginator
        storage._client = mock_client

        assert storage.list_completed("2024") == {"N40W075"}
        paginator.paginate.assert_called_once_with(Bucket="b", Prefix="p/lst-p95-2024/")
        mock_client.head_object.assert_not_called()

    def test_list_completed_handles_empty_prefix(self):
        storage = S3Storage(bucket="b", prefix="p", region="r")
        mock_client = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{}]  # no Contents key at all
        mock_client.get_paginator.return_value = paginator
        storage._client = mock_client

        assert storage.list_completed("2024") == set()


class TestGetStorage:
    """Tests for storage factory function."""

    def test_returns_local_by_default(self):
        with patch("landsat_lst.storage.settings") as mock_settings:
            mock_settings.storage_backend = "local"
            mock_settings.output_dir = Path("/tmp/test")

            assert isinstance(get_storage(), LocalStorage)

    def test_returns_s3_when_configured(self):
        with patch("landsat_lst.storage.settings") as mock_settings:
            mock_settings.storage_backend = "s3"
            mock_settings.s3_bucket = "test-bucket"
            mock_settings.s3_prefix = "test-prefix"
            mock_settings.s3_region = "us-west-2"

            assert isinstance(get_storage(), S3Storage)


class TestCatalogLayoutContract:
    """The storage layout IS the published catalog layout.

    storage.collection_prefix duplicates catalog.spec's collection id rather
    than importing it, so workers do not pay for the catalog stack. This test
    is the contract that keeps the two from drifting: if either side changes,
    the published COG paths and the STAC item hrefs disagree and publication
    stops being a metadata-only sync.
    """

    def test_prefix_matches_the_collection_id(self):
        from landsat_lst.catalog.spec import spec_for_window
        from landsat_lst.storage import collection_prefix

        for window in ("2021-2025", "2024"):
            assert collection_prefix(window) == spec_for_window(window).collection_id

    def test_key_starts_with_the_collection_id(self, tmp_path):
        from landsat_lst.catalog.spec import spec_for_window

        storage = LocalStorage(output_dir=tmp_path)
        key = storage.cog_key("2021-2025", "N40W075", "lst_p95")
        assert key.startswith(spec_for_window("2021-2025").collection_id + "/")
