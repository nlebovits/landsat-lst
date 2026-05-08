"""Unit tests for storage abstraction module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from landsat_lst.storage import LocalStorage, S3Storage, get_storage


class TestLocalStorage:
    """Tests for local filesystem storage backend."""

    def test_cog_path_creates_directory(self, tmp_path):
        """Should create parent directory for COG path."""
        storage = LocalStorage(output_dir=tmp_path)

        path = storage.cog_path(2023, "N40W075")

        assert path == str(tmp_path / "2023" / "N40W075.tif")
        assert (tmp_path / "2023").exists()

    def test_cog_exists_true(self, tmp_path):
        """Should return True when COG exists."""
        storage = LocalStorage(output_dir=tmp_path)
        cog_path = tmp_path / "2023" / "N40W075.tif"
        cog_path.parent.mkdir(parents=True)
        cog_path.touch()

        assert storage.cog_exists(2023, "N40W075") is True

    def test_cog_exists_false(self, tmp_path):
        """Should return False when COG doesn't exist."""
        storage = LocalStorage(output_dir=tmp_path)

        assert storage.cog_exists(2023, "N40W075") is False

    def test_icechunk_storage_creates_directory(self, tmp_path):
        """Should create Icechunk directory."""
        with patch("landsat_lst.storage.settings") as mock_settings:
            mock_settings.icechunk_prefix = "icechunk"
            storage = LocalStorage(output_dir=tmp_path)

            storage.icechunk_storage()

            assert (tmp_path / "icechunk").exists()


class TestS3Storage:
    """Tests for S3 storage backend."""

    def test_cog_path_returns_s3_url(self):
        """Should return S3 URL for COG path."""
        storage = S3Storage(
            bucket="test-bucket",
            prefix="landsat-lst",
            region="us-west-2",
        )

        path = storage.cog_path(2023, "N40W075")

        assert path == "s3://test-bucket/landsat-lst/2023/N40W075.tif"

    def test_cog_key_format(self):
        """Should format COG key correctly."""
        storage = S3Storage(bucket="b", prefix="p", region="r")

        key = storage._cog_key(2023, "N40W075")

        assert key == "p/2023/N40W075.tif"

    def test_cog_exists_true(self):
        """Should return True when head_object succeeds."""
        storage = S3Storage(bucket="b", prefix="p", region="r")
        mock_client = MagicMock()
        mock_client.head_object.return_value = {}
        storage._client = mock_client

        assert storage.cog_exists(2023, "N40W075") is True

    def test_cog_exists_false_on_404(self):
        """Should return False on 404 error."""
        from botocore.exceptions import ClientError

        storage = S3Storage(bucket="b", prefix="p", region="r")
        mock_client = MagicMock()
        mock_client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
        storage._client = mock_client

        assert storage.cog_exists(2023, "N40W075") is False

    def test_cog_exists_raises_other_errors(self):
        """Should raise non-404 errors."""
        from botocore.exceptions import ClientError

        storage = S3Storage(bucket="b", prefix="p", region="r")
        mock_client = MagicMock()
        mock_client.head_object.side_effect = ClientError({"Error": {"Code": "403"}}, "HeadObject")
        storage._client = mock_client

        with pytest.raises(ClientError):
            storage.cog_exists(2023, "N40W075")


class TestGetStorage:
    """Tests for storage factory function."""

    def test_returns_local_by_default(self):
        """Should return LocalStorage when backend is 'local'."""
        with patch("landsat_lst.storage.settings") as mock_settings:
            mock_settings.storage_backend = "local"
            mock_settings.output_dir = Path("/tmp/test")

            storage = get_storage()

            assert isinstance(storage, LocalStorage)

    def test_returns_s3_when_configured(self):
        """Should return S3Storage when backend is 's3'."""
        with patch("landsat_lst.storage.settings") as mock_settings:
            mock_settings.storage_backend = "s3"
            mock_settings.s3_bucket = "test-bucket"
            mock_settings.s3_prefix = "test-prefix"
            mock_settings.s3_region = "us-west-2"

            storage = get_storage()

            assert isinstance(storage, S3Storage)
