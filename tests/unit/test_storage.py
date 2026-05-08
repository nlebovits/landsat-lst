"""Unit tests for storage abstraction module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from landsat_lst.storage import IcechunkStorage, LocalStorage, S3Storage, get_storage


class TestLocalStorage:
    """Tests for local filesystem storage backend."""

    def test_zarr_path_creates_directory(self, tmp_path):
        """Should create parent directory for Zarr path."""
        storage = LocalStorage(output_dir=tmp_path)

        path = storage.zarr_path(2023, "N40W075")

        assert path == str(tmp_path / "2023" / "N40W075.zarr")
        assert (tmp_path / "2023").exists()

    def test_zarr_exists_true(self, tmp_path):
        """Should return True when Zarr exists."""
        storage = LocalStorage(output_dir=tmp_path)
        zarr_path = tmp_path / "2023" / "N40W075.zarr"
        zarr_path.parent.mkdir(parents=True)
        zarr_path.mkdir()  # Zarr stores are directories

        assert storage.zarr_exists(2023, "N40W075") is True

    def test_zarr_exists_false(self, tmp_path):
        """Should return False when Zarr doesn't exist."""
        storage = LocalStorage(output_dir=tmp_path)

        assert storage.zarr_exists(2023, "N40W075") is False


class TestS3Storage:
    """Tests for S3 storage backend."""

    def test_zarr_path_returns_s3_url(self):
        """Should return S3 URL for Zarr path."""
        storage = S3Storage(
            bucket="test-bucket",
            prefix="landsat-lst",
            region="us-west-2",
        )

        path = storage.zarr_path(2023, "N40W075")

        assert path == "s3://test-bucket/landsat-lst/2023/N40W075.zarr"

    def test_zarr_key_format(self):
        """Should format Zarr key correctly."""
        storage = S3Storage(bucket="b", prefix="p", region="r")

        key = storage._zarr_key(2023, "N40W075")

        assert key == "p/2023/N40W075.zarr"

    def test_zarr_exists_true(self):
        """Should return True when head_object succeeds for .zmetadata."""
        storage = S3Storage(bucket="b", prefix="p", region="r")
        mock_client = MagicMock()
        mock_client.head_object.return_value = {}
        storage._client = mock_client

        assert storage.zarr_exists(2023, "N40W075") is True
        mock_client.head_object.assert_called_once_with(
            Bucket="b",
            Key="p/2023/N40W075.zarr/.zmetadata",
        )

    def test_zarr_exists_false_on_404(self):
        """Should return False on 404 error for both .zmetadata and zarr.json."""
        from botocore.exceptions import ClientError

        storage = S3Storage(bucket="b", prefix="p", region="r")
        mock_client = MagicMock()
        mock_client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
        storage._client = mock_client

        assert storage.zarr_exists(2023, "N40W075") is False

    def test_zarr_exists_raises_other_errors(self):
        """Should raise non-404 errors."""
        from botocore.exceptions import ClientError

        storage = S3Storage(bucket="b", prefix="p", region="r")
        mock_client = MagicMock()
        mock_client.head_object.side_effect = ClientError({"Error": {"Code": "403"}}, "HeadObject")
        storage._client = mock_client

        with pytest.raises(ClientError):
            storage.zarr_exists(2023, "N40W075")


class TestIcechunkStorage:
    """Tests for Icechunk storage backend."""

    def test_from_local_creates_repo(self, tmp_path):
        """Should create repository from local path."""
        storage = IcechunkStorage.from_local(tmp_path / "icechunk")

        assert storage.repo is not None
        assert (tmp_path / "icechunk").exists()

    def test_zarr_path_returns_group_format(self, tmp_path):
        """Should return group path format (not filesystem path)."""
        storage = IcechunkStorage.from_local(tmp_path / "icechunk")

        path = storage.zarr_path(2023, "N40W075")

        assert path == "2023/N40W075"

    def test_zarr_exists_false_initially(self, tmp_path):
        """Should return False for non-existent group."""
        storage = IcechunkStorage.from_local(tmp_path / "icechunk")

        assert storage.zarr_exists(2023, "N40W075") is False

    def test_writable_session_returns_session(self, tmp_path):
        """Should return writable session with store."""
        storage = IcechunkStorage.from_local(tmp_path / "icechunk")

        session = storage.writable_session()

        assert hasattr(session, "store")
        assert hasattr(session, "commit")

    def test_readonly_session_returns_session(self, tmp_path):
        """Should return readonly session."""
        storage = IcechunkStorage.from_local(tmp_path / "icechunk")

        session = storage.readonly_session()

        assert hasattr(session, "store")


class TestGetStorage:
    """Tests for storage factory function."""

    def test_returns_local_by_default(self):
        """Should return LocalStorage when backend is 'local' and icechunk disabled."""
        with patch("landsat_lst.storage.settings") as mock_settings:
            mock_settings.use_icechunk = False
            mock_settings.storage_backend = "local"
            mock_settings.output_dir = Path("/tmp/test")

            storage = get_storage()

            assert isinstance(storage, LocalStorage)

    def test_returns_s3_when_configured(self):
        """Should return S3Storage when backend is 's3' and icechunk disabled."""
        with patch("landsat_lst.storage.settings") as mock_settings:
            mock_settings.use_icechunk = False
            mock_settings.storage_backend = "s3"
            mock_settings.s3_bucket = "test-bucket"
            mock_settings.s3_prefix = "test-prefix"
            mock_settings.s3_region = "us-west-2"

            storage = get_storage()

            assert isinstance(storage, S3Storage)

    def test_returns_icechunk_local_when_enabled(self, tmp_path):
        """Should return IcechunkStorage for local when use_icechunk=True."""
        with patch("landsat_lst.storage.settings") as mock_settings:
            mock_settings.use_icechunk = True
            mock_settings.storage_backend = "local"
            mock_settings.output_dir = tmp_path
            mock_settings.icechunk_prefix = "icechunk"

            storage = get_storage()

            assert isinstance(storage, IcechunkStorage)
