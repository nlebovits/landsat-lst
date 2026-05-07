"""Shared test fixtures."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from landsat_lst.models import ProcessingJob, TileId


@pytest.fixture
def tiny_bbox() -> tuple[float, float, float, float]:
    """A tiny 0.25 x 0.25 degree bounding box for smoke tests.

    Located near Buenos Aires, Argentina - good Landsat coverage.
    """
    return (-58.5, -34.75, -58.25, -34.5)


@pytest.fixture
def pergamino_bbox() -> tuple[float, float, float, float]:
    """Bounding box for Pergamino municipality, Argentina.

    Used for integration tests - small city with good coverage.
    """
    return (-60.75, -34.25, -60.25, -33.75)


@pytest.fixture
def sample_tile() -> TileId:
    """A sample tile for testing."""
    return TileId(lat=-30, lon=-60)


@pytest.fixture
def sample_job(sample_tile: TileId) -> ProcessingJob:
    """A sample processing job for testing."""
    return ProcessingJob(tile=sample_tile, year=2023)


@pytest.fixture
def mock_qa_pixel() -> xr.DataArray:
    """Mock QA pixel band for testing masking functions.

    Creates a 10x10 array with various QA conditions:
    - Clear pixels (value 0)
    - Cloud pixels (bit 3 set = 8)
    - Shadow pixels (bit 4 set = 16)
    - Snow pixels (bit 5 set = 32)
    """
    data = np.zeros((10, 10), dtype=np.uint16)

    data[0:2, :] = 8
    data[2:4, :] = 16
    data[4:5, :] = 32
    data[5:, :] = 0

    return xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": np.arange(10), "x": np.arange(10)},
    )


@pytest.fixture
def mock_lwir_band() -> xr.DataArray:
    """Mock LWIR11 thermal band for testing temperature conversion.

    Values are in raw DN that should convert to ~20-40°C range.
    """
    data = np.full((10, 10), 40000, dtype=np.float32)

    return xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": np.arange(10), "x": np.arange(10)},
    )


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def data_cache_dir(fixtures_dir: Path) -> Path:
    """Path to cached test data directory."""
    cache_dir = fixtures_dir / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
