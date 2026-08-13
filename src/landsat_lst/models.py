"""Pydantic models for the LST pipeline."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator


class TileId(BaseModel):
    """Identifier for a 5-degree tile."""

    lat: int = Field(ge=-55, le=60, description="Northwest corner latitude")
    lon: int = Field(ge=-180, le=175, description="Northwest corner longitude")

    @computed_field
    @property
    def name(self) -> str:
        """Tile name in format N40W075 or S10E030."""
        lat_prefix = "N" if self.lat >= 0 else "S"
        lon_prefix = "E" if self.lon >= 0 else "W"
        return f"{lat_prefix}{abs(self.lat):02d}{lon_prefix}{abs(self.lon):03d}"

    @computed_field
    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box as (west, south, east, north)."""
        return (
            float(self.lon),
            float(self.lat - 5),
            float(self.lon + 5),
            float(self.lat),
        )


class YearRange(BaseModel):
    """A range of years to process."""

    start: int = Field(ge=2013, le=2030, description="Start year (inclusive)")
    end: int = Field(ge=2013, le=2030, description="End year (inclusive)")

    def __iter__(self):
        return iter(range(self.start, self.end + 1))

    def __contains__(self, year: int) -> bool:
        return self.start <= year <= self.end


class ProcessingJob(BaseModel):
    """A single processing job for one tile and a one- or multi-year window.

    ``year`` is the first (inclusive) year of the window. For a multi-year
    composite, set ``end_year`` to the last (inclusive) year; the P95 is then
    pooled across every scene in ``[year, end_year]``. When ``end_year`` is
    ``None`` the job is a single-year composite (backward compatible).
    """

    tile: TileId
    year: int = Field(ge=2013, le=2030)
    end_year: int | None = Field(
        default=None,
        ge=2013,
        le=2030,
        description="Last inclusive year for a multi-year window; None = single year",
    )
    max_scenes: int | None = Field(
        default=None,
        gt=0,
        description="Keep at most this many scenes, sampled evenly across the "
        "window. For exercising the machinery at tile geometry in minutes "
        "instead of hours; the composite it produces is not the product. The "
        "sample is stamped into window_label so it can never be written over a "
        "real tile.",
    )

    @model_validator(mode="after")
    def _check_year_window(self) -> "ProcessingJob":
        if self.end_year is not None and self.end_year < self.year:
            msg = f"end_year ({self.end_year}) must be >= year ({self.year})"
            raise ValueError(msg)
        return self

    @computed_field
    @property
    def datetime_range(self) -> str:
        """ISO datetime range for STAC query (spans the full window)."""
        last = self.end_year or self.year
        return f"{self.year}-01-01/{last}-12-31"

    @computed_field
    @property
    def window_label(self) -> str:
        """Storage/label token: ``2024`` for single year, ``2020-2024`` for a range.

        A sampled job carries ``-sample{n}``. Every storage key is built from
        this token, so a throwaway run cannot land on the keys a real tile owns,
        and ``list_completed`` for the real window never counts one as done.
        """
        window = (
            str(self.year)
            if self.end_year is None or self.end_year == self.year
            else f"{self.year}-{self.end_year}"
        )
        return window if self.max_scenes is None else f"{window}-sample{self.max_scenes}"

    def asset_filename(self, product: str) -> str:
        """Filename for one output asset of this job.

        Args:
            product: Asset name, ``"lst_p95"`` or ``"qa_count"``.

        Returns:
            ``{product}_{window_label}_{tile}.tif``
        """
        return f"{product}_{self.window_label}_{self.tile.name}.tif"


class CompositeStats(BaseModel):
    """Statistics for a computed composite."""

    tile: TileId
    year: int
    valid_pixel_count: int
    total_pixel_count: int
    min_observations: int
    max_observations: int
    mean_observations: float
    lst_p95_min: float | None = None
    lst_p95_max: float | None = None


class LandsatScene(BaseModel):
    """Metadata for a Landsat scene."""

    scene_id: str
    datetime: date
    cloud_cover: float = Field(ge=0, le=100)
    platform: Literal["landsat-8", "landsat-9"]
    path: int
    row: int

    @computed_field
    @property
    def is_daytime(self) -> bool:
        """Check if this is a daytime scene (descending orbit)."""
        return True
