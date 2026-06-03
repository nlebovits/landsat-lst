"""Pydantic models for the LST pipeline."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, computed_field


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
    """A single processing job for one tile and one year."""

    tile: TileId
    year: int = Field(ge=2013, le=2030)

    @computed_field
    @property
    def datetime_range(self) -> str:
        """ISO datetime range for STAC query."""
        return f"{self.year}-01-01/{self.year}-12-31"

    @computed_field
    @property
    def output_filename(self) -> str:
        """Output COG filename."""
        return f"lst_{self.year}_{self.tile.name}.tif"


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
