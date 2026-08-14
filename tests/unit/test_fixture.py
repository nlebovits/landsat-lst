"""Unit tests for the cached-fixture store.

The size arithmetic is the part that matters most here. A fixture at
production's offset factor is 97 GB, and a command that only discovers this
after an hour of downloading is worse than no command.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from landsat_lst.fixture import (
    BANDS,
    DEFAULT_MAX_GB,
    FixtureMeta,
    FixtureSpec,
    build_fixture,
    estimate_bytes,
    grid_shape,
    list_fixtures,
    load_fixture,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def fixture_root(tmp_path, monkeypatch):
    """Point the store at a temp directory for the duration of a test."""
    monkeypatch.setattr("landsat_lst.fixture.fixture_root", lambda: tmp_path)
    return tmp_path


class TestFixtureSpec:
    def test_name_carries_every_field_that_changes_the_pixels(self):
        spec = FixtureSpec(tile="N40W075", year=2021, end_year=2025, max_scenes=300, factor=8)
        assert spec.name == "N40W075_2021-2025_n300_f8"

    def test_a_different_factor_is_a_different_fixture(self):
        base = {"tile": "N40W075", "max_scenes": 300}
        assert FixtureSpec(**base, factor=2).name != FixtureSpec(**base, factor=8).name

    def test_single_year_window_reads_as_one_year(self):
        assert FixtureSpec(tile="N40W075", year=2024, end_year=None).window == "2024"

    def test_unsampled_window_says_so(self):
        assert "_nall_" in FixtureSpec(tile="N40W075", max_scenes=None).name

    def test_exists_is_false_when_a_band_is_missing(self, fixture_root):
        spec = FixtureSpec(tile="N40W075")
        spec.path.mkdir(parents=True)
        spec.meta_path.write_text("{}")
        assert not spec.exists()


class TestGridArithmetic:
    def test_native_grid_is_the_shared_five_degree_tile(self):
        """18,000 squared at 3,600 px/degree. Never 17,999: see ADR-008."""
        assert grid_shape(FixtureSpec(tile="N40W075", factor=1)) == (18_000, 18_000)

    @pytest.mark.parametrize(("factor", "side"), [(2, 9_000), (4, 4_500), (8, 2_250)])
    def test_each_doubling_halves_the_side(self, factor, side):
        assert grid_shape(FixtureSpec(tile="N40W075", factor=factor)) == (side, side)

    def test_production_offset_factor_is_not_a_laptop_fixture(self):
        """The number the CLI exists to show before anybody starts a download."""
        planned = estimate_bytes(FixtureSpec(tile="N40W075", max_scenes=300, factor=2))
        assert planned / 1e9 == pytest.approx(97.2, rel=0.01)
        assert planned / 1e9 > DEFAULT_MAX_GB

    def test_coarsening_divides_the_stack_by_four(self):
        at_4 = estimate_bytes(FixtureSpec(tile="N40W075", max_scenes=300, factor=4))
        at_8 = estimate_bytes(FixtureSpec(tile="N40W075", max_scenes=300, factor=8))
        assert at_4 / at_8 == pytest.approx(4.0)

    def test_estimate_accepts_a_measured_scene_count(self):
        spec = FixtureSpec(tile="N40W075", max_scenes=300, factor=16)
        assert estimate_bytes(spec, scenes=10) * 30 == estimate_bytes(spec, scenes=300)


class TestBuildGuard:
    def test_an_oversized_fetch_is_refused_before_any_query(self, fixture_root, monkeypatch):
        """The refusal must not cost a STAC round trip to discover."""

        def _boom(*args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("query_stac was called despite the size guard")

        monkeypatch.setattr("landsat_lst.pipeline.query_stac", _boom)
        spec = FixtureSpec(tile="N40W075", max_scenes=300, factor=2)

        with pytest.raises(ValueError, match="past the"):
            build_fixture(spec, max_gb=DEFAULT_MAX_GB)

    def test_the_refusal_names_both_ways_out(self, fixture_root):
        spec = FixtureSpec(tile="N40W075", max_scenes=300, factor=2)
        with pytest.raises(ValueError) as excinfo:
            build_fixture(spec, max_gb=1.0)

        message = str(excinfo.value)
        assert "--factor" in message
        assert "--max-gb" in message
        assert "97.2 GB" in message


def _write_fixture(spec: FixtureSpec, *, scenes: int = 4, side: int = 8) -> None:
    """Put a fixture on disk without going near the network."""
    spec.path.mkdir(parents=True, exist_ok=True)
    for i, band in enumerate(BANDS):
        np.save(spec.band_path(band), np.full((scenes, side, side), i + 1, dtype=np.uint16))
    meta = FixtureMeta(
        spec={
            "tile": spec.tile,
            "year": spec.year,
            "end_year": spec.end_year,
            "max_scenes": spec.max_scenes,
            "factor": spec.factor,
        },
        shape=(scenes, side, side),
        times=[f"2021-0{i + 1}-01T00:00:00" for i in range(scenes)],
        latitude=[40.0, 35.0],
        longitude=[-75.0, -70.0],
        stac_url="https://example.invalid/stac",
        scene_count=scenes,
        bytes_on_disk=sum(spec.band_path(b).stat().st_size for b in BANDS),
    )
    spec.meta_path.write_text(json.dumps(meta.__dict__))


class TestLoad:
    def test_missing_fixture_says_how_to_build_it(self, fixture_root):
        with pytest.raises(FileNotFoundError, match="landsat-lst fixture"):
            load_fixture(FixtureSpec(tile="N40W075"))

    def test_round_trip_returns_the_bands_load_scenes_would_have(self, fixture_root):
        spec = FixtureSpec(tile="N40W075", max_scenes=4, factor=64)
        _write_fixture(spec)

        ds = load_fixture(spec)

        assert set(ds.data_vars) == set(BANDS)
        assert ds["lwir11"].dims == ("time", "latitude", "longitude")
        assert ds["lwir11"].dtype == np.uint16
        assert ds.sizes == {"time": 4, "latitude": 8, "longitude": 8}

    def test_load_stays_lazy(self, fixture_root):
        """Materializing here would make every downstream measurement a lie."""
        spec = FixtureSpec(tile="N40W075", max_scenes=4, factor=64)
        _write_fixture(spec)

        ds = load_fixture(spec)

        assert ds["lwir11"].chunks is not None

    def test_latitude_descends_matching_the_north_down_grid(self, fixture_root):
        spec = FixtureSpec(tile="N40W075", max_scenes=4, factor=64)
        _write_fixture(spec)

        lat = load_fixture(spec)["latitude"].values

        assert lat[0] > lat[-1]

    def test_values_survive_the_round_trip(self, fixture_root):
        spec = FixtureSpec(tile="N40W075", max_scenes=4, factor=64)
        _write_fixture(spec)

        ds = load_fixture(spec)

        assert int(ds["lwir11"].max().compute()) == 1
        assert int(ds["qa_pixel"].max().compute()) == 2


class TestListing:
    def test_empty_root_lists_nothing(self, fixture_root):
        assert list_fixtures() == []

    def test_built_fixtures_are_listed(self, fixture_root):
        _write_fixture(FixtureSpec(tile="N40W075", max_scenes=4, factor=64))
        _write_fixture(FixtureSpec(tile="S30W065", max_scenes=4, factor=64))

        assert len(list_fixtures()) == 2

    def test_an_unreadable_fixture_is_skipped_rather_than_fatal(self, fixture_root):
        _write_fixture(FixtureSpec(tile="N40W075", max_scenes=4, factor=64))
        broken = fixture_root / "broken"
        broken.mkdir()
        (broken / "meta.json").write_text("{not json")

        assert len(list_fixtures()) == 1
