"""Unit tests for the per-scene offset cache (issue #77 item 2).

Two properties carry the feature and are tested here.

The key must change whenever the offsets would. A cache that serves a stale
answer to a changed configuration is worse than no cache, because the wrong
number arrives fast and looks like the right one. Every input in
:meth:`OffsetKey.build` gets a miss-on-change test.

A cache failure must never fail a tile. Read errors, write errors, malformed
records, and a record whose time axis does not line up all have to degrade to a
recompute, matching the rule the heartbeat already follows.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from landsat_lst.config import settings
from landsat_lst.offsets import (
    ALGORITHM_VERSION,
    OffsetCache,
    OffsetKey,
    cache_for_items,
)
from landsat_lst.storage import LocalStorage, offset_cache_key

SCENES = ("LC08_L2SP_014032_20210704", "LC09_L2SP_014032_20220711")


def _key(**overrides) -> OffsetKey:
    base = {
        "tile": "N40W075",
        "window": "2021-2025",
        "factor": 2,
        "scene_ids": SCENES,
    }
    return OffsetKey.build(**{**base, **overrides})


def _times(n: int = 2) -> np.ndarray:
    """A real datetime axis, sub-second components and all.

    Landsat solar-day stamps carry them. Serializing at second precision
    truncated the axis, which the coordinate join in
    ``normalization.debias_with_offsets`` then could not match -- and every
    fixture here used whole seconds, so nothing failed until real data hit it.
    """
    return pd.to_datetime(
        [
            f"2021-{i + 1:02d}-15T14:07:{i % 60:02d}.{(123_456 + 977 * i) % 1_000_000:06d}"
            for i in range(n)
        ]
    ).values


def _arrays(n: int = 2, *, offset=None, n_valid=None):
    """An ``(offset, n_valid)`` pair on a real datetime axis."""
    times = _times(n)
    coords = {"time": times}
    return (
        xr.DataArray(
            np.array(offset if offset is not None else [1.5, -2.0]),
            dims=["time"],
            coords=coords,
        ),
        xr.DataArray(
            np.array(n_valid if n_valid is not None else [900, 800]),
            dims=["time"],
            coords=coords,
        ),
    )


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(output_dir=tmp_path)


class TestKey:
    """What the digest must and must not distinguish."""

    def test_scene_order_does_not_change_the_key(self):
        """STAC pages in whatever order it likes; the same set must key alike."""
        assert _key(scene_ids=SCENES).digest == _key(scene_ids=tuple(reversed(SCENES))).digest

    def test_scene_set_changes_the_key(self):
        """--max-scenes changes the pooled climatology, so every offset moves."""
        assert _key().digest != _key(scene_ids=(*SCENES, "LC08_extra")).digest

    def test_factor_changes_the_key(self):
        """The factor decides the grid the median rests on."""
        assert _key(factor=2).digest != _key(factor=4).digest

    def test_algorithm_version_changes_the_key(self):
        """The escape hatch for code changes a hash cannot see."""
        assert _key().digest != _key(algorithm_version=ALGORITHM_VERSION + 1).digest

    def test_clamp_changes_the_key(self, monkeypatch):
        """convert_to_celsius applies the clamp before the median sees a pixel."""
        before = _key().digest
        monkeypatch.setattr(settings, "lst_valid_max", settings.lst_valid_max + 1)
        assert _key().digest != before

    def test_tile_and_window_change_the_key(self):
        assert _key().digest != _key(tile="N35W080").digest
        assert _key().digest != _key(window="2024").digest

    def test_storage_key_carries_its_terms_in_the_path(self):
        """A bucket listing should be readable without opening the objects."""
        assert _key().storage_key == offset_cache_key(
            tile="N40W075",
            window="2021-2025",
            factor=2,
            algorithm_version=ALGORITHM_VERSION,
            digest=_key().digest,
        )
        assert _key().storage_key.startswith("_offsets/N40W075/2021-2025/f2/")


class TestRoundTrip:
    """A write followed by a read returns what went in."""

    def test_values_survive(self, storage):
        cache = OffsetCache(storage=storage, key=_key())
        offset, n_valid = _arrays()
        cache.write(offset, n_valid, duration_s=1612.0)

        hit = cache.read(offset.time)
        assert hit is not None
        np.testing.assert_allclose(hit[0].values, offset.values)
        np.testing.assert_array_equal(hit[1].values, n_valid.values)
        assert cache.last_read_hit is True

    def test_rejected_scenes_round_trip_as_nan(self, storage):
        """A scene with no valid pixels has no offset, and that has to survive.

        NaN is stored as JSON ``null`` rather than the bare ``NaN`` token
        ``json.dumps`` emits by default, which no strict parser accepts.
        """
        cache = OffsetCache(storage=storage, key=_key())
        offset, n_valid = _arrays(offset=[np.nan, -2.0], n_valid=[0, 800])
        cache.write(offset, n_valid)

        raw = json.loads(storage.read_text(_key().storage_key))
        assert raw["offset"][0] is None

        hit = cache.read(offset.time)
        assert hit is not None
        assert np.isnan(hit[0].values[0])
        assert hit[0].values[1] == pytest.approx(-2.0)

    def test_hit_is_reattached_to_the_live_time_coordinate(self, storage):
        """The caller indexes the result against its own stack, not the record's."""
        cache = OffsetCache(storage=storage, key=_key())
        offset, n_valid = _arrays()
        cache.write(offset, n_valid)

        hit = cache.read(offset.time)
        assert hit is not None
        assert hit[0].time.equals(offset.time)


class TestMiss:
    """Every way a lookup declines to answer, none of which may raise."""

    def test_cold_cache_misses(self, storage):
        cache = OffsetCache(storage=storage, key=_key())
        offset, _ = _arrays()
        assert cache.read(offset.time) is None
        assert cache.last_read_hit is False

    def test_changed_input_misses(self, storage):
        """The point of the whole key: a changed factor cannot read the old record."""
        offset, n_valid = _arrays()
        OffsetCache(storage=storage, key=_key(factor=2)).write(offset, n_valid)
        assert OffsetCache(storage=storage, key=_key(factor=4)).read(offset.time) is None

    def test_disabled_cache_neither_reads_nor_writes(self, storage):
        """--no-offset-cache leaves a good record on disk untouched."""
        offset, n_valid = _arrays()
        OffsetCache(storage=storage, key=_key()).write(offset, n_valid)

        disabled = OffsetCache(storage=storage, key=_key(), enabled=False)
        assert disabled.read(offset.time) is None
        disabled.write(*_arrays(offset=[99.0, 99.0]))

        # The original record is still what it was.
        fresh = OffsetCache(storage=storage, key=_key()).read(offset.time)
        assert fresh is not None
        np.testing.assert_allclose(fresh[0].values, offset.values)

    def test_refresh_skips_the_read_but_still_writes(self, storage):
        """--force: rebuild the estimate and replace what was stored."""
        offset, n_valid = _arrays()
        OffsetCache(storage=storage, key=_key()).write(offset, n_valid)

        refreshing = OffsetCache(storage=storage, key=_key(), read=False)
        assert refreshing.read(offset.time) is None
        refreshing.write(*_arrays(offset=[7.0, 8.0]))

        after = OffsetCache(storage=storage, key=_key()).read(offset.time)
        assert after is not None
        np.testing.assert_allclose(after[0].values, [7.0, 8.0])

    def test_time_mismatch_misses(self, storage):
        """A record whose axis does not match the stack in hand is not usable.

        The digest should have prevented this, so it means the key is
        under-specified. Recomputing is right either way.
        """
        offset, n_valid = _arrays(n=2)
        OffsetCache(storage=storage, key=_key()).write(offset, n_valid)

        longer, _ = _arrays(n=3, offset=[1.0, 2.0, 3.0], n_valid=[1, 2, 3])
        assert OffsetCache(storage=storage, key=_key()).read(longer.time) is None

    def test_malformed_record_misses(self, storage):
        storage.write_text(_key().storage_key, "{not json")
        offset, _ = _arrays()
        assert OffsetCache(storage=storage, key=_key()).read(offset.time) is None

    def test_record_missing_a_field_misses(self, storage):
        storage.write_text(_key().storage_key, json.dumps({"times": []}))
        offset, _ = _arrays()
        assert OffsetCache(storage=storage, key=_key()).read(offset.time) is None


class TestNeverFailsTheTile:
    """A broken cache costs 27 minutes. A raised exception costs the run."""

    def test_read_error_is_swallowed(self, storage, monkeypatch):
        monkeypatch.setattr(
            storage, "read_text", lambda _key: (_ for _ in ()).throw(OSError("bucket on fire"))
        )
        offset, _ = _arrays()
        assert OffsetCache(storage=storage, key=_key()).read(offset.time) is None

    def test_write_error_is_swallowed(self, storage, monkeypatch):
        monkeypatch.setattr(
            storage,
            "write_text",
            lambda *_a, **_k: (_ for _ in ()).throw(OSError("no space left")),
        )
        # The assertion is that this returns rather than raising.
        OffsetCache(storage=storage, key=_key()).write(*_arrays())


class TestCacheForItems:
    """The constructor the pipeline actually calls."""

    def test_builds_the_key_from_item_ids(self, storage):
        items = [SimpleNamespace(id=scene) for scene in SCENES]
        cache = cache_for_items(
            tile="N40W075", window="2021-2025", items=items, factor=2, storage=storage
        )
        assert cache.key.digest == _key().digest

    def test_sampled_window_cannot_read_the_full_window(self, storage):
        """window_label carries -sampleN, so a sample keys separately.

        Belt and braces: the scene hash already differs, since a sample is a
        different scene set. The window token means the two are distinguishable
        in a listing even before anyone opens a record.
        """
        items = [SimpleNamespace(id=scene) for scene in SCENES]
        full = cache_for_items(
            tile="N40W075", window="2021-2025", items=items, factor=2, storage=storage
        )
        sampled = cache_for_items(
            tile="N40W075",
            window="2021-2025-sample300",
            items=items,
            factor=2,
            storage=storage,
        )
        assert full.key.storage_key != sampled.key.storage_key


def _stack(times: np.ndarray) -> xr.DataArray:
    """A tiny Celsius stack on ``times``, for the join to run against."""
    rng = np.random.default_rng(11)
    return xr.DataArray(
        rng.normal(25.0, 2.0, (len(times), 4, 4)).astype(np.float32),
        dims=["time", "latitude", "longitude"],
        coords={"time": times, "latitude": np.arange(4.0), "longitude": np.arange(4.0)},
    )


def _debias(lst: xr.DataArray, offset: xr.DataArray, n_valid: xr.DataArray):
    from landsat_lst.normalization import debias_with_offsets

    return debias_with_offsets(
        lst,
        offset,
        n_valid,
        max_offset_c=15.0,
        min_scene_pixels=0,
        min_offset_samples=0,
        offset_source_given=True,
    )


class TestTimestampPrecision:
    """The stamps are load-bearing, so they have to round-trip exactly.

    ``debias_with_offsets`` joins offsets to a stack **by coordinate value**.
    The axis was serialized at second precision, and real Landsat solar-day
    stamps carry sub-second components, so a warm record reconstructed
    timestamps the loaded axis does not hold. Observed on S30W065: every
    composite shard died with ``lst carries a time step the offsets do not
    ... ("not all values found in index 'time'")``.

    Under the old positional alignment the truncation was harmless, which is
    why it survived this long -- and every synthetic fixture used whole
    seconds, so only real data could catch it. This is a single-VM warm-cache
    bug too, not a sharded-run one.
    """

    def test_a_warm_cache_still_joins_and_gives_the_same_answer(self, storage):
        """Write, read back, debias -- bit-identical to never having cached."""
        offset, n_valid = _arrays(6, offset=[1.5, -2.0, 0.25, 3.0, -0.5, 0.75], n_valid=[900] * 6)
        lst = _stack(np.asarray(offset.time.values))
        cache = OffsetCache(storage=storage, key=_key())
        cache.write(offset, n_valid)

        hit = cache.read(offset.time)

        assert hit is not None, "a record this process just wrote must be readable"
        cold = _debias(lst, offset.astype(np.float32), n_valid)
        warm = _debias(lst, *hit)
        np.testing.assert_array_equal(np.asarray(cold[0].values), np.asarray(warm[0].values))
        np.testing.assert_array_equal(np.asarray(cold[2].values), np.asarray(warm[2].values))

    def test_an_axis_rebuilt_from_the_stored_stamps_still_joins(self, storage):
        """The exact shape the S30W065 failure took.

        A reader that only ever asks the cache about an axis it *loaded* cannot
        see the truncation: the arrays come back on whatever axis was passed in.
        The failure needs the round trip through the strings --
        ``shard_tasks._time_coord`` rebuilds the tile's axis from the stamps the
        plan froze, and a composite shard then joins that estimate onto a stack
        loaded at full precision. Truncated stamps rebuild a *different* axis,
        and every stamp in the join misses.
        """
        offset, n_valid = _arrays(4, offset=[1.5, -2.0, 0.25, 3.0], n_valid=[900] * 4)
        lst = _stack(np.asarray(offset.time.values))
        cache = OffsetCache(storage=storage, key=_key())
        cache.write(offset, n_valid)

        stored = json.loads(storage.read_text(cache.key.storage_key))["times"]
        rebuilt = xr.DataArray(pd.to_datetime(stored).values, dims=["time"])
        hit = cache.read(rebuilt)

        assert hit is not None
        # Raised `lst carries a time step the offsets do not` before the fix.
        assert int(_debias(lst, *hit)[0].sizes["time"]) == 4

    def test_the_stored_stamps_keep_their_sub_second_component(self, storage):
        offset, n_valid = _arrays(3, offset=[1.0, 2.0, 3.0], n_valid=[900] * 3)
        cache = OffsetCache(storage=storage, key=_key())

        cache.write(offset, n_valid)

        stored = json.loads(storage.read_text(cache.key.storage_key))["times"]
        assert all("." in stamp for stamp in stored)
        assert stored[0].endswith("123456000"), stored[0]

    def test_a_legacy_second_precision_record_is_accepted_by_truncation(self, storage):
        """S30W065's merged record is one of these, and it is a valid answer.

        The offsets never depended on how their timestamps were spelled, so
        rejecting the record would recompute half an hour of work to arrive at
        the same numbers.
        """
        offset, n_valid = _arrays(4, offset=[1.5, -2.0, 0.25, 3.0], n_valid=[900] * 4)
        cache = OffsetCache(storage=storage, key=_key())
        _write_legacy_record(storage, cache.key, offset, n_valid)

        hit = cache.read(offset.time)

        assert hit is not None
        assert cache.last_read_hit is True
        np.testing.assert_array_equal(np.asarray(hit[0].values), np.asarray(offset.values))
        np.testing.assert_array_equal(np.asarray(hit[1].values), np.asarray(n_valid.values))

    def test_a_legacy_record_comes_back_on_the_loaded_axis(self, storage):
        """Full precision, not the truncated stamps it was stored under.

        The returned arrays go straight into the coordinate join, so an axis
        rebuilt from the stored strings would fail exactly the way the bug did.
        """
        offset, n_valid = _arrays(4, offset=[1.5, -2.0, 0.25, 3.0], n_valid=[900] * 4)
        lst = _stack(np.asarray(offset.time.values))
        cache = OffsetCache(storage=storage, key=_key())
        _write_legacy_record(storage, cache.key, offset, n_valid)

        hit = cache.read(offset.time)

        assert hit is not None
        np.testing.assert_array_equal(
            np.asarray(hit[0].time.values), np.asarray(offset.time.values)
        )
        # And the join it exists to feed runs.
        assert int(_debias(lst, *hit)[0].sizes["time"]) == 4

    def test_an_ambiguous_truncation_is_a_miss_not_a_guess(self, storage):
        """Two scenes inside one second: the record fits more than one axis.

        Recompute rather than serve numbers that might belong to a different
        scene set. The digest should have caught that first, so this is the
        second line rather than the first.
        """
        times = pd.to_datetime(["2021-07-04T14:07:11.100000", "2021-07-04T14:07:11.900000"]).values
        coords = {"time": times}
        offset = xr.DataArray(np.array([1.5, -2.0]), dims=["time"], coords=coords)
        n_valid = xr.DataArray(np.array([900, 800]), dims=["time"], coords=coords)
        cache = OffsetCache(storage=storage, key=_key())
        _write_legacy_record(storage, cache.key, offset, n_valid)

        assert cache.read(offset.time) is None
        assert cache.last_read_hit is False

    def test_a_genuinely_different_axis_is_still_a_miss(self, storage):
        """The tolerance must not become "any axis of the same length"."""
        offset, n_valid = _arrays(3, offset=[1.0, 2.0, 3.0], n_valid=[900] * 3)
        cache = OffsetCache(storage=storage, key=_key())
        cache.write(offset, n_valid)

        other = xr.DataArray(_times(3) + np.timedelta64(1, "D"), dims=["time"])

        assert cache.read(other) is None

    def test_merge_accepts_a_partial_written_at_the_old_precision(self):
        """A run half-finished under the old spelling can still be merged."""
        from landsat_lst.offsets import merge_scene_partials

        offset, n_valid = _arrays(4, offset=[1.5, -2.0, 0.25, 3.0], n_valid=[900] * 4)
        legacy = {
            "times": [
                str(np.datetime_as_string(t, unit="s")) for t in np.asarray(offset.time.values)
            ],
            "offset": [float(v) for v in np.asarray(offset.values)],
            "n_valid": [int(v) for v in np.asarray(n_valid.values)],
        }

        merged_offset, merged_valid = merge_scene_partials([legacy], offset.time)

        np.testing.assert_array_equal(np.asarray(merged_offset.values), np.asarray(offset.values))
        np.testing.assert_array_equal(np.asarray(merged_valid.values), np.asarray(n_valid.values))
        np.testing.assert_array_equal(
            np.asarray(merged_offset.time.values), np.asarray(offset.time.values)
        )


def _write_legacy_record(storage, key, offset, n_valid) -> None:
    """A record as it was written before nanosecond serialization."""
    storage.write_text(
        key.storage_key,
        json.dumps(
            {
                "tile": key.tile,
                "window": key.window,
                "offset_resolution_factor": key.factor,
                "algorithm_version": key.algorithm_version,
                "digest": key.digest,
                "scenes": int(offset.sizes["time"]),
                "times": [
                    str(np.datetime_as_string(t, unit="s")) for t in np.asarray(offset.time.values)
                ],
                "offset": [float(v) for v in np.asarray(offset.values)],
                "n_valid": [int(v) for v in np.asarray(n_valid.values)],
                "duration_s": 1234.5,
            }
        ),
    )
