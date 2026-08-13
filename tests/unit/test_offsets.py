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


def _arrays(n: int = 2, *, offset=None, n_valid=None):
    """An ``(offset, n_valid)`` pair on a real datetime axis."""
    times = pd.to_datetime([f"2021-0{i + 1}-15" for i in range(n)]).values
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
