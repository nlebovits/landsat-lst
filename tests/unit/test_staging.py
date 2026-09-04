"""The coarse observation stage (issue #125).

Phase A stages what it decodes; phase B reads that instead of the Landsat
sources. The properties that make it an execution change rather than a
scientific one are pinned here: the estimator's inputs and outputs are
unchanged, a stale stage is unreachable, a land-free block is never staged and
never read as data, and the scratch does not outlive the record it produced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from landsat_lst.normalization import (
    _read_values,
    _scene_batches,
    climatology_by_blocks,
    offsets_by_scene,
)
from landsat_lst.offsets import OffsetKey
from landsat_lst.qa import (
    DN_SENTINEL,
    apply_qa_mask,
    celsius_stack,
    convert_to_celsius,
    dn_stack,
)
from landsat_lst.shards import block_spans
from landsat_lst.staging import (
    CoarseStage,
    StageKey,
    stage_batches,
    staged_batch_reader,
    staging_block_reader,
)
from landsat_lst.storage import LocalStorage

CLOUD = 1 << 3
TIME_CHUNK = 10
BLOCK = 16


def _fixture(seed=0, t=40, h=48, w=48, land_rows=18):
    """A chunked stack plus a land mask whose top rows hold no land at all."""
    rng = np.random.default_rng(seed)
    dn = rng.integers(1, 65_535, size=(t, h, w), dtype=np.uint16)
    dn[rng.random(dn.shape) < 0.05] = 0
    qa = np.where(rng.random(dn.shape) < 0.15, CLOUD, 0).astype(np.uint16)
    times = pd.date_range("2021-01-03T13:56:04.018086", periods=t, freq="29D")
    ds = xr.Dataset(
        {
            "lwir11": (("time", "latitude", "longitude"), dn),
            "qa_pixel": (("time", "latitude", "longitude"), qa),
        },
        coords={
            "time": times,
            "latitude": np.arange(h, dtype=float),
            "longitude": np.arange(w, dtype=float),
        },
    ).chunk({"time": TIME_CHUNK, "latitude": BLOCK, "longitude": BLOCK})
    land = np.ones((h, w), bool)
    land[:land_rows] = False
    mask = xr.DataArray(
        land,
        dims=("latitude", "longitude"),
        coords={"latitude": ds.latitude, "longitude": ds.longitude},
    )
    return ds, mask


def _parts(ds, mask):
    direct = convert_to_celsius(apply_qa_mask(ds)["lwir11"]).where(mask)
    dn = dn_stack(ds).where(mask, DN_SENTINEL)
    h, w = int(ds.sizes["latitude"]), int(ds.sizes["longitude"])
    spans = block_spans((h, w), BLOCK)
    has_land = [bool(np.asarray(mask.values)[y0:y1, x0:x1].any()) for y0, y1, x0, x1 in spans]
    return direct, dn, celsius_stack(dn), spans, has_land, (h, w)


def _stage(root: Path, digest="deadbeefdeadbeef", algorithm_version=2):
    return CoarseStage(
        LocalStorage(root),
        StageKey(root="run/T", algorithm_version=algorithm_version, digest=digest),
    )


def _run_phase_a(via, dn, mask, spans, stage):
    index_of = {span: i for i, span in enumerate(spans)}
    reader = staging_block_reader(
        dn,
        stage,
        block_index=lambda span: index_of[span],
        batches=stage_batches(dn),
        read_values=_read_values,
    )
    return climatology_by_blocks(via, block=BLOCK, land_mask=mask, spans=spans, block_reader=reader)


class TestRepresentation:
    def test_staged_objects_are_uint16(self, tmp_path):
        ds, mask = _fixture()
        _direct, dn, via, spans, _has_land, _shape = _parts(ds, mask)
        stage = _stage(tmp_path)
        _run_phase_a(via, dn, mask, spans, stage)
        keys = sorted(stage.storage.list_prefix(stage.key.prefix))
        assert keys
        for key in keys:
            assert np.load(tmp_path / key).dtype == np.uint16

    def test_phase_b_input_is_the_direct_read_bit_for_bit(self, tmp_path):
        ds, mask = _fixture()
        direct, dn, via, spans, has_land, shape = _parts(ds, mask)
        stage = _stage(tmp_path)
        _run_phase_a(via, dn, mask, spans, stage)
        reader = staged_batch_reader(
            stage, blocks=spans, block_has_land=has_land, batches=stage_batches(dn), shape=shape
        )
        for span in _scene_batches(direct, 8):
            want = _read_values(direct.isel(time=slice(*span)), np.dtype(np.float32))
            np.testing.assert_array_equal(reader(span), want)


class TestEstimatorIsUnchanged:
    def test_offsets_and_n_valid_are_identical(self, tmp_path):
        ds, mask = _fixture(seed=3)
        direct, dn, via, spans, has_land, shape = _parts(ds, mask)
        ref0, months0 = climatology_by_blocks(direct, block=BLOCK, land_mask=mask, spans=spans)
        batches = _scene_batches(direct, 8)
        off0, nv0 = offsets_by_scene(direct, ref0, months0, batches=batches)

        stage = _stage(tmp_path)
        ref1, months1 = _run_phase_a(via, dn, mask, spans, stage)
        reader = staged_batch_reader(
            stage, blocks=spans, block_has_land=has_land, batches=stage_batches(dn), shape=shape
        )
        off1, nv1 = offsets_by_scene(via, ref1, months1, batches=batches, batch_reader=reader)

        np.testing.assert_array_equal(ref0, ref1)
        np.testing.assert_array_equal(months0, months1)
        np.testing.assert_array_equal(off0.values, off1.values)
        np.testing.assert_array_equal(nv0.values, nv1.values)


class TestPartialLand:
    def test_a_land_free_block_is_never_staged(self, tmp_path):
        ds, mask = _fixture()
        _direct, dn, via, spans, has_land, _shape = _parts(ds, mask)
        assert not all(has_land), "the fixture must contain a land-free block"
        stage = _stage(tmp_path)
        _run_phase_a(via, dn, mask, spans, stage)
        staged_blocks = {
            int(Path(k).name.split(".")[0][1:]) for k in stage.storage.list_prefix(stage.key.prefix)
        }
        assert staged_blocks == {i for i, land in enumerate(has_land) if land}

    def test_a_land_free_block_reads_back_as_no_observation(self, tmp_path):
        ds, mask = _fixture()
        direct, dn, via, spans, has_land, shape = _parts(ds, mask)
        stage = _stage(tmp_path)
        _run_phase_a(via, dn, mask, spans, stage)
        reader = staged_batch_reader(
            stage, blocks=spans, block_has_land=has_land, batches=stage_batches(dn), shape=shape
        )
        got = reader((0, TIME_CHUNK))
        for (y0, y1, x0, x1), land in zip(spans, has_land, strict=True):
            if not land:
                assert np.isnan(got[:, y0:y1, x0:x1]).all()
        # and the count the estimator derives from it is the direct one
        want = _read_values(direct.isel(time=slice(0, TIME_CHUNK)), np.dtype(np.float32))
        assert int(np.isfinite(got).sum()) == int(np.isfinite(want).sum())

    def test_a_missing_land_block_raises_rather_than_thinning_n_valid(self, tmp_path):
        ds, mask = _fixture()
        _direct, dn, via, spans, has_land, shape = _parts(ds, mask)
        stage = _stage(tmp_path)
        _run_phase_a(via, dn, mask, spans, stage)
        victim = next(k for k in sorted(stage.storage.list_prefix(stage.key.prefix)))
        (tmp_path / victim).unlink()
        reader = staged_batch_reader(
            stage, blocks=spans, block_has_land=has_land, batches=stage_batches(dn), shape=shape
        )
        with pytest.raises(FileNotFoundError, match="staged block"):
            reader((0, TIME_CHUNK))


class TestKeying:
    def test_the_prefix_carries_the_offset_key_terms(self):
        key = OffsetKey.build(tile="N40W075", window="2021-2025", factor=2, scene_ids=["a", "b"])
        stage = StageKey.from_offset_key("run/N40W075", key)
        assert key.digest in stage.prefix
        assert f"v{key.algorithm_version}" in stage.prefix

    @pytest.mark.parametrize(
        ("field", "value"),
        [("digest", "0123456789abcdef"), ("algorithm_version", 99)],
    )
    def test_a_stale_stage_is_at_a_prefix_this_one_never_lists(self, tmp_path, field, value):
        ds, mask = _fixture()
        _direct, dn, via, spans, _has_land, _shape = _parts(ds, mask)
        written = _stage(tmp_path)
        _run_phase_a(via, dn, mask, spans, written)
        assert written.storage.list_prefix(written.key.prefix)

        fresh = _stage(tmp_path, **{field: value})
        assert fresh.key.prefix != written.key.prefix
        assert fresh.storage.list_prefix(fresh.key.prefix) == {}
        assert fresh.read(block=1, batch=0) is None

    def test_scene_set_change_moves_the_prefix(self):
        base = OffsetKey.build(tile="T", window="2021-2025", factor=2, scene_ids=["a", "b"])
        more = OffsetKey.build(tile="T", window="2021-2025", factor=2, scene_ids=["a", "b", "c"])
        coarser = OffsetKey.build(tile="T", window="2021-2025", factor=4, scene_ids=["a", "b"])
        prefixes = {StageKey.from_offset_key("r/T", k).prefix for k in (base, more, coarser)}
        assert len(prefixes) == 3


class TestLifecycle:
    def test_batches_align_to_the_source_time_chunks(self):
        ds, mask = _fixture()
        _direct, dn, _via, _spans, _has_land, _shape = _parts(ds, mask)
        spans = stage_batches(dn)
        assert spans == [(i, i + TIME_CHUNK) for i in range(0, 40, TIME_CHUNK)]
        # every phase-B group is a whole number of staged objects
        for start, stop in _scene_batches(dn, 8):
            covered = [s for s in spans if s[0] >= start and s[1] <= stop]
            assert sum(b - a for a, b in covered) == stop - start

    def test_a_rerun_overwrites_rather_than_duplicates(self, tmp_path):
        ds, mask = _fixture()
        _direct, dn, via, spans, _has_land, _shape = _parts(ds, mask)
        stage = _stage(tmp_path)
        _run_phase_a(via, dn, mask, spans, stage)
        first = dict(stage.storage.list_prefix(stage.key.prefix))
        payload = {k: (tmp_path / k).read_bytes() for k in first}
        _run_phase_a(via, dn, mask, spans, stage)
        second = dict(stage.storage.list_prefix(stage.key.prefix))
        assert set(first) == set(second)
        assert all((tmp_path / k).read_bytes() == payload[k] for k in second)

    def test_cleanup_removes_everything_and_is_idempotent(self, tmp_path):
        ds, mask = _fixture()
        _direct, dn, via, spans, _has_land, _shape = _parts(ds, mask)
        stage = _stage(tmp_path)
        _run_phase_a(via, dn, mask, spans, stage)
        n = len(stage.storage.list_prefix(stage.key.prefix))
        assert n > 0
        assert stage.cleanup() == n
        assert stage.storage.list_prefix(stage.key.prefix) == {}
        assert stage.cleanup() == 0

    def test_cleanup_leaves_a_peer_stage_alone(self, tmp_path):
        ds, mask = _fixture()
        _direct, dn, via, spans, _has_land, _shape = _parts(ds, mask)
        mine = _stage(tmp_path)
        theirs = _stage(tmp_path, digest="ffffffffffffffff")
        _run_phase_a(via, dn, mask, spans, mine)
        _run_phase_a(via, dn, mask, spans, theirs)
        mine.cleanup()
        assert mine.storage.list_prefix(mine.key.prefix) == {}
        assert theirs.storage.list_prefix(theirs.key.prefix) != {}

    def test_write_refuses_anything_but_uint16(self, tmp_path):
        stage = _stage(tmp_path)
        with pytest.raises(TypeError, match="uint16"):
            stage.write(0, 0, np.zeros((2, 2, 2), dtype=np.float32))


class TestDefaultPathUntouched:
    def test_no_reader_is_the_direct_read(self, tmp_path):
        """The seam is additive: unset must reproduce the pre-#125 behaviour."""
        ds, mask = _fixture(seed=7)
        direct, _dn, _via, spans, _has_land, _shape = _parts(ds, mask)
        a_ref, a_months = climatology_by_blocks(direct, block=BLOCK, land_mask=mask, spans=spans)
        batches = _scene_batches(direct, 8)
        a_off, a_nv = offsets_by_scene(direct, a_ref, a_months, batches=batches)
        b_ref, b_months = climatology_by_blocks(
            direct, block=BLOCK, land_mask=mask, spans=spans, block_reader=None
        )
        b_off, b_nv = offsets_by_scene(direct, b_ref, b_months, batches=batches, batch_reader=None)
        np.testing.assert_array_equal(a_ref, b_ref)
        np.testing.assert_array_equal(a_off.values, b_off.values)
        np.testing.assert_array_equal(a_nv.values, b_nv.values)
