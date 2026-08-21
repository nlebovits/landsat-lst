"""Where a tile is cut, and what the pieces are called.

Every function here is pure arithmetic over a shape, which is exactly why it is
worth pinning: the cut is agreed on by processes that never talk to each other,
so a disagreement about it does not raise anywhere. It shows up as a band that
merges into a plausible tile with the wrong pixels in it.

Three properties carry the module. The block loop must be the *same*
construction the climatology walks, or a shard reduces a different set of
pixels than the whole-tile path does. Band boundaries must be chunk-aligned and
cover the grid exactly, or the merge stops being a windowed copy. And a plan
must refuse a process whose configuration has drifted, because every other
check downstream would pass.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from landsat_lst.config import settings
from landsat_lst.shards import (
    SHARD_PREFIX,
    TilePlan,
    balance_by_land,
    band_edges,
    band_key,
    block_spans,
    items_key,
    partition,
    plan_key,
    ref_block_key,
    ref_marker_key,
    scene_partial_key,
    shard_log_key,
    shard_root,
    shard_state_key,
)

pytestmark = pytest.mark.unit

#: The production grid. 18,000 = 512 x 35 + 80, so every band test that matters
#: runs against a ragged tail rather than a tidy division.
NATIVE = 18_000
BLOCKSIZE = 512


def _inline_spans(shape, block):
    """The construction that used to live inside ``climatology_by_blocks``.

    Kept verbatim so the lift is checked against the thing it replaced rather
    than against a paraphrase of it.
    """
    height, width = shape
    return [
        (y0, min(y0 + block, height), x0, min(x0 + block, width))
        for y0 in range(0, height, block)
        for x0 in range(0, width, block)
    ]


class TestBlockSpans:
    """One definition of the phase-A block loop, shared with normalization."""

    @pytest.mark.parametrize(
        ("shape", "block"),
        [((9000, 9000), 2048), ((4096, 4096), 1024), ((100, 250), 64), ((7, 7), 4)],
    )
    def test_matches_the_lifted_construction(self, shape, block):
        assert block_spans(shape, block) == _inline_spans(shape, block)

    def test_normalization_calls_this_one(self):
        """The lift is only worth anything if there is no second copy."""
        from landsat_lst import normalization

        assert normalization.block_spans is block_spans

    def test_spans_are_y_major(self):
        spans = block_spans((256, 256), 128)
        assert [(s[0], s[2]) for s in spans] == [(0, 0), (0, 128), (128, 0), (128, 128)]

    def test_edge_blocks_are_ragged_not_padded(self):
        spans = block_spans((100, 100), 64)
        assert spans[-1] == (64, 100, 64, 100)

    def test_every_pixel_is_covered_exactly_once(self):
        height, width, block = 130, 97, 32
        seen = set()
        for y0, y1, x0, x1 in block_spans((height, width), block):
            for y in range(y0, y1):
                for x in range(x0, x1):
                    assert (y, x) not in seen
                    seen.add((y, x))
        assert len(seen) == height * width

    def test_a_non_positive_block_is_refused(self):
        with pytest.raises(ValueError, match="must be positive"):
            block_spans((10, 10), 0)


class TestBandEdges:
    """Row bands, cut where the merge can copy blocks whole."""

    def test_the_production_grid_has_a_ragged_tail(self):
        """18,000 is not a multiple of 512, and the rule has to say so.

        A cut that assumed an exact division would be wrong on every tile this
        project writes, so the last band absorbing the remainder is mandatory
        rather than a tidiness preference.
        """
        bands = band_edges(NATIVE, 4, BLOCKSIZE)

        assert bands[-1][1] == NATIVE
        assert (bands[-1][1] - bands[-1][0]) % BLOCKSIZE == 80 % BLOCKSIZE
        assert all((start % BLOCKSIZE) == 0 for start, _ in bands)

    @pytest.mark.parametrize("n_bands", [1, 2, 3, 4, 5, 7, 12, 35])
    def test_bands_cover_the_grid_exactly(self, n_bands):
        bands = band_edges(NATIVE, n_bands, BLOCKSIZE)

        assert len(bands) == n_bands
        assert bands[0][0] == 0
        assert bands[-1][1] == NATIVE
        for (_, stop), (next_start, _) in pairwise(bands):
            assert stop == next_start
        assert all(start < stop for start, stop in bands)

    @pytest.mark.parametrize("n_bands", [1, 2, 3, 4, 5, 7, 12, 35])
    def test_every_boundary_is_a_block_row(self, n_bands):
        """What makes the merge a copy rather than a read-modify-write."""
        bands = band_edges(NATIVE, n_bands, BLOCKSIZE)

        for start, _ in bands:
            assert start % BLOCKSIZE == 0

    def test_it_is_deterministic(self):
        assert band_edges(NATIVE, 6, BLOCKSIZE) == band_edges(NATIVE, 6, BLOCKSIZE)

    def test_more_bands_than_chunk_rows_is_refused(self):
        """Silently returning empty bands would hand a shard nothing to do."""
        with pytest.raises(ValueError, match="only 2 chunk rows"):
            band_edges(600, 3, BLOCKSIZE)

    def test_an_exact_division_still_works(self):
        assert band_edges(1024, 2, 512) == [(0, 512), (512, 1024)]


class TestPartition:
    """Contiguous, deterministic, and identical in two processes."""

    def test_contiguous_and_complete(self):
        groups = partition(list(range(10)), 3)

        assert groups == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test_the_remainder_goes_to_the_earliest_groups(self):
        assert [len(g) for g in partition(range(11), 4)] == [3, 3, 3, 2]

    def test_it_is_deterministic(self):
        assert partition(range(97), 7) == partition(range(97), 7)

    def test_one_group_is_the_whole_thing(self):
        assert partition(range(4), 1) == [[0, 1, 2, 3]]

    def test_more_groups_than_items_is_refused(self):
        with pytest.raises(ValueError, match="cannot split"):
            partition(range(2), 3)


class TestBalanceByLand:
    """An equal split of blocks is not an equal split of work."""

    def test_land_is_spread_more_evenly_than_a_plain_split(self):
        """The coastal case: all the land sits in the first quarter.

        A plain partition hands one shard every block it must read. Balancing
        moves the boundaries so each group carries a similar share.
        """
        spans = block_spans((400, 20), 20)
        has_land = [i < len(spans) // 4 for i in range(len(spans))]

        balanced = balance_by_land(spans, has_land, 4)
        plain = partition(spans, 4)

        def land_counts(groups):
            index = dict(zip(spans, has_land, strict=True))
            return [sum(index[s] for s in group) for group in groups]

        spread_balanced = max(land_counts(balanced)) - min(land_counts(balanced))
        spread_plain = max(land_counts(plain)) - min(land_counts(plain))
        assert spread_balanced < spread_plain

    def test_groups_stay_contiguous_and_complete(self):
        spans = block_spans((400, 40), 20)
        has_land = [i % 3 == 0 for i in range(len(spans))]

        groups = balance_by_land(spans, has_land, 5)

        assert [s for group in groups for s in group] == spans
        assert all(groups)

    def test_every_group_is_non_empty_even_when_land_is_one_block(self):
        spans = block_spans((200, 20), 20)
        has_land = [i == 0 for i in range(len(spans))]

        groups = balance_by_land(spans, has_land, 4)

        assert len(groups) == 4
        assert all(groups)

    def test_no_land_anywhere_degenerates_to_partition(self):
        spans = block_spans((200, 20), 20)

        assert balance_by_land(spans, [False] * len(spans), 3) == partition(spans, 3)

    def test_it_is_deterministic(self):
        spans = block_spans((400, 40), 20)
        has_land = [i % 7 < 3 for i in range(len(spans))]

        assert balance_by_land(spans, has_land, 4) == balance_by_land(spans, has_land, 4)

    def test_mismatched_inputs_are_refused(self):
        with pytest.raises(ValueError, match="disagree"):
            balance_by_land(block_spans((40, 40), 20), [True], 2)


class TestKeyGrammar:
    """One module owns the suffixes, as ``runs.py`` does for the run prefix."""

    def test_the_root_names_the_run_and_the_tile(self):
        assert shard_root("r1", "N40W075") == "_shards/r1/N40W075"

    def test_shard_keys_never_land_under_the_run_prefix(self):
        """``runs.classify`` reads everything under ``_runs/`` as a tile artifact.

        ``plan.json`` filed there would become a tile named ``plan``. The two
        prefixes are disjoint so that cannot happen.
        """
        from landsat_lst.storage import RUN_RECORD_PREFIX

        assert SHARD_PREFIX != RUN_RECORD_PREFIX
        assert not shard_root("r1", "N40W075").startswith(f"{RUN_RECORD_PREFIX}/")

    def test_plan_and_items_sit_at_the_root(self):
        root = shard_root("r1", "N40W075")

        assert plan_key(root) == "_shards/r1/N40W075/plan.json"
        assert items_key(root) == "_shards/r1/N40W075/items.json"

    def test_ref_blocks_are_zero_padded_so_a_listing_sorts(self):
        root = shard_root("r1", "N40W075")

        assert ref_block_key(root, 7) == "_shards/r1/N40W075/offsets/ref/b0007.npy"
        assert ref_block_key(root, 1234) == "_shards/r1/N40W075/offsets/ref/b1234.npy"

    def test_the_nan_marker_hangs_off_its_block(self):
        root = shard_root("r1", "N40W075")

        assert ref_marker_key(root, 7) == ref_block_key(root, 7) + ".nan"

    def test_a_scene_partial_names_its_range(self):
        root = shard_root("r1", "N40W075")

        assert scene_partial_key(root, 0, 120) == (
            "_shards/r1/N40W075/offsets/scene/s000000-000120.json"
        )

    def test_a_band_names_its_product(self):
        root = shard_root("r1", "N40W075")

        assert band_key(root, "lst_p95", 2) == "_shards/r1/N40W075/composite/lst_p95/band002.tif"
        assert band_key(root, "qa_count", 2) == "_shards/r1/N40W075/composite/qa_count/band002.tif"

    def test_state_and_log_carry_stage_index_and_attempt(self):
        root = shard_root("r1", "N40W075")

        assert shard_state_key(root, "ref", 3, 2) == "_shards/r1/N40W075/state/ref.0003.2.json"
        assert shard_log_key(root, "band", 0, 1) == "_shards/r1/N40W075/state/band.0000.1.log"

    def test_two_attempts_of_one_shard_do_not_share_a_key(self):
        """The lesson ``runs.py`` paid for: a retry must not erase its predecessor."""
        root = shard_root("r1", "N40W075")

        assert shard_state_key(root, "ref", 3, 1) != shard_state_key(root, "ref", 3, 2)


def _plan(**overrides) -> TilePlan:
    base = {
        "tile": "N40W075",
        "window": "2021-2025",
        "scene_ids": ["LC08_a", "LC09_b"],
        "scene_times": ["2021-07-04T15:30:00", "2022-07-11T15:30:00"],
        "offset_factor": 2,
        "coarse_shape": (9000, 9000),
        "native_shape": (NATIVE, NATIVE),
        "block_edge": 2048,
        "blocks": block_spans((9000, 9000), 2048),
        "block_has_land": [True] * len(block_spans((9000, 9000), 2048)),
        "scene_batches": [(0, 1), (1, 2)],
        "bands": band_edges(NATIVE, 4, BLOCKSIZE),
        "ref_shards": 3,
        "scene_shards": 2,
        "band_shards": 4,
    }
    return TilePlan(**{**base, **overrides})


class TestTilePlan:
    """A plan survives storage, and refuses a process that has drifted."""

    def test_round_trip_preserves_every_field(self):
        plan = _plan()

        back = TilePlan.from_dict(plan.to_dict())

        assert back.tile == plan.tile
        assert back.window == plan.window
        assert back.scene_ids == plan.scene_ids
        assert back.scene_times == plan.scene_times
        assert back.offset_factor == plan.offset_factor
        assert back.coarse_shape == plan.coarse_shape
        assert back.native_shape == plan.native_shape
        assert back.block_edge == plan.block_edge
        assert back.blocks == plan.blocks
        assert back.block_has_land == plan.block_has_land
        assert back.scene_batches == plan.scene_batches
        assert back.bands == plan.bands
        assert (back.ref_shards, back.scene_shards, back.band_shards) == (3, 2, 4)

    def test_the_record_is_plain_json(self):
        import json

        payload = _plan().to_dict()

        assert TilePlan.from_dict(json.loads(json.dumps(payload))).blocks == _plan().blocks

    def test_the_digest_is_stamped_into_the_record(self):
        plan = _plan()

        assert plan.to_dict()["digest"] == plan.digest

    def test_a_drifted_clamp_is_refused(self, monkeypatch):
        """``convert_to_celsius`` applies the clamp before any median sees a pixel."""
        payload = _plan().to_dict()
        monkeypatch.setattr(settings, "lst_valid_max", settings.lst_valid_max + 1)

        with pytest.raises(ValueError, match="different configuration"):
            TilePlan.from_dict(payload)

    def test_a_drifted_offset_factor_is_refused(self, monkeypatch):
        payload = _plan().to_dict()
        monkeypatch.setattr(
            settings,
            "destripe_offset_resolution_factor",
            settings.destripe_offset_resolution_factor * 2,
        )

        with pytest.raises(ValueError, match="different configuration"):
            TilePlan.from_dict(payload)

    def test_a_drifted_chunk_size_is_refused(self, monkeypatch):
        """A shard loading on other chunks cuts its batches on other boundaries."""
        payload = _plan().to_dict()
        monkeypatch.setattr(
            settings, "load_chunk_size_offsets", settings.load_chunk_size_offsets * 2
        )

        with pytest.raises(ValueError, match="different configuration"):
            TilePlan.from_dict(payload)

    def test_the_scene_set_changes_the_digest(self):
        assert _plan().digest != _plan(scene_ids=["LC08_a"]).digest

    def test_scene_order_does_not_change_the_digest(self):
        """STAC pages in whatever order it likes; the same set must key alike."""
        assert _plan().digest == _plan(scene_ids=["LC09_b", "LC08_a"]).digest

    def test_the_algorithm_version_is_in_the_digest(self, monkeypatch):
        """The escape hatch for code changes a hash cannot see."""
        from landsat_lst import offsets as offsets_module

        payload = _plan().to_dict()
        monkeypatch.setattr(offsets_module, "ALGORITHM_VERSION", 99)

        with pytest.raises(ValueError, match="different configuration"):
            TilePlan.from_dict(payload)

    def test_a_record_without_a_digest_is_accepted(self):
        """Nothing writes one, but a hand-written plan should still load."""
        payload = _plan().to_dict()
        del payload["digest"]

        assert TilePlan.from_dict(payload).tile == "N40W075"
