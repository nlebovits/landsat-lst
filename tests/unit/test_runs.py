"""The key grammar of ``_runs/{run_id}/``.

Every reader of a run prefix used to re-derive this with its own suffix tests,
and one of them got it wrong in a way nobody saw until a profiled tile grew a
phantom sibling in the watch table. These tests are pure functions over a
listing, so the rule can be pinned without a storage backend at all.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from landsat_lst.runs import (
    POINTER_ATTEMPT,
    classify,
    resolve_attempt,
    split_attempt,
    tile_artifact_prefix,
)
from landsat_lst.storage import LocalStorage

pytestmark = pytest.mark.unit

STAMP = datetime(2026, 8, 14, 10, 34, tzinfo=UTC)


def _listing(*keys: str) -> dict[str, datetime]:
    return dict.fromkeys(keys, STAMP)


class TestSplitAttempt:
    def test_reads_the_attempt_off_a_stem(self):
        assert split_attempt("N40W075.2") == ("N40W075", 2)

    def test_a_bare_stem_is_the_pointer(self):
        assert split_attempt("N40W075") == ("N40W075", POINTER_ATTEMPT)

    def test_double_digit_attempts_parse(self):
        assert split_attempt("N40W075.12") == ("N40W075", 12)

    def test_a_label_is_not_an_attempt(self):
        # ".destripe_offsets" is not digits, so the whole stem is the tile.
        assert split_attempt("N40W075.destripe_offsets") == ("N40W075.destripe_offsets", 0)


class TestClassify:
    def test_a_profile_key_is_not_a_tile(self):
        """The bug this module exists to close.

        ``{tile}.{label}.profile.json`` also ends in ``.json``. Read as a state
        object it produced a tile named ``N40W075.destripe_offsets``, which
        watch rendered as a finished row and subtracted from its pending count.
        """
        found = classify(_listing("_runs/r/N40W075.1.destripe_offsets.profile.json"))

        assert set(found) == {"N40W075"}
        assert found["N40W075"].states == {}
        assert found["N40W075"].profiles == {
            1: {"destripe_offsets": "_runs/r/N40W075.1.destripe_offsets.profile.json"}
        }

    def test_groups_one_attempt_completely(self):
        found = classify(
            _listing(
                "_runs/r/N40W075.1.json",
                "_runs/r/N40W075.1.log",
                "_runs/r/N40W075.1.destripe_offsets.profile.json",
            )
        )
        art = found["N40W075"]

        assert art.attempt == 1
        assert art.state_key == "_runs/r/N40W075.1.json"
        assert art.log_key == "_runs/r/N40W075.1.log"
        assert art.profile_keys == {
            "destripe_offsets": "_runs/r/N40W075.1.destripe_offsets.profile.json"
        }

    def test_the_newest_attempt_wins(self):
        found = classify(
            _listing(
                "_runs/r/N40W075.1.json",
                "_runs/r/N40W075.1.log",
                "_runs/r/N40W075.3.json",
                "_runs/r/N40W075.3.log",
                "_runs/r/N40W075.2.json",
            )
        )
        art = found["N40W075"]

        assert art.attempt == 3
        assert art.state_key == "_runs/r/N40W075.3.json"
        assert art.log_key == "_runs/r/N40W075.3.log"
        assert art.attempts == [1, 2, 3]

    def test_an_earlier_attempt_survives_its_successor(self):
        """The whole point of A2. Attempt 2 reached land_mask and died."""
        found = classify(_listing("_runs/r/N40W075.1.json", "_runs/r/N40W075.2.json"))

        assert found["N40W075"].states == {
            1: "_runs/r/N40W075.1.json",
            2: "_runs/r/N40W075.2.json",
        }

    def test_tiles_do_not_bleed_into_each_other(self):
        found = classify(_listing("_runs/r/N40W075.1.json", "_runs/r/S05W060.1.json"))

        assert set(found) == {"N40W075", "S05W060"}

    def test_an_unrecognised_key_is_ignored(self):
        assert classify(_listing("_runs/r/something.txt")) == {}

    def test_an_empty_listing_is_empty(self):
        assert classify({}) == {}


class TestSettled:
    def test_a_running_tile_has_not_settled(self):
        found = classify(_listing("_runs/r/N40W075.1.json"))

        assert found["N40W075"].settled is False

    def test_the_pointer_means_settled(self):
        found = classify(_listing("_runs/r/N40W075.1.json", "_runs/r/N40W075.json"))

        assert found["N40W075"].settled is True

    def test_the_body_read_is_the_attempt_not_the_pointer(self):
        """They hold the same body, and the attempt key is the one that scales."""
        found = classify(_listing("_runs/r/N40W075.2.json", "_runs/r/N40W075.json"))

        assert found["N40W075"].body_key == "_runs/r/N40W075.2.json"


class TestLegacyRuns:
    """A run submitted before this scheme, read after it."""

    def test_the_heartbeat_is_the_body(self):
        found = classify(_listing("_runs/r/N40W075.progress.json", "_runs/r/N40W075.log"))
        art = found["N40W075"]

        assert art.body_key == "_runs/r/N40W075.progress.json"
        assert art.log_key == "_runs/r/N40W075.log"
        assert art.settled is False

    def test_a_record_beside_a_heartbeat_settles_it(self):
        found = classify(_listing("_runs/r/N40W075.progress.json", "_runs/r/N40W075.json"))
        art = found["N40W075"]

        assert art.settled is True
        # The heartbeat still wins as the body: it is the one with a phase.
        assert art.body_key == "_runs/r/N40W075.progress.json"

    def test_a_tile_that_only_left_a_record_is_still_read(self):
        found = classify(_listing("_runs/r/N40W075.json"))
        art = found["N40W075"]

        assert art.body_key == "_runs/r/N40W075.json"
        assert art.settled is True
        assert art.attempt == POINTER_ATTEMPT

    def test_a_legacy_profile_is_still_not_a_tile(self):
        found = classify(_listing("_runs/r/N40W075.destripe_offsets.profile.json"))

        assert set(found) == {"N40W075"}
        assert found["N40W075"].profiles == {
            POINTER_ATTEMPT: {"destripe_offsets": "_runs/r/N40W075.destripe_offsets.profile.json"}
        }

    def test_a_mixed_run_reads_both_shapes(self):
        """A retry running new code inside a run that started on old code."""
        found = classify(_listing("_runs/r/N40W075.progress.json", "_runs/r/N40W075.2.json"))
        art = found["N40W075"]

        assert art.attempt == 2
        assert art.legacy_progress == "_runs/r/N40W075.progress.json"


class TestArtifactPrefix:
    def test_selects_one_tile(self):
        assert tile_artifact_prefix("r", "N40W075") == "_runs/r/N40W075."

    def test_the_trailing_dot_stops_a_longer_name_matching(self):
        assert not "_runs/r/N40W075.1.json".startswith(tile_artifact_prefix("r", "N40W07"))


class TestResolveAttempt:
    def test_a_first_attempt_is_one(self, tmp_path):
        storage = LocalStorage(output_dir=tmp_path)

        assert resolve_attempt(storage, "r", "N40W075") == 1

    def test_a_retry_takes_the_next_number(self, tmp_path):
        storage = LocalStorage(output_dir=tmp_path)
        storage.write_text("_runs/r/N40W075.1.json", "{}")

        assert resolve_attempt(storage, "r", "N40W075") == 2

    def test_the_log_of_an_earlier_attempt_counts(self, tmp_path):
        """A VM killed before it wrote state still leaves a log."""
        storage = LocalStorage(output_dir=tmp_path)
        storage.write_text("_runs/r/N40W075.1.log", "")

        assert resolve_attempt(storage, "r", "N40W075") == 2

    def test_the_pointer_does_not_count_as_an_attempt(self, tmp_path):
        storage = LocalStorage(output_dir=tmp_path)
        storage.write_text("_runs/r/N40W075.json", "{}")

        assert resolve_attempt(storage, "r", "N40W075") == 1

    def test_another_tile_does_not_advance_this_one(self, tmp_path):
        storage = LocalStorage(output_dir=tmp_path)
        storage.write_text("_runs/r/S05W060.4.json", "{}")

        assert resolve_attempt(storage, "r", "N40W075") == 1

    def test_a_failed_listing_falls_back_to_one(self, tmp_path):
        """A tile must never fail because it could not name its own log."""

        class BrokenStorage(LocalStorage):
            def list_prefix(self, prefix):
                raise OSError("bucket unreachable")

        assert resolve_attempt(BrokenStorage(output_dir=tmp_path), "r", "N40W075") == 1
