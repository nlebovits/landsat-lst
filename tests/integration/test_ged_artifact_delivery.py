"""Can a fleet VM actually find the GED gap mask?

``settings.ged_gap_mask`` defaults on, so every production tile reaches for a
mask. ``settings.ged_artifact``'s default, ``data/ged_gap_mask.npz``, resolves
against the *working directory* -- and a Coiled VM runs from wherever Coiled
drops it, with no repo checkout and no granule archive. The artifact is also
gitignored at the repo root and excluded from the wheel by default. Nothing in
the unit suite notices any of that, because the unit suite runs inside the
checkout.

So this builds a wheel, installs it into a clean virtualenv, and runs it from
a directory that has no repo, no ``data/``, and no ``data/aster_ged``. That is
the VM's situation, reproduced.

Since #118 a complete artifact ships inside the wheel at
``landsat_lst/data/ged_gap_mask.npz``, so the assertion is that the resolver
finds it from a foreign CWD with no ``data/`` at all, and that its content
hash is pinned. The loud-failure branch is kept for a build that somehow lacks
the file: the one thing that must never happen is a quiet wrong answer.
"""

from __future__ import annotations

import json
import subprocess
import sys
import venv
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Run inside the clean venv, from a directory with no repo. Prints one JSON
#: object describing what the resolver found -- never the mask itself, which
#: would need granules.
PROBE = """
import json, os, pathlib, sys
from landsat_lst import ged
from landsat_lst.config import settings

out = {
    "cwd": os.getcwd(),
    "packaged": None,
    "kind": None,
    "error": None,
    "pinned": ged.GED_ARTIFACT_CONTENT_SHA256,
    "ged_dir_exists": settings.ged_dir.exists(),
    "configured_artifact_exists": settings.ged_artifact.exists(),
}
packaged = ged.packaged_artifact_path()
out["packaged"] = None if packaged is None else str(packaged)
try:
    kind, source = ged._resolve_source()
    out["kind"] = kind
    out["source"] = str(source)
except FileNotFoundError as e:
    out["error"] = str(e)
print("PROBE" + json.dumps(out))
"""


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    """Build the wheel the way a release would."""
    dist = tmp_path_factory.mktemp("dist")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(dist), str(REPO_ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"could not build a wheel here: {result.stderr[-2000:]}")
    wheels = list(dist.glob("landsat_lst-*.whl"))
    if not wheels:
        pytest.skip("no wheel produced")
    return wheels[0]


@pytest.fixture(scope="module")
def installed(wheel, tmp_path_factory) -> Path:
    """A clean virtualenv with only the wheel and its dependencies."""
    env_dir = tmp_path_factory.mktemp("venv")
    venv.create(env_dir, with_pip=True)
    python = env_dir / "bin" / "python"
    result = subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", str(wheel)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"could not install the wheel: {result.stderr[-2000:]}")
    return python


@pytest.fixture(scope="module")
def probe(installed, tmp_path_factory) -> dict:
    """Run the resolver from a foreign CWD with no repo data."""
    foreign = tmp_path_factory.mktemp("foreign_cwd")
    assert not (foreign / "data").exists()
    result = subprocess.run(
        [str(installed), "-c", PROBE],
        cwd=foreign,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    line = next(x for x in result.stdout.splitlines() if x.startswith("PROBE"))
    return json.loads(line[len("PROBE") :])


class TestDelivery:
    def test_the_probe_really_ran_without_repo_data(self, probe):
        """Otherwise the test proves nothing about a VM."""
        assert probe["ged_dir_exists"] is False
        assert probe["configured_artifact_exists"] is False
        assert REPO_ROOT.as_posix() not in probe["cwd"]

    def test_resolution_is_either_a_packaged_artifact_or_a_loud_failure(self, probe):
        """The one thing that must never happen is a quiet wrong answer."""
        if probe["packaged"] is None:
            assert probe["kind"] is None
            assert probe["error"] is not None
        else:
            assert probe["kind"] == "artifact"
            assert probe["error"] is None

    def test_a_packaged_artifact_is_what_the_resolver_picks(self, probe):
        """The acceptance criterion for #118: a VM with the wheel and nothing
        else resolves the production mask."""
        assert probe["packaged"] is not None, "the wheel carries no ged_gap_mask.npz"
        assert probe["kind"] == "artifact"
        assert probe["packaged"] in probe["source"]

    def test_a_packaged_artifact_has_its_digest_pinned(self, probe):
        assert probe["packaged"] is not None
        assert probe["pinned"] is not None

    def test_the_failure_names_every_path_it_tried(self, probe):
        if probe["error"] is None:
            pytest.skip("resolution succeeded, so there is no message to check")
        message = probe["error"]
        assert "ged_gap_mask.npz" in message
        assert "landsat_lst/data/ged_gap_mask.npz" in message
        assert "aster_ged" in message
        assert "LST_GED_GAP_MASK=false" in message

    def test_the_failure_is_not_a_silent_empty_mask(self, probe):
        """The defect this whole file exists for: a VM must not proceed."""
        assert probe["kind"] != "granules" or probe["ged_dir_exists"]
