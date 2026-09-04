"""Exercise every repository-owned Vale rule."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / ".vale.ini"

pytestmark = pytest.mark.skipif(
    shutil.which("vale") is None,
    reason="Vale is not installed; the prose CI job runs these tests",
)


def _words(count: int) -> str:
    return " ".join(["word"] * count) + "."


CASES = {
    "Landsat-Docs.Sentence26": (_words(27), _words(26)),
    "Landsat-Mechanics.Ellipsis": ("Wait...", "Wait."),
    "Landsat-Mechanics.EmDash": ("word—word", "word — word"),
    "Landsat-Mechanics.EmDashDensity": (
        "a — b — c — d — e",
        "a — b — c — d",
    ),
    "Landsat-Mechanics.Headings": ("# Bad Heading Here", "# Good heading"),
    "Landsat-Mechanics.Oxford": (
        "Choose red, blue and green options.",
        "Choose red, blue, and green options.",
    ),
    "Landsat-Mechanics.Quotes": ("“quoted”", '"quoted"'),
    "Landsat-Terms.Casing": ("Geotiff data.", "GeoTIFF data."),
    "Landsat-Terms.Hype": ("A seamless tool.", "A direct tool."),
    "Landsat-Voice.AffirmativeNegativeEcho": (
        "It reads the policy. It does not read the answers.",
        "Although it reads the policy, the answers remain unavailable.",
    ),
    "Landsat-Voice.ChatbotResidue": (
        "I hope this helps.",
        "The command prints the result.",
    ),
    "Landsat-Voice.ClosingTail": (
        "In conclusion, publish the files.",
        "Publish the files.",
    ),
    "Landsat-Voice.ConsequenceCadence": (
        "It is indexed, so people can search. It is open, so people can query.",
        "It is indexed, so people can search. People query the open files.",
    ),
    "Landsat-Voice.ContrastSlogan": (
        "It is not just a report, but a complete transformation.",
        "The report includes the measured results.",
    ),
    "Landsat-Voice.DramaticColon": (
        "Remember: this changes the final result.",
        "Remember that this changes the result.",
    ),
    "Landsat-Voice.Filler": ("It basically works.", "It works."),
    "Landsat-Voice.Passive": (
        "The file was written yesterday.",
        "The publisher wrote the file yesterday.",
    ),
    "Landsat-Voice.SerialListCadence": (
        "It reads red, blue, and green files. It writes one, two, and three.",
        "It reads red, blue, and green files. The output contains three files.",
    ),
    "Landsat-Voice.SoYouCan": (
        "It is open, so you can read it.",
        "Because it is open, you can read it.",
    ),
    "Landsat-Voice.StockTransitions": (
        "In today's landscape, catalogs matter.",
        "Catalogs make distributed data searchable.",
    ),
}


def _checks(tmp_path: Path, text: str, level: str = "suggestion") -> set[str]:
    target = tmp_path / "page.md"
    target.write_text(text + "\n", encoding="utf-8")
    completed = subprocess.run(
        [
            "vale",
            "--config",
            str(CONFIG),
            "--minAlertLevel",
            level,
            "--output",
            "JSON",
            target.name,
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode in {0, 1}, completed.stderr
    parsed = json.loads(completed.stdout or "{}")
    return {alert["Check"] for alerts in parsed.values() for alert in alerts}


def test_rule_inventory_matches_the_cases() -> None:
    rules = {
        f"{path.parent.name}.{path.stem}" for path in (ROOT / "styles").glob("Landsat-*/*.yml")
    }
    assert rules == set(CASES)


def test_microsoft_package_is_active(tmp_path: Path) -> None:
    assert "Microsoft.Wordiness" in _checks(tmp_path, "Tools utilize the cached file.")


def test_only_selected_readability_metrics_are_active(tmp_path: Path) -> None:
    sentence = "Administrative interoperability documentation complicates implementation."
    dense = " ".join([sentence] * 20)
    readability = {check for check in _checks(tmp_path, dense) if check.startswith("Readability.")}
    assert readability == {
        "Readability.AutomatedReadability",
        "Readability.FleschReadingEase",
    }


def test_filler_allows_a_concrete_not_just_contrast(tmp_path: Path) -> None:
    assert "Landsat-Voice.Filler" not in _checks(
        tmp_path, "The scanner checks content, not just paths."
    )


def test_dramatic_colon_does_not_treat_a_stage_label_as_prose(
    tmp_path: Path,
) -> None:
    assert "Landsat-Voice.DramaticColon" not in _checks(
        tmp_path,
        "**Stage 3: field matching.** Apply the matching rule.",
    )


@pytest.mark.parametrize(("check", "examples"), CASES.items())
def test_each_rule_reports_bad_and_accepts_good(
    tmp_path: Path, check: str, examples: tuple[str, str]
) -> None:
    bad, good = examples
    assert check in _checks(tmp_path, bad)
    assert check not in _checks(tmp_path, good)
