#!/usr/bin/env python3
"""Rebuild a read trace's summary from its retained raw log.

The raw log is the evidence; the summary is derived from it. Keeping this
script means a reviewer can check the summary, or compute a different one, from
a retained ``readtrace.jsonl.gz`` without paying for a second cloud run.

Usage::

    python scripts/summarize_read_trace.py path/to/readtrace.jsonl.gz
    python scripts/summarize_read_trace.py trace.jsonl.gz --out summary.json

See issue #135 and :mod:`landsat_lst.readtrace`.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landsat_lst.readtrace import Record, summarize


def load(path: Path) -> tuple[list[Record], dict[str, object]]:
    """Read the raw log back into the tuples :func:`summarize` expects."""
    opener = gzip.open if path.suffix == ".gz" else open
    records: list[Record] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            entry = json.loads(line)
            records.append(
                (
                    float(entry["t"]),
                    int(entry["thread"]),
                    str(entry["kind"]),
                    str(entry["url"]),
                    entry.get("start"),
                    entry.get("end"),
                    entry.get("bytes"),
                )
            )
    return records, {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="a retained readtrace.jsonl.gz")
    parser.add_argument("--out", type=Path, default=None, help="write here instead of stdout")
    args = parser.parse_args()

    records, context = load(args.log)
    summary = summarize(records, context=context)
    text = json.dumps(summary, indent=2)
    if args.out is None:
        print(text)
    else:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
