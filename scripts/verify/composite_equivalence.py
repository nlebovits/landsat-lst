"""Independent equivalence harness for the composite layout change.

Written by the verifier, not the implementer. It does not read
``results/perf/truth-*.npz`` and shares no code with
``scripts/perf/composite_experiment.py``.

One child process per (checkout, fixture). It builds the composite through the
public ``compute_annual_composite`` entry point with de-striping off -- the
change under test lives strictly downstream of the offset estimator, and
disabling it removes the only non-deterministic, network-dependent input --
then writes every array the contract names, plus its sha256 and its dimension
order, to an ``.npz`` under the scratch directory.

Usage: python verify_composite_equivalence.py <fixture-name> <out.npz>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import dask
import numpy as np

from landsat_lst.config import settings
from landsat_lst.encoding import encode_lst_uint16
from landsat_lst.fixture import FixtureSpec, load_fixture
from landsat_lst.pipeline import compute_annual_composite


def sha(a: np.ndarray) -> str:
    """sha256 over the C-contiguous bytes, so layout cannot smuggle a match."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> int:
    name, out = sys.argv[1], Path(sys.argv[2])

    # Bounded: 3 threads over 512-px blocks. The machine is shared.
    dask.config.set(scheduler="threads", num_workers=3)
    settings.destripe = False
    settings.load_chunk_size = 512

    tile, window, scenes, factor = name.split("_", 3)
    year, end_year = window.split("-")
    spec = FixtureSpec(
        tile=tile,
        year=int(year),
        end_year=int(end_year),
        max_scenes=int(scenes.removeprefix("n")),
        factor=int(factor.removeprefix("f")),
    )

    data = load_fixture(spec)
    result = compute_annual_composite(data)

    lst = result["lst_p95"]
    qa = result["qa_count"]
    enc = encode_lst_uint16(lst)

    lst_v = lst.compute().values
    qa_v = qa.compute().values
    enc_v = enc.compute().values

    meta = {
        "fixture": name,
        "lst_dims": list(lst.dims),
        "lst_dtype": str(lst_v.dtype),
        "lst_shape": list(lst_v.shape),
        "lst_sha": sha(lst_v),
        "qa_dims": list(qa.dims),
        "qa_dtype": str(qa_v.dtype),
        "qa_shape": list(qa_v.shape),
        "qa_sha": sha(qa_v),
        "enc_dtype": str(enc_v.dtype),
        "enc_sha": sha(enc_v),
        "month_coord": [int(m) for m in qa["month"].values],
        "month_sha": sha(np.asarray(qa["month"].values)),
        "lat_sha": sha(np.asarray(lst["latitude"].values)),
        "lon_sha": sha(np.asarray(lst["longitude"].values)),
        "time_values": [str(t) for t in data["time"].values[:5]],
        "time_sha": sha(np.asarray(data["time"].values)),
        "nan_count": int(np.isnan(lst_v).sum()),
        "nodata_count": int((lst_v == settings.nodata).sum()),
        "nan_mask_sha": sha(np.isnan(lst_v)),
        "nodata_mask_sha": sha(lst_v == settings.nodata),
        "qa_zero_mask_sha": sha(qa_v == 0),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, lst=lst_v, qa=qa_v, enc=enc_v)
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
