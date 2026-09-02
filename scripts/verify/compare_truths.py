"""Byte-level comparison of the two truth sets. Zero tolerance."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

S = Path(sys.argv[1])
ok = True
report = {}

for tile in ("N40W075", "S30W065"):
    a = np.load(S / f"main-{tile}.npz")
    b = np.load(S / f"cand-{tile}.npz")
    ma = json.loads((S / f"main-{tile}.npz.json").read_text())
    mb = json.loads((S / f"cand-{tile}.npz.json").read_text())
    row = {}
    for key, dt in (("lst", np.float32), ("qa", np.uint8), ("enc", np.uint16)):
        x, y = a[key], b[key]
        eq = np.array_equal(x, y, equal_nan=True)
        same_dtype = x.dtype == y.dtype == dt
        same_shape = x.shape == y.shape
        h = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        h2 = hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()
        row[key] = {
            "array_equal": bool(eq),
            "dtype_ok": bool(same_dtype),
            "dtype": str(x.dtype),
            "shape": list(x.shape),
            "shape_ok": bool(same_shape),
            "sha_main": h,
            "sha_cand": h2,
            "sha_match": h == h2,
        }
        ok &= eq and same_dtype and same_shape and h == h2
    # masks, position by position
    nan_a, nan_b = np.isnan(a["lst"]), np.isnan(b["lst"])
    nod_a, nod_b = a["lst"] == -9999.0, b["lst"] == -9999.0
    row["nan_mask_identical"] = bool(np.array_equal(nan_a, nan_b))
    row["nodata_mask_identical"] = bool(np.array_equal(nod_a, nod_b))
    row["nodata_px"] = int(nod_a.sum())
    row["qa_zero_mask_identical"] = bool(np.array_equal(a["qa"] == 0, b["qa"] == 0))
    ok &= row["nan_mask_identical"] and row["nodata_mask_identical"]
    ok &= row["qa_zero_mask_identical"]
    # dims / coords, from the sidecars
    row["qa_dims_main"] = ma["qa_dims"]
    row["qa_dims_cand"] = mb["qa_dims"]
    row["qa_dims_ok"] = mb["qa_dims"] == ["month", "latitude", "longitude"]
    row["lst_dims_ok"] = mb["lst_dims"] == ma["lst_dims"] == ["latitude", "longitude"]
    row["month_coord_ok"] = mb["month_coord"] == ma["month_coord"] == list(range(1, 13))
    row["lat_sha_match"] = ma["lat_sha"] == mb["lat_sha"]
    row["lon_sha_match"] = ma["lon_sha"] == mb["lon_sha"]
    row["time_sha_match"] = ma["time_sha"] == mb["time_sha"]
    for k in (
        "qa_dims_ok",
        "lst_dims_ok",
        "month_coord_ok",
        "lat_sha_match",
        "lon_sha_match",
        "time_sha_match",
    ):
        ok &= row[k]
    report[tile] = row

print(json.dumps(report, indent=2))
print("OVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
