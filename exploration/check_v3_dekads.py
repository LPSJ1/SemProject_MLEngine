import glob
import os
import re
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.windows import from_bounds

# repo root, so it runs from any folder
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------
# PURPOSE
# Is the 0.919 match between CHIRPS v3 pentads and our HDX rainfall
# caused by the VERSION (v3 vs v2) or by a bug in my pentad code?
# Test: CHIRPS v3 also publishes dekads. v3 pentads summed into dekads
# and v3 dekads are the same product, so they should agree (>= 0.99).
# ---------------------------------------------------------------

YEAR = "2002"
URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/dekads/africa/tifs/"
dek_dir = ROOT / "data/raw/chirps_dekads" / YEAR
pen_dir = ROOT / "data/raw/chirps_pentads" / YEAR
os.makedirs(dek_dir, exist_ok=True)

# ---------- download the 36 dekad files ----------
html = requests.get(URL, timeout=60).text
names = sorted(set(f for f in re.findall(r'href="([^"]+\.tif)"', html) if f".{YEAR}." in f))
print(len(names), "dekad files for", YEAR)
for n in names:
    if (dek_dir / n).exists():
        continue
    for attempt in range(8):
        r = requests.get(URL + n, timeout=120)
        if r.status_code != 429:       # 429 = the server says "too many requests"
            break
        wait = int(r.headers.get("Retry-After", 15)) * (attempt + 1)
        print("  rate-limited, waiting", wait, "s")
        time.sleep(wait)
    r.raise_for_status()
    (dek_dir / n).write_bytes(r.content)
    time.sleep(3)      # a pause between files

# ---------- grid cells inside each sub-county (same as the pentad check) ----------
first = sorted(glob.glob(str(pen_dir / "*.tif")))[0]
with rasterio.open(first) as src:
    win = from_bounds(33.5, -5.0, 42.5, 5.5, src.transform).round_offsets().round_lengths()
    tr = src.window_transform(win)
    h, w = int(win.height), int(win.width)

a2 = gpd.read_file(ROOT / "KEN_AdminBoundaries_working.gdb", layer="ken_admbnda_adm2_iebc_20191031")
a2["geometry"] = a2.geometry.make_valid()
rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
xs, ys = rasterio.transform.xy(tr, rows.ravel(), cols.ravel(), offset="center")
cells = gpd.GeoDataFrame({"r": rows.ravel(), "c": cols.ravel()},
                         geometry=gpd.points_from_xy(xs, ys), crs="EPSG:4326")
j = gpd.sjoin(cells, a2[["admin2Pcode", "geometry"]], predicate="within")
feat = pd.read_csv(ROOT / "data/processed/features.csv", usecols=["PCODE"]).PCODE.unique()
j = j[j.admin2Pcode.isin(feat)]


def polygon_means(folder):
    # one row per (sub-county, month, number in the filename)
    out = []
    for f in sorted(glob.glob(str(folder / "*.tif"))):
        month, k = os.path.basename(f).replace(".tif", "").split(".")[-2:]
        with rasterio.open(f) as src:
            a = src.read(1, window=win).astype("float64")
        a[a < 0] = np.nan
        s = pd.Series(a[j.r.values, j.c.values]).groupby(j.admin2Pcode.values).mean()
        out += [(pc, int(month), int(k), v) for pc, v in s.items()]
    return pd.DataFrame(out, columns=["PCODE", "month", "k", "mm"])


# v3 dekads, straight from the files (k = 1, 2, 3 -> day 1, 11, 21)
dk = polygon_means(dek_dir)
dk["date"] = pd.to_datetime(dict(year=int(YEAR), month=dk.month, day=(dk.k - 1) * 10 + 1))
dk = dk.rename(columns={"mm": "v3_dekad"})[["PCODE", "date", "v3_dekad"]]

# v3 pentads summed into dekads (k = 1..6 -> dekad (k+1)//2)
pt = polygon_means(pen_dir)
pt["date"] = pd.to_datetime(dict(year=int(YEAR), month=pt.month, day=((pt.k + 1) // 2 - 1) * 10 + 1))
pt = pt.groupby(["PCODE", "date"]).mm.sum().rename("v3_pentad_sum").reset_index()

m = dk.merge(pt, on=["PCODE", "date"])
print("matched rows:", len(m))
print("correlation v3 dekad vs v3 pentads summed:", round(m.v3_dekad.corr(m.v3_pentad_sum), 4))
print("max absolute difference (mm):", round(float((m.v3_dekad - m.v3_pentad_sum).abs().max()), 3))

# and v3 dekads against our HDX rainfall, for comparison with the 0.919
rain = pd.read_csv(ROOT / "data/raw/ken-rainfall-subnat-full.csv", parse_dates=["date"])
rain = rain[(rain.adm_level == 2) & (rain.date.dt.year == int(YEAR))][["PCODE", "date", "rfh"]]
h2 = rain.merge(dk, on=["PCODE", "date"])
print("correlation v3 dekad vs HDX rfh:", round(h2.rfh.corr(h2.v3_dekad), 4))
