import os
import re
from pathlib import Path

import requests

# repo root, so the files land in data/raw/ wherever this is run from
ROOT = Path(__file__).resolve().parent.parent

# CHIRPS v3.0 pentads (5-day totals) for Africa. 6 pentads a month, and
# pentads 1+2, 3+4, 5+6 are exactly the three dekads of the month.
URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/pentads/africa/tifs/"
YEAR = "2002"

out = ROOT / "data/raw/chirps_pentads" / YEAR
os.makedirs(out, exist_ok=True)

html = requests.get(URL, timeout=60).text
files = sorted(set(f for f in re.findall(r'href="([^"]+\.tif(?:\.gz)?)"', html)
                   if f".{YEAR}." in f))
print(len(files), "files for", YEAR, "| e.g.", files[:2])   # expect 72
if not files:
    raise SystemExit("no files found: check the URL in a browser")

for f in files:
    name = os.path.basename(f)
    if not (out / name).exists():
        r = requests.get(URL + name, timeout=120)
        r.raise_for_status()
        (out / name).write_bytes(r.content)
print("done")
