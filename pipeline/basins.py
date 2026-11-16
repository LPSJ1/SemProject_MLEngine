import pandas as pd
import geopandas as gpd
from pathlib import Path

#repo root, so script works from any folder
ROOT = Path(__file__).resolve().parent.parent

SUBCOUNTIES = ROOT / "data/raw/kenya_subcounties.csv"
FEATURES_CSV = ROOT /"data/processed/features.csv"
BASINS = ROOT / "data/raw/hydrobasins/hybas_af_lev07_v1c.shp"
OUT = ROOT / "data/processed/subcounty_basins.csv"

 #on;y the 73 sub-countoes that are in our modelling table
sampled = pd.read_csv(FEATURES_CSV, usecols=["PCODE"]).PCODE.unique()

subs = pd.read_csv(SUBCOUNTIES)
subs = subs[subs.admin2Pcode.isin(sampled)]

#turn the lon/lat columns into map points
points = gpd.GeoDataFrame(
    subs,
    geometry=gpd.points_from_xy(subs.POINT_X, subs.POINT_Y),
    crs="EPSG:4326",
)
print("sampled sub-counties:", len(points))    

#no bbox = here, the .sbn index in this download is corrupt anf crashes it
all_basins = gpd.read_file(BASINS)
basins = gpd.read_file(BASINS). cx[32:43, -6:6]
print("basins polygons near Kenya:", len(basins))

#finding each point's basin
joined = gpd.sjoin(
    points,
    basins[["HYBAS_ID", "NEXT_DOWN", "MAIN_BAS", "geometry"]],
    how="left",
    predicate="within",
)
print("matched to a basin:", joined.HYBAS_ID.notna().sum(), "of", len(points))

#polygon -> the polygon it drains into (0 means it reaches the sea or a sink)
next_down = dict(zip(all_basins.HYBAS_ID, all_basins.NEXT_DOWN))

def downstream_chain(hybas_id):
    #every polygon the water passes through, starting with this one
    chain = []
    while hybas_id in next_down and hybas_id not in chain:
        chain.append(hybas_id)
        hybas_id = next_down[hybas_id]
    return chain

#working out who is upstream of whom
poly_of = dict(zip(joined.admin2Pcode, joined.HYBAS_ID))
chain_of = {pc: downstream_chain(h) for pc, h in poly_of.items()}

upstream = {}
for pc, hid in poly_of.items():
    ups = [other for other, other_hid in poly_of.items()
           if other != pc and other_hid != hid and hid in chain_of[other]]
    upstream[pc] = ";".join(sorted(ups))

out = joined[["admin2Pcode", "HYBAS_ID", "NEXT_DOWN", "MAIN_BAS"]].rename(
    columns={"admin2Pcode": "PCODE"})
out["upstream"] = out.PCODE.map(upstream)
out.to_csv(OUT, index=False)

print("rows saved:", len(out))
print("sub-counties with at least one upstream sample:", (out.upstream != "").sum(), "of", len(out))
print("Garsen's upstream:", out.loc[out.PCODE == "KE004018", "upstream"].iloc[0])