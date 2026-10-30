import calendar
import pandas as pd
from county_match import to_pcode, counties_in


#-------------------------------------------------------------
#PURPOSE
#Turn flood RECORDS ("KKisumu flooded 12-20 April 2013") into a
#LABEL for every (sub-county, dekad) row in the rainfall data:
#  flooded = 1 if any source reports a flood touching that dekad
#  flooded = 0 otherwise
#Output: data/labels.csv, one row per sub-county per dekad.
#----------------------------------------------------------------


#labelled window: our flood sources are reliable from 2002
#and EM-DAT's coverage of Kenya ends in Dec-2024
WINDOW_START = pd.Timestamp("2002-01-01")
WINDOW_END = pd.Timestamp("2024-12-31")

#Some records span 60-120 dats - that describes a whole rainsy season.
#not 4 months of continuous flooding. Cap how long one record can
#label fo. None = no cap. (Decision = see the cap comparison printed below)
MAX_EVENT_DAYS = 60

#One EM-DAT event has no end month. DFO records the same event
#(DFO #5477) as ending 11 Nov 2023, so we borrow that date.
END_OVERRIDES = {"2023-0721-KEN": pd.Timestamp("2023-11-11")}


#-----dekad grid from rainfall data-------
rain = pd.read_csv("data/ken-rainfall-subnat-full.csv", parse_dates=["date"])
rain = rain[(rain.adm_level == 2)                       #sub-counties only
            & rain.date.between(WINDOW_START, WINDOW_END)]
dekads = sorted(rain.date.unique())
subcounties = sorted(rain.PCODE.unique())


def dekad_start(ts):
    #Snap any date to the start of its dekad: 1st, 11th or 21st.
    return ts.replace(day=1 if ts.day <= 10 else 11 if ts.day <= 20 else 21)


def month_end(year, month):
    return pd.Timestamp(year, month, calendar.monthrange(year, month)[1])


def month_end(year, month):
    return pd.Timestamp(year, month, calendar.monthrange(year, month)[1])


#Each record becomes: (county code, start date, end date, source)
records = []

# ---------- Source 1: DesInventar - one county, one date per row ----------
di = pd.read_csv("data/DesInventar_Geocoded.csv", low_memory=False)
di = di[di.Country == "Kenya"]
for _, r in di.iterrows():
    county = to_pcode(r.Adm1)
    date = pd.to_datetime(r.Date, errors="coerce")
    if pd.isna(date):
        # One record reads "2016/03/00" - day unknown, so it covers the month
        y, m, _ = map(int, str(r.Date).split("/"))
        start, end = pd.Timestamp(y, m, 1), month_end(y, m)
    else:
        start = date
        end = date + pd.Timedelta(days=int(r["Duration (d)"]))
    records.append((county, start, end, "DesInventar"))
 
# ---------- Source 2: EM-DAT - several counties per event ----------
em = pd.read_csv("data/EMDAT_Geocoded.csv", low_memory=False)
em = em[(em.iso == "KEN") & (em["start year"] >= 2002)].dropna(subset=["location"])
for _, r in em.iterrows():
    sy, sm = int(r["start year"]), int(r["start month"])
    # Missing day -> assume the event spans the whole month (2 events)
    start = pd.Timestamp(sy, sm, int(r["start day"]) if pd.notna(r["start day"]) else 1)
    if r["disno."] in END_OVERRIDES:
        end = END_OVERRIDES[r["disno."]]
    else:
        ey, em_ = int(r["end year"]), int(r["end month"])
        end = (pd.Timestamp(ey, em_, int(r["end day"])) if pd.notna(r["end day"])
               else month_end(ey, em_))
    for county in counties_in(r.location)[0]:
        records.append((county, start, end, "EM-DAT"))
 
events = pd.DataFrame(records, columns=["county", "start", "end", "source"])
events["days"] = (events.end - events.start).dt.days
 
 
def build_labels(cap):
    # Expand every record into (sub-county, dekad) pairs it touches.
    hits = []
    for e in events.itertuples():
        end = e.end if cap is None else min(e.end, e.start + pd.Timedelta(days=cap))
        first, last = dekad_start(e.start), dekad_start(end)
        touched = [d for d in dekads if first <= d <= last]
        # A county-level record labels EVERY sampled sub-county in that county
        for pcode in subcounties:
            if pcode[:5] == e.county:
                hits += [(pcode, d, e.source) for d in touched]
    hits = pd.DataFrame(hits, columns=["PCODE", "date", "source"])
 
    # Union across sources: one flood row per (sub-county, dekad), however
    # many records mention it. This is what makes duplicates harmless.
    pos = (hits.groupby(["PCODE", "date"])["source"]
               .agg(lambda s: ";".join(sorted(set(s)))).reset_index())
 
    grid = pd.MultiIndex.from_product([subcounties, dekads],
                                      names=["PCODE", "date"]).to_frame(index=False)
    labels = grid.merge(pos, on=["PCODE", "date"], how="left")
    labels["flooded"] = labels.source.notna().astype(int)
    return labels
 
 
# ---------- Decide the cap with evidence, not a guess ----------
print("records:", len(events), "| by source:", events.source.value_counts().to_dict())
print("records longer than 30 days:", (events.days > 30).sum())
for cap in [None, 60, 30]:
    lab = build_labels(cap)
    print(f"cap={cap}: {lab.flooded.sum()} positive rows of {len(lab)} "
          f"({100 * lab.flooded.mean():.2f}%)")
 
labels = build_labels(MAX_EVENT_DAYS)
labels.to_csv("data/labels.csv", index=False)
print("\nsaved data/labels.csv with cap =", MAX_EVENT_DAYS)
 
# ---------- Sanity check: MAM 2024 in Nairobi (Lang'ata) ----------
check = labels[(labels.PCODE == "KE047277")
               & labels.date.between("2024-03-01", "2024-05-31")]
print("\nNairobi MAM 2024:")
print(check.to_string(index=False))
