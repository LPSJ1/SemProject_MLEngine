import pandas as pd
#________________________________________________________________________________________________________________________
#once again using the CHIRPS rainfall indicator by the UN
#ONE ROW = 1 admin unit for one DEKAD(a fixed 10 day reporting period, about 3 in a month)
#uses 10 day window no t a single day, this matters cause flooding is in hours but the finest time reolution is 10 days
#so ideally cant say which day but rather which 10 day window is at risk
#________________________________________________________________________________________________________________________


df = pd.read_csv("data/ken-rainfall-subnat-full.csv")


#________________________________________________________________________________________________________________________
#adm_level will tell us how coarse each row is:
#   level 1 = county     (Kenya has 47)
#   level 2 = sub-county (Kenya has roughly 290)
#Both levels sit in the same file stacked on top of each other.
#the row and its sub-county row can describe the same rain.
#
#PCODE is the unit's "place code", structures as:
#   KE047    -> country KE, county 047 (Nairobi)
#   KE047277 -> country KE, county 047, sub-county 277
#So the first 5 characters of ANY code tell us its county.
#_________________________________________________________________________________________________________________________


county_units    = df[df.adm_level == 1]["PCODE"].unique() #county-level units
subcounty_units = df[df.adm_level == 2]["PCODE"].unique() #sub-county-level units

print("level-1 units:", len(county_units))
print(sorted(county_units))
print("\nlevel-2 units:", len(subcounty_units))


#___________________________________________________________________________________________________
#coverage check,
# Kenya has 47 counties and this file doesn't enirely cover that
# #this is a partial export and every label we build on it would be incomplete
# slicing the county prefix off each sub-county code will tell us how many counties are represented,
#even where no county-level row exists.
#___________________________________________________________________________________________________
counties = sorted({p[:5] for p in subcounty_units}) #set comprehension for unique prefixes
print("\ndistinct counties appearing in level-2 codes:", len(counties))
print(counties)

#_____________________________________________________________________________________
#How many sub-counties we actually have per county. If this is
#1-2 each rather than all of them we have a sample of each county rather than 
#full coverage which is a limitation
#______________________________________________________________________________________

per = pd.Series([p[:5] for p in subcounty_units]).value_counts().sort_index()
print("\nsub-counties per county:")
print(per.to_string())


#______________________________________________________________________________________________________________
#The 8 counties have an official county-level row and sub-county row
#if aggregating sub-counties reproduces the official figure our aggregation method can be considered valid for
#the other 39 counties.
#_______________________________________________________________________________________________________________


for code in county_units:
    official = df[(df.PCODE == code)][["date", "rfh"]].set_index("date")["rfh"]
    subs = df[(df.adm_level == 2) & (df.PCODE.str.startswith(code))]
    derived = subs.groupby("date")["rfh"].mean()
    diff = (official - derived).abs()
    print(code, "| n_subs:", subs.PCODE.nunique(),
          "| mean abs diff:", round(diff.mean(), 3),
          "| max:", round(diff.max(), 3))


#____________________________________________________________________
# a plain mean treats a small sub-county and a huge one equally.
#n-pixels is how many CHIRPS grid cells each unit covers, so it
#stands in for area where weighting by it should track the official
#county figure more closely
#____________________________________________________________________


for code in county_units:
    official = df[df.PCODE == code][["date", "rfh"]].set_index("date")["rfh"]
    subs = df[(df.adm_level == 2) & (df.PCODE.str.startswith(code))].copy()
    subs["weighted"] = subs["rfh"] * subs["n_pixels"]
    agg = subs.groupby("date").apply(
        lambda g: g["weighted"].sum() / g["n_pixels"].sum()
    )
    diff = (official - agg).abs()
    print(code,"| weighted mean abs diff:", round(diff.mean(), 3),
          "| max:", round(diff.max(), 3) )