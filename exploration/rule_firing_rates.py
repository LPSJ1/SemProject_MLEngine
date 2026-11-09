#tests the 50mm rule that KMD currently uses to choose
#what the model's baseline must beat
#the real rule is 50mm in 24hours, but the data is in 10-day totals,
#we can't reproduce it exactly, so we use three stand-ins:
#  rule_50mm   rfh >= 50       at least 50mm fell this dekad
#  rule_anom   rfq >= 200      at least double the normal rain(an assumption)
#  rule_both   both at once    heavy(50mm) AND unusual(anom) for this time of the year

#it then prints every dekad of 2024 and which ruled fired
#how much each rule fires across the whole record:
#50mm 221/1648(13.4%), anomaly 131(7.9%), both 95 (5.8%).



#what we learned:
# -50mm rule fires in every normal rainy season, its a loose.
# -the anomaly rule fires in dry months on small amounts
#  (on July 2024: 20mm  was 294% of normal, this is misleading lol) 
# -Combining both keeps only rain that is heavy AND unusual.
#we keep the two baselines, plain and combined, and report both, so
#its not seen as picking an easy one to beat
import pandas as pd

df = pd.read_csv("data/ken-rainfall-subnat-full.csv")
nbi = df[df.PCODE == "KE047"].copy()

nbi["rule_50mm"] = nbi["rfh"] >= 50
nbi["rule_anom"] = nbi["rfq"] >= 200
nbi["rule_both"] = nbi["rule_50mm"] & nbi["rule_anom"]

recent = nbi[(nbi.date >= "2024-01-01") & (nbi.date <= "2024-12-31")]
print(recent[["date", "rfh", "rfq", "rule_50mm", "rule_anom", "rule_both"]].to_string(index=False))

print("\n50mm rule fires:", nbi["rule_50mm"].sum(), "of", len(nbi), "dekads")
print("anomaly rule fires:", nbi["rule_anom"].sum(), "of", len(nbi), "dekads" )
print("combined rule fires:", nbi["rule_both"].sum(), "of", len(nbi), "dekads")