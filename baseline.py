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