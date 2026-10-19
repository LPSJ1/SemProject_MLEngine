import pandas as pd

df = pd.read_csv("data/ken-rainfall-subnat-full.csv")
nbi = df[(df.PCODE =="KE047") & (df.date >= "2024-03-01") & (df.date <= "2024-06-01")]
print(nbi[["date", "rfh", "rfh_avg", "rfq"]].to_string(index=False))