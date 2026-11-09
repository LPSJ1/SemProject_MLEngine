#This was to check whether the data shows the March-May 2024 Nairobi floods that we know happened
#It prints out mar-jun dekads in(mm) in Nairobi, county code(KE047) for this dekad in(mm)
# rfh         rainfall in this dekad(mm)
# rfh_avg     the average of that dekad against historic dekads in that specific location
# rfq         the anomaly of the rain represented as a percentage
import pandas as pd

df = pd.read_csv("data/ken-rainfall-subnat-full.csv")
nbi = df[(df.PCODE =="KE047") & (df.date >= "2024-03-01") & (df.date <= "2024-06-01")]
print(nbi[["date", "rfh", "rfh_avg", "rfq"]].to_string(index=False))