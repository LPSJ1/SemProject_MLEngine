import re
import pandas as pd

#_______________________________________________________________________________
#Purpose of this script is as follows:
#the 2 sources naming conventions are messed up in free text
#("TANA-RIVER" OR "Kisumu districts or provinces like nyanza or even mispellings like tarkana")
#Our rainfall data uses official codes like KE042 = Kisumu
#So this script is to ubild a bridge in that any place name -> county code
#Exit test is every name either matches or is intentionally ignored
#________________________________________________________________________________


#official IEBC 2019 Boundaries (extraccted from the OCHA geodatabase)
counties = pd.read_csv("data/kenya_counties.csv")  #47 counties
subs = pd.read_csv("data/kenya_subcounties.csv")    #290 sub-counties


def norm(name):
    # Reduce a name to lowercase letters only, so that
    #Homa Bay", "HOMABAY" and "Homa0Bay" all resolve as "homabay".
    #This basically fixes most spelling diffs between the sources
    return re.sub(r"[^a-z]", "", str(name).lower())


#county name -> county code, e.g. "kisumu" -> "KE042"
county_code = {norm(n): c for n, c in zip(counties.admin1Name_en, counties.admin1Pcode)}

#sub-county name -> its parent couonty code e.g "kinango" -> "KE002" (Kwale).
#it lets us place records that name a subcounty instead of a county
sub_to_county = {norm(n): c for n, c in zip(subs.admin2Name_en, subs.admin1Pcode)}

#Names that normalising can't fix: old pre-2013 district names, towns,
#rivers and a typo, each map to the county it falls in.

ALIASES = {
    #Elgeyo-Marakwet was two districts(Kiyo, Matakwet) before 2013
    "keiyo": "KE028", "marakwet": "KE028", "elgeyo":"KE028", "keiyomarakwet": "KE028",
    #Turkana: typo in EM-DAT, plus towns within the county
    "tarkana": "KE023", "lodwar": "KE023", "katilu" : "KE023",
    "rongai": "KE032", "naivasha": "KE032",                        #Nakuru
    "thika": "KE022",                                              #Kiambu
    "maragua": "KE021",                                            #Murang'a
    "nyando": "KE042", "kisimu": "KE042",                          #Kisumu
    "rachuonyo": "KE043", "suba": "KE043",                         #Homa Bay
    "koibatek": "KE030", "marigat": "KE030", "salabani": "KE030",  #Baringo
    "mtelgon": "KE039",                                            #Bungoma
    "malindi" : "KE003",                                           #Kilifi
    "moyale" : "KE010", "chalbi": "KE010",                         #Marsabit
    "garsen" : "KE004", "bura": "KE004", "madogo": "KE004",
    "bangale": "KE004", "tanariverdelta": "KE004",                 #Tana River
    "ijara" : "KE007", "dadaab" : "KE007",                         #Garissa
    "taveta": "KE006",                                             #Taita Taveta
    "kitengela": "KE034", "ngong": "KE034",                        #Kajiado
    "centralkisii":"KE045",                                        #Kisii
    "nandinorth": "KE029", "nandisouth": "KE029",                  #Nandi
    "athiriver": "KE016",
}

#Pre-2013 PRONVINCES. Each covers many counties("Rift Valley" = 14),
#so we cant tell which county flooded. DECISION: ignore province-only
#mentions rather than label every county inside them- labelling all
#14 Rift Vallry counties would invent far more floods than happened.
PROVINCES = {"coast", "coastal", "nyanza", "western", "riftvalley", "central",
              "eastern", "northeastern", "northwestern", "nairobi"}


def to_pcode(name):
    #Tries in the order of known alias, county name and sub-county name
    key = norm(name)
    if key in ALIASES:
        return ALIASES[key]
    if key in county_code:
        return county_code[key]
    return sub_to_county.get(key) #if nothing matches


#Filler word in EM-DAT's location text that aren't part of any name
JUNK = (r"\b(counties|county|near|region|regions|and|the|areas?|districts?|provinces?|"
        r"town|city|villages?|camp)\b")

def clean(part):
    return re.sub(JUNK, "", part, flags=re.I).strip(" ()")


def counties_in(location):
    #Turn one EM-DAT location string into a set of county codes.
    #an example is "Athi River (Machakos" -> tries "Athi River" AND "Machakos",
    #since older records put the county inside brackets.
    found, unmatched = set(), []
    for part in re.split(r"[,;/]| and ", location):
        pieces = [p for p in part.split("(") if clean(p)]
        codes = {to_pcode(clean(p)) for p in pieces} - {None}
        if codes :
            found |= codes
        elif norm (clean(part.split("(")[0])) not in PROVINCES and clean(part):
            unmatched.append(clean(part))
    return found, unmatched


if __name__ == "__main__":
    # Only runs when this file is executed directly,
    # NOT when make_labels.py imports functions from it.

    #---checking source 1, DesInventar (1 county per row)
    di = pd.read_csv("data/DesInventar_Geocoded.csv", low_memory=False)
    di = di[di.Country == "Kenya"]
    print("DesInventar unmatched:", sorted({n for n in di.Adm1.dropna() if to_pcode(n) is None}))

    #---checking source 2: EM-DAT (free-text list of places)
    em = pd.read_csv("data/EMDAT_Geocoded.csv", low_memory=False)
    em = em[(em.iso == "KEN") & (em["start year"] >= 2002)] #the labelled window

    bad = set()
    for loc in em.location.dropna():
        bad |= set(counties_in(loc)[1])
    print("EM-DAT unmatched names:", sorted(bad))

    #Events where NO county could be identified at all, they'll be lost
    lost = [(d, l) for d, l in zip(em["disno."], em.location)
            if isinstance(l, str) and not counties_in(l)[0]]
    print("EM-DAT events with no county found:", lost)
    print("EM-DAT events with no location at all:", em.location.isna().sum())