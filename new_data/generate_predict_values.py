# Pulls specific addresses to be predicted on for 2019, 2020, and 2021

import csv
import pandas as pd
import numpy as np

data2018 = pd.read_csv('data2018.csv')
def write_property(address,wr,data = data2018):
    i = np.where(data["address"] == address)
    out = []
    header = ["latitude", "longitude","address", "year", "bedrooms", "full_bth", "half_bth", "square_foot", "res", "condo", "yr_built"]
    for head in header:
        out.append(data[head][i[0][0]])

    # Get desired years
    out[3]=2019
    wr.writerow(out)
    out[3]=2020
    wr.writerow(out)
    out[3]=2021
    wr.writerow(out)

with open('future_values.csv','w') as myfile:
    wr = csv.writer(myfile,quoting=csv.QUOTE_ALL)
    wr.writerow(["latitude", "longitude","address", "year", "bedrooms", "full_bth", "half_bth", "square_foot", "res", "condo", "yr_built"])
    
    write_property("257 255 EVERETT ST 02128",wr)
    write_property("18 R POLK ST 02129",wr)
    write_property("81 83 N MARGIN ST 02113",wr)
    write_property("536 538 COMMERCIAL ST 02109",wr)
    write_property("28 32 ATLANTIC AV 02110",wr)
    write_property("234 236 JAMAICAWAY ST 02130",wr)
    write_property("132 134 HOMESTEAD ST 02121",wr)
    write_property("25 R MAYFIELD ST 02125",wr)
    write_property("325 327 SAVIN HILL AV 02125",wr)
    write_property("86 88 BERNARD ST 02124",wr)
    write_property("47 49 CLARKWOOD ST 02126",wr)
    write_property("72 74 WELLSMERE RD 02131",wr)
