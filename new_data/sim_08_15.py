import csv
import numpy as np
import pandas as pd

data2008 = pd.read_csv('data2008.csv')
data2015 = pd.read_csv('data2015.csv')

with open('sim_08_15.csv','w') as myfile:
    wr = csv.writer(myfile,quoting=csv.QUOTE_ALL)
    for address in data2008["address"]:
        i = np.where(data2015["address"] == address)
        out = []
        header = ["latitude", "longitude","address", "year", "bedrooms", "full_bth", "half_bth", "square_foot", "res", "condo", "yr_built", "bldg_price", "land_price"]
        for head in header:
            out.append(data2015[head][i[0][0]])
        wr.writerow(out)