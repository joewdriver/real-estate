###
# realty2011.py
# 
# Cleans the dataset for 2011
# 
# Outputs data2011.csv
###

import csv
import pandas as pd
import numpy as np


# Import data from file by year
data = pd.read_csv('./old_data/property-assessment-fy2011.csv')
data2015 = pd.read_csv('data2015.csv')


# Prices
bldg_price = data['AV_BLDG']
land_price = data['AV_LAND']
total_price = data['AV_TOTAL']

st_num = data['ST_NUM']
st_name = data['ST_NAME']
st_name_suf = data['ST_NAME_SUF']
zip_code = data['ZIPCODE']
land_sf = data['LAND_SF']
r_bdrms = data['R_BDRMS']
r_full_bth = data['R_FULL_BTH']
r_half_bth = data['R_HALF_BTH']
u_bdrms = data['U_BDRMS']
u_full_bth = data['U_FULL_BTH']
u_half_bth = data['U_HALF_BTH']
cm_id = data['CM_ID']
yr_built = data['YR_BUILT']

LU = data['LU']
LU_keys = ['R1', 'R2', 'R3', 'R4', 'CD']

add = []
# zc = []
lat = []
lon = []
bldg = []
land = []
sf = []
bdrms = []
fbath = []
hbath = []
res = []
condo = []
built = []



for i in range(len(LU)):
    if LU[i] not in LU_keys:
        continue
    else:

        # Match with address in 2015 dataset
        # Get data point address
        address = ''
        if type(st_name_suf[i]) == type(""):
            address = str(st_num[i]) + ' ' + st_name[i] + ' ' + st_name_suf[i] + ' ' + "0" + str(zip_code[i])[:-2]
        else:
            address = str(st_num[i]) + ' ' + st_name[i] + "0" + str(zip_code[i])[:-2]

        # Check if data point is in 2015 data
        stored = np.where(data2015["address"] == address.upper())
        if stored[0].size == 0:
            continue
        else:
            ind = stored[0][0]
            add.append(address)
            lat.append(data2015["latitude"][ind])
            lon.append(data2015["longitude"][ind])

            # No land_sf value
            if land_sf[i] is None or np.isnan(land_sf[i]):
                sf.append(data2015["square_foot"][ind])
            else:
                sf.append(float(land_sf[i]))

            if yr_built[i] is None or np.isnan(yr_built[i]):
                built.append(data2015["yr_built"][ind])
            else:
                built.append(float(yr_built[i]))

            if LU[i] == 'CD':
                condo.append(1)
                res.append(0)
                if u_bdrms[i] is not None and u_full_bth[i] is not None and u_half_bth[i] is not None:
                    if np.isnan(float(u_bdrms[i])) or np.isnan(float(u_full_bth[i])) or np.isnan(float(u_half_bth[i])):
                        bdrms.append(data2015["bedrooms"][ind])
                        fbath.append(data2015["full_bth"][ind])
                        hbath.append(data2015["half_bth"][ind])

                    else:
                        bdrms.append(float(u_bdrms[i]))
                        fbath.append(float(u_full_bth[i]))
                        hbath.append(float(u_half_bth[i]))
                else:
                    bdrms.append(data2015["bedrooms"][ind])
                    fbath.append(data2015["full_bth"][ind])
                    hbath.append(data2015["half_bth"][ind])                

            else:
                condo.append(0)
                res.append(1)

                if r_bdrms[i] is not None and r_full_bth[i] is not None and r_half_bth[i] is not None:
                    if np.isnan(float(r_bdrms[i])) or np.isnan(float(r_full_bth[i])) or np.isnan(float(r_half_bth[i])):
                        bdrms.append(data2015["bedrooms"][ind])
                        fbath.append(data2015["full_bth"][ind])
                        hbath.append(data2015["half_bth"][ind])
                    else:
                        bdrms.append(float(r_bdrms[i]))
                        fbath.append(float(r_full_bth[i]))
                        hbath.append(float(r_half_bth[i]))
                else:
                    bdrms.append(data2015["bedrooms"][ind])
                    fbath.append(data2015["full_bth"][ind])
                    hbath.append(data2015["half_bth"][ind])

            bldg.append(float(bldg_price[i]))
            land.append(float(land_price[i]))



year = [2011]*len(lat)
lat = pd.Series(lat,name="latitude")
lon = pd.Series(lon,name="longitude")
add = pd.Series(add,name="address")
year = pd.Series(year,name="year")
bdrms = pd.Series(bdrms,name="bedrooms")
fbath = pd.Series(fbath,name="full_bth")
hbath = pd.Series(hbath,name="half_bth")
sf = pd.Series(sf,name="square_foot")
res = pd.Series(res,name="res")
condo = pd.Series(condo,name="condo")
built = pd.Series(built,name="yr_built")
bldg = pd.Series(bldg,name="bldg_price")
land = pd.Series(land,name="land_price")


data = pd.concat([lat,lon,add,year,bdrms,fbath,hbath,sf,res,condo,built,bldg,land], axis = 1)
data.to_csv('./new_data/data2011.csv', index=False)