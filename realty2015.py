import csv
import pandas as pd
import numpy as np


# Import data from file by year
data = pd.read_csv('property-assessment-fy2015.csv')

# TODO: Need to filter all data

# Values to predict on
locations = data['Location']

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


# Find the location of a specific condo id
condo_loc = {}
for i in range(len(cm_id)):
    if LU[i] == 'CM':
        if type(locations[i]) == type(''):
            condo_loc[cm_id[i]] = locations[i]
    else:
        continue


for i in range(len(LU)):

    if LU[i] not in LU_keys:
        continue
    else:
        cur_lat = 0
        cur_lon = 0
        # The only locations that aren't filled in are condos. If it isn't filled, get the location 
        # for the condo from the dictionary.
        if type(locations[i]) == type(0.0) or locations[i] == "0" or locations[i] == "(0.000000000| 0.000000000)":
            if cm_id[i] is not None or np.isnan(cm_id[i]):
                if cm_id[i] in condo_loc.keys():
                    a,b = condo_loc[cm_id[i]].split("|")
                    cur_lat = float(a[1:])
                    cur_lon = float(b[:-1])
                    if cur_lat == 0 or cur_lon == 0:
                        continue
                else:
                    continue
            else:
                continue

        # The location is filled.
        else:
            a,b = locations[i].split("|")
            cur_lat = float(a[1:])
            cur_lon = float(b[:-1])
            if cur_lat == 0 or cur_lon == 0:
                continue

        # No land_sf value
        if land_sf[i] is None or np.isnan(land_sf[i]):
            continue

        if yr_built[i] is None or np.isnan(yr_built[i]):
            continue


        # Condo and residential bedrooms/bathrooms stored separately initially
        # Store if condo or residential
        if LU[i] == 'CD':
            # Make sure a value for bathrooms and bedrooms is provided otherwise pass
            if u_bdrms[i] is not None and u_full_bth[i] is not None and u_half_bth[i] is not None:
                if np.isnan(float(u_bdrms[i])) or np.isnan(float(u_full_bth[i])) or np.isnan(float(u_half_bth[i])):
                    continue
                else:
                    bdrms.append(float(u_bdrms[i]))
                    fbath.append(float(u_full_bth[i]))
                    hbath.append(float(u_half_bth[i]))
                    condo.append(1)
                    res.append(0)
            else:
                continue
        else:
            # Make sure a value for bathrooms and bedrooms is provided otherwise pass
            if r_bdrms[i] is not None and r_full_bth[i] is not None and r_half_bth[i] is not None:
                if np.isnan(float(r_bdrms[i])) or np.isnan(float(r_full_bth[i])) or np.isnan(float(r_half_bth[i])):
                    continue
                else:
                    bdrms.append(float(r_bdrms[i]))
                    fbath.append(float(r_full_bth[i]))
                    hbath.append(float(r_half_bth[i]))
                    condo.append(0)
                    res.append(1)
            else:
                continue
        if type(st_name_suf[i]) == type(""):
            add.append((str(st_num[i]) + ' ' + st_name[i] + ' ' + st_name_suf[i] + ' ' + str(zip_code[i][:-1])).upper())
        else:
            add.append((str(st_num[i]) + ' ' + st_name[i] + str(zip_code[i][:-1])).upper())    
        # zc.append(zip_code[i][:-1])
        bldg.append(float(bldg_price[i]))
        land.append(float(land_price[i]))
        sf.append(float(land_sf[i]))
        built.append(float(yr_built[i]))
        lat.append(cur_lat)
        lon.append(cur_lon)

year = [2015]*len(lat)
lat = pd.Series(lat,name="latitude")
lon = pd.Series(lon,name="longitude")
add = pd.Series(add,name="address")
# zc = pd.Series(zc,name="zipcode")
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
data.to_csv('data2015.csv', index=False)