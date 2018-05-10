import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import censusgeocode as cg
from geopy.geocoders import Nominatim
geolocator = Nominatim()

data = pd.read_csv('property-assessment-fy2018.csv')
# data.ZIPCODE = data.ZIPCODE.str.replace('_', '')

# st_num = data['ST_NUM']
# st_name = data['ST_NAME']
# st_name_suf = data['ST_NAME_SUF']
# zip_code = data['ZIPCODE']

# data['ST_ADDR'] = pd.Series(st_num).str.cat([st_name, st_name_suf], sep=' ')
# st_addr = data['ST_ADDR']
# data['ST_ADDR'] = pd.Series(st_addr).str.cat([zip_code], sep=' ')

# locations = []
# for address in tqdm(st_addr):
#     for i in range(len(address)):
#         temp = cg.onelineaddress(address[i], returntype='locations')
#         #temp = temp['coordinates']
#         locations.append(temp)
# print(locations)
# #lats = []
# #lons = []
# #for i in range(len(data['ST_ADDR'])):
# #    temp = geolocator.geocode(data['ST_ADDR'][i])
# #    if temp != None:
# #        lats.append(temp.latitude)
# #        lons.append(temp.longitude)
# #data['LATITUDE'] = lats
# #data['LONGITUDE'] = lons

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



year = [2018]*len(lat)
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
data.to_csv('data2018.csv', index=False)
