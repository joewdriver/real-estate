import pandas as pd
from tqdm import tqdm
import censusgeocode as cg
from geopy.geocoders import Nominatim
geolocator = Nominatim()

data = pd.read_csv('property-assessment-fy2019.csv')
data.ZIPCODE = data.ZIPCODE.str.replace('_', '')

st_num = data['ST_NUM']
st_name = data['ST_NAME']
st_name_suf = data['ST_NAME_SUF']
zip_code = data['ZIPCODE']

data['ST_ADDR'] = pd.Series(st_num).str.cat([st_name, st_name_suf], sep=' ')
st_addr = data['ST_ADDR']
data['ST_ADDR'] = pd.Series(st_addr).str.cat([zip_code], sep=' ')

locations = []
for address in tqdm(st_addr):
    for i in range(len(address)):
        temp = cg.onelineaddress(address[i], returntype='locations')
        #temp = temp['coordinates']
        locations.append(temp)
print(locations)
#lats = []
#lons = []
#for i in range(len(data['ST_ADDR'])):
#    temp = geolocator.geocode(data['ST_ADDR'][i])
#    if temp != None:
#        lats.append(temp.latitude)
#        lons.append(temp.longitude)
#data['LATITUDE'] = lats
#data['LONGITUDE'] = lons