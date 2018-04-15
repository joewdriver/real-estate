import pandas as pd
from geopy.geocoders import Nominatim
geolocator = Nominatim()

data = pd.read_csv('property-assessment-fy2017.csv')

st_num = data['ST_NUM']
st_name = data['ST_NAME']
st_name_suf = data['ST_NAME_SUF']

data['ST_ADDR'] = pd.Series(data['ST_NUM']).str.cat([data['ST_NAME'], data['ST_NAME_SUF']], sep=' ')