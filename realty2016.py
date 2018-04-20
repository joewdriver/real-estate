import pandas as pd

data = pd.read_csv('property-assessment-fy2016.csv')
lats = data['LATITUDE']
lons = data['LONGITUDE']