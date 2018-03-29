import csv
import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
import sklearn.learning_curve as curves
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
# from neupy import environment

data = pd.read_csv('property-assessment-fy2015.csv')


# TODO: Need to filter all data

locations = data['Location']

bldg_price = data[['AV_BLDG']].copy()
land_price = data[['AV_LAND']].copy()
total_price = data[['AV_TOTAL']].copy()

lat = []
lon = []
prev = 0
for i in range(len(locations)):

    # TODO: TEMP FIX, no lat/lon.
    if type(locations[i]) == type(0.0) or locations[i] == "0":
        a,b = locations[prev].split("|")
        lat.append(a[1:])
        lon.append(b[:-1])
    else:
        a,b = locations[i].split("|")
        lat.append(a[1:])
        lon.append(b[:-1])
        prev = i

lat = pd.Series(lat,name="latitute")
lon = pd.Series(lon,name="longitude")

data = pd.concat([lat,lon,bldg_price,land_price,total_price])



data_scaler = preprocessing.MinMaxScaler()
# # target_scaler = preprocessing.MinMaxScaler()


# TODO: Check after preprocessing data. Running into NaN or infinities here. Likely from missing values
data = data_scaler.fit_transform(data.values)
# target = target_scaler.fit_transform(target.reshape(-1, 1))

# # environment.reproducible()

# # split data into training and validation
x_train, x_test, y_train, y_test = train_test_split(
    data, data, train_size=0.85
)

