import csv
import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
import sklearn.learning_curve as curves
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from neupy import environment
from neupy import algorithms, layers
from neupy import plots
from neupy.estimators import rmsle

data = pd.read_csv('property-assessment-fy2015.csv')


# TODO: Need to filter all data

locations = data['Location']

bldg_price = data[['AV_BLDG']].copy()
land_price = data[['AV_LAND']].copy() # TODO REMOVE
total_price = data[['AV_TOTAL']].copy()

# TODO:Add LU

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

lat = pd.Series(lat,name="latitude")
lon = pd.Series(lon,name="longitude")

data = pd.concat([lat,lon,bldg_price,land_price,total_price], axis = 1)


data_scaler = preprocessing.MinMaxScaler()
target_scaler = preprocessing.MinMaxScaler()


# TODO: Check after preprocessing data. Running into NaN or infinities here. Likely from missing values
data = data_scaler.fit_transform(data.values)
target = target_scaler.fit_transform(data.reshape(-1, 1))

environment.reproducible()

# # split data into training and validation
x_train, x_test, y_train, y_test = train_test_split(
    data, data, train_size=0.85
)

cgnet = algorithms.ConjugateGradient(
    connection=[
        layers.Input(5),
        layers.Sigmoid(50),
        layers.Sigmoid(5),
    ],
    search_method='golden',
    show_epoch=25,
    verbose=True,
    addons=[algorithms.LinearSearch],
)

cgnet.train(x_train, y_train, x_test, y_test, epochs=100)

# plots.error_plot(cgnet)
print(x_test)
print("Starting predictions")
y_predict = cgnet.predict(x_test).round(1)
error = rmsle(target_scaler.inverse_transform(y_test),
              target_scaler.inverse_transform(y_predict))
print(y_predict)