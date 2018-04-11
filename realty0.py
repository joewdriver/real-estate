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

# Import data from file
data = pd.read_csv('property-assessment-fy2015.csv')


# TODO: Need to filter all data

# Values to predict on
locations = data['Location']

# Prices
bldg_price = data[['AV_BLDG']].copy()
land_price = data[['AV_LAND']].copy()
total_price = data[['AV_TOTAL']].copy()

# TODO:Add LU

lat = []
lon = []
prev = 0
for i in range(len(locations)):

    # TODO: TEMP FIX, no lat/lon.
    if type(locations[i]) == type(0.0) or locations[i] == "0":
        a,b = locations[prev].split("|")
        lat.append(float(a[1:]))
        lon.append(float(b[:-1]))
    else:
        a,b = locations[i].split("|")
        lat.append(float(a[1:]))
        lon.append(float(b[:-1]))
        prev = i

lat = pd.Series(lat,name="latitude")
lon = pd.Series(lon,name="longitude")

# Only the terms used to predict value
data = pd.concat([lat,lon], axis = 1)

# Target values. Currently bldg_price, land_price, and total_price
target = pd.concat([bldg_price,land_price,total_price],axis=1)


data_scaler = preprocessing.MinMaxScaler()
target_scaler = preprocessing.MinMaxScaler()


data = data_scaler.fit_transform(data.values)
target = target_scaler.fit_transform(target.values)

# Setting seed for reproducibility
environment.reproducible()

# # split data into training and validation
x_train, x_test, y_train, y_test = train_test_split(
    data, target, train_size=0.85
)

# Creating the neural network
#   connection:
#       Values being trained on. Currently only lat and lon (2)
#       Size of hidden layer. Currently arbitrarily set to 50
#       Size of output values. Currently bldg_price, land_price, and total_price (3)
cgnet = algorithms.ConjugateGradient(
    connection=[
        layers.Input(2),
        layers.Sigmoid(50),
        layers.Sigmoid(3),
    ],
    search_method='golden',
    show_epoch=25,
    verbose=True,
    addons=[algorithms.LinearSearch],
)

# Train neural net
cgnet.train(x_train, y_train, x_test, y_test, epochs=100)


# Make predictions
print("Starting predictions")
y_predict = cgnet.predict(x_test)
error = rmsle(target_scaler.inverse_transform(y_test),
    target_scaler.inverse_transform(y_predict))
print(error)

# write values to csv
#   Row 1: Test cases (lat and lon)
#   Row 2: Predicted values (prices)
#   Row 3: Test values (prices)
#   Row 4: Error
with open('test_180410.csv','w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(data_scaler.inverse_transform(x_test))
    wr.writerow(target_scaler.inverse_transform(y_predict))
    wr.writerow(target_scaler.inverse_transform(y_test))
    wr.writerow([error])
