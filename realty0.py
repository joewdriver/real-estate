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
bldg_price = data['AV_BLDG']
land_price = data['AV_LAND']
total_price = data['AV_TOTAL']
LU = data['LU']
LU_keys = ['R1', 'R2', 'R3', 'R4']

lat1 = []
lon1 = []
lat2 = []
lon2 = []
lat3 = []
lon3 = []
lat4 = []
lon4 = []
bldg1 = []
bldg2 = []
bldg3 = []
bldg4 = []
land1 = []
land2 = []
land3 = []
land4 = []

print(LU[0])

prev = 0
for i in range(len(LU)):

    if LU[i] not in LU_keys:
        continue
    else:
        if LU[i] == 'R1':

            # TODO: TEMP FIX, no lat/lon.
            if type(locations[i]) == type(0.0) or locations[i] == "0":
                a,b = locations[prev].split("|")
                lat1.append(float(a[1:]))
                lon1.append(float(b[:-1]))
            else:
                a,b = locations[i].split("|")
                lat1.append(float(a[1:]))
                lon1.append(float(b[:-1]))
                prev = i
            bldg1.append(bldg_price[i])
            land1.append(land_price[i])
        if LU[i] == 'R2':

            # TODO: TEMP FIX, no lat/lon.
            if type(locations[i]) == type(0.0) or locations[i] == "0":
                a,b = locations[prev].split("|")
                lat2.append(float(a[1:]))
                lon2.append(float(b[:-1]))
            else:
                a,b = locations[i].split("|")
                lat2.append(float(a[1:]))
                lon2.append(float(b[:-1]))
                prev = i
            bldg2.append(bldg_price[i])
            land2.append(land_price[i])
        if LU[i] == 'R3':

            # TODO: TEMP FIX, no lat/lon.
            if type(locations[i]) == type(0.0) or locations[i] == "0":
                a,b = locations[prev].split("|")
                lat3.append(float(a[1:]))
                lon3.append(float(b[:-1]))
            else:
                a,b = locations[i].split("|")
                lat3.append(float(a[1:]))
                lon3.append(float(b[:-1]))
                prev = i
            bldg3.append(bldg_price[i])
            land3.append(land_price[i])
        if LU[i] == 'R4':

            # TODO: TEMP FIX, no lat/lon.
            if type(locations[i]) == type(0.0) or locations[i] == "0":
                a,b = locations[prev].split("|")
                lat4.append(float(a[1:]))
                lon4.append(float(b[:-1]))
            else:
                a,b = locations[i].split("|")
                lat4.append(float(a[1:]))
                lon4.append(float(b[:-1]))
                prev = i
            bldg4.append(bldg_price[i])
            land4.append(land_price[i])

lat1 = pd.Series(lat1,name="latitude")
lon1 = pd.Series(lon1,name="longitude")
lat2 = pd.Series(lat2,name="latitude")
lon2 = pd.Series(lon2,name="longitude")
lat3 = pd.Series(lat3,name="latitude")
lon3 = pd.Series(lon3,name="longitude")
lat4 = pd.Series(lat4,name="latitude")
lon4 = pd.Series(lon4,name="longitude")
bldg1 = pd.Series(bldg1,name="bldg_price")
land1 = pd.Series(land1,name="land_price")
bldg2 = pd.Series(bldg2,name="bldg_price")
land2 = pd.Series(land2,name="land_price")
bldg3 = pd.Series(bldg3,name="bldg_price")
land3 = pd.Series(land3,name="land_price")
bldg4 = pd.Series(bldg4,name="bldg_price")
land4 = pd.Series(land4,name="land_price")


# Only the terms used to predict value
data1 = pd.concat([lat1,lon1], axis = 1)
data2 = pd.concat([lat2,lon2], axis = 1)
data3 = pd.concat([lat3,lon3], axis = 1)
data4 = pd.concat([lat4,lon4], axis = 1)



# Target values. Currently bldg_price, land_price, and total_price
target1 = pd.concat([bldg1,land1],axis=1)
target2 = pd.concat([bldg2,land2],axis=1)
target3 = pd.concat([bldg3,land3],axis=1)
target4 = pd.concat([bldg4,land4],axis=1)


data_scaler1 = preprocessing.MinMaxScaler()
target_scaler1 = preprocessing.MinMaxScaler()
data_scaler2 = preprocessing.MinMaxScaler()
target_scaler2 = preprocessing.MinMaxScaler()
data_scaler3 = preprocessing.MinMaxScaler()
target_scaler3 = preprocessing.MinMaxScaler()
data_scaler4 = preprocessing.MinMaxScaler()
target_scaler4 = preprocessing.MinMaxScaler()

data1 = data_scaler1.fit_transform(data1.values)
target1 = target_scaler1.fit_transform(target1.values)
data2 = data_scaler2.fit_transform(data2.values)
target2 = target_scaler2.fit_transform(target2.values)
data3 = data_scaler3.fit_transform(data3.values)
target3 = target_scaler3.fit_transform(target3.values)
data4 = data_scaler4.fit_transform(data4.values)
target4 = target_scaler4.fit_transform(target4.values)

# Setting seed for reproducibility
environment.reproducible()

# # split data into training and validation
x_train1, x_test1, y_train1, y_test1 = train_test_split(
    data1, target1, train_size=0.85
)
x_train2, x_test2, y_train2, y_test2 = train_test_split(
    data2, target2, train_size=0.85
)
x_train3, x_test3, y_train3, y_test3 = train_test_split(
    data3, target3, train_size=0.85
)
x_train4, x_test4, y_train4, y_test4 = train_test_split(
    data4, target4, train_size=0.85
)

# Creating the neural network
#   connection:
#       Values being trained on. Currently only lat and lon (2)
#       Size of hidden layer. Currently arbitrarily set to 50
#       Size of output values. Currently bldg_price, land_price, and total_price (3)
cgnet1 = algorithms.ConjugateGradient(
    connection=[
        layers.Input(2),
        layers.Sigmoid(500),
        layers.Sigmoid(2),
    ],
    search_method='golden',
    show_epoch=25,
    verbose=True,
    addons=[algorithms.LinearSearch],
)
cgnet2 = algorithms.ConjugateGradient(
    connection=[
        layers.Input(2),
        layers.Sigmoid(500),
        layers.Sigmoid(2),
    ],
    search_method='golden',
    show_epoch=25,
    verbose=True,
    addons=[algorithms.LinearSearch],
)
cgnet3 = algorithms.ConjugateGradient(
    connection=[
        layers.Input(2),
        layers.Sigmoid(500),
        layers.Sigmoid(2),
    ],
    search_method='golden',
    show_epoch=25,
    verbose=True,
    addons=[algorithms.LinearSearch],
)
cgnet4 = algorithms.ConjugateGradient(
    connection=[
        layers.Input(2),
        layers.Sigmoid(500),
        layers.Sigmoid(2),
    ],
    search_method='golden',
    show_epoch=25,
    verbose=True,
    addons=[algorithms.LinearSearch],
)

# Train neural net
cgnet1.train(x_train1, y_train1, x_test1, y_test1, epochs=100)
cgnet2.train(x_train2, y_train2, x_test2, y_test2, epochs=100)
cgnet3.train(x_train3, y_train3, x_test3, y_test3, epochs=100)
cgnet4.train(x_train4, y_train4, x_test4, y_test4, epochs=100)


# Make predictions
print("Starting predictions")
y_predict1 = cgnet1.predict(x_test1)
error1 = rmsle(target_scaler1.inverse_transform(y_test1),
    target_scaler1.inverse_transform(y_predict1))
print(error1)
y_predict2 = cgnet2.predict(x_test2)
error2 = rmsle(target_scaler2.inverse_transform(y_test2),
    target_scaler2.inverse_transform(y_predict2))
print(error2)
y_predict3 = cgnet3.predict(x_test3)
error3 = rmsle(target_scaler3.inverse_transform(y_test3),
    target_scaler3.inverse_transform(y_predict3))
print(error3)
y_predict4 = cgnet4.predict(x_test4)
error4 = rmsle(target_scaler4.inverse_transform(y_test4),
    target_scaler4.inverse_transform(y_predict4))
print(error4)
# write values to csv
#   Row 1: Test cases (lat and lon)
#   Row 2: Predicted values (prices)
#   Row 3: Test values (prices)
#   Row 4: Error
with open('test_180415_1.csv','w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(data_scaler1.inverse_transform(x_test1))
    wr.writerow(target_scaler1.inverse_transform(y_predict1))
    wr.writerow(target_scaler1.inverse_transform(y_test1))
    wr.writerow([error1])
    wr.writerow([])
    wr.writerow([])
    wr.writerow(data_scaler2.inverse_transform(x_test2))
    wr.writerow(target_scaler2.inverse_transform(y_predict2))
    wr.writerow(target_scaler2.inverse_transform(y_test2))
    wr.writerow([error2])
    wr.writerow([])
    wr.writerow([])
    wr.writerow(data_scaler3.inverse_transform(x_test3))
    wr.writerow(target_scaler3.inverse_transform(y_predict3))
    wr.writerow(target_scaler3.inverse_transform(y_test3))
    wr.writerow([error3])
    wr.writerow([])
    wr.writerow([])
    wr.writerow(data_scaler4.inverse_transform(x_test4))
    wr.writerow(target_scaler4.inverse_transform(y_predict4))
    wr.writerow(target_scaler4.inverse_transform(y_test4))
    wr.writerow([error4])
    wr.writerow([])
    wr.writerow([])