###
# realty13-18Predict.py
# 
# Predicts housing prices for the properties in data2018.csv for the year 2021.
# 
# Outputs predict13-18.csv
###


import csv
import numpy as np
import pandas as pd
from neupy import plots
from neupy import environment
import matplotlib.pyplot as pl
from sklearn import preprocessing
from neupy.estimators import rmsle
from neupy import algorithms, layers
import sklearn.learning_curve as curves
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# Read training data files and testing data files
data = pd.read_csv('./new_data/data13-18.csv')

# Shuffle data
data = data.iloc[np.random.permutation(len(data))]

test = pd.read_csv('./new_data/future_values.csv')

# Get specific columns that we will be using and transfer them into the appropriate type
# For training data
lat = data["latitude"]
lon = data["longitude"]
year = data["year"]
bdrms = data["bedrooms"]
fbath = data["full_bth"]
hbath = data["half_bth"]
sf = data["square_foot"]
res = data["res"]
condo = data["condo"]
built = data["yr_built"]
bldg = data["bldg_price"]
land = data["land_price"]

lat.astype('float')
lon.astype('float')
year.astype('float')
bdrms.astype('float')
fbath.astype('float')
hbath.astype('float')
sf.astype('float')
res.astype('float')
condo.astype('float')
built.astype('float')
bldg.astype('float')
land.astype('float')

# Get specific columns that we will be using and transfer them into the appropriate type
# For testing data
lat_test = test["latitude"]
lon_test = test["longitude"]
year_test = test["year"]
bdrms_test = test["bedrooms"]
fbath_test = test["full_bth"]
hbath_test = test["half_bth"]
sf_test = test["square_foot"]
res_test = test["res"]
condo_test = test["condo"]
built_test = test["yr_built"]

lat_test.astype('float')
lon_test.astype('float')
year_test.astype('float')
bdrms_test.astype('float')
fbath_test.astype('float')
hbath_test.astype('float')
sf_test.astype('float')
res_test.astype('float')
condo_test.astype('float')
built_test.astype('float')


# Concatenate all values above
data = pd.concat([lat,lon,year,bdrms,fbath,hbath,sf,res,condo,built], axis = 1)
test = pd.concat([lat_test,lon_test,year_test,bdrms_test,fbath_test,hbath_test,sf_test,res_test,condo_test,built_test], axis = 1)



# Target values. Currently bldg_price, land_price
target = pd.concat([bldg,land],axis=1)

# Normalize data
data_scaler = preprocessing.MinMaxScaler()
target_scaler = preprocessing.MinMaxScaler()

data = data_scaler.fit_transform(data.values)
target = target_scaler.fit_transform(target.values)

test = data_scaler.fit_transform(test.values)

# Setting seed for reproducibility
environment.reproducible()

# # split data into training and validation
x_train, x_test, y_train, y_test = train_test_split(
    data, target, train_size=0.85
)

# Creating the neural network
#   connection:
#       Values being trained on. Currently lat, lon, year, bdrms, bathrooms, square feet, 
#           residential, condo, year built (10)
#       Size of hidden layer. Currently arbitrarily set to 50
#       Size of output values. Currently bldg_price and land_price (2)
cgnet = algorithms.ConjugateGradient(
    connection=[
        layers.Input(10),
        layers.Sigmoid(50),
        layers.Sigmoid(2),
    ],
    search_method='golden',
    show_epoch=25,
    verbose=True,
    addons=[algorithms.LinearSearch],
)

# Train neural net
cgnet.train(x_train, y_train, x_test, y_test, epochs=100)


# Make predictions
y_predict = cgnet.predict(test)



# write values to csv
#   lat,lon,year,bdrms,fbath,hbath,sf,res,condo,built
with open('predict13-18.csv','w') as myfile:
    wr = csv.writer(myfile,quoting=csv.QUOTE_ALL)
    wr.writerow(["latitude", "longitude", "year", "bedrooms", "full_bth", "half_bth", "square_foot", "res", "condo", "yr_built", "bldg_price", "land_price"])
    for i in range(len(y_predict)):
        wr.writerow(data_scaler.inverse_transform(test)[i].tolist() + target_scaler.inverse_transform(y_predict)[i].tolist())
