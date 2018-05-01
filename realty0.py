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

data2015 = pd.read_csv('data2015.csv')


lat = data2015["latitude"]
lon = data2015["longitude"]
year = data2015["year"]
bdrms = data2015["bedrooms"]
fbath = data2015["full_bth"]
hbath = data2015["half_bth"]
sf = data2015["square_foot"]
res = data2015["res"]
condo = data2015["condo"]
built = data2015["yr_built"]
bldg = data2015["bldg_price"]
land = data2015["land_price"]

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

for i in range(len(lat)):
    if np.isnan(lat[i]) or np.isnan(lon[i]) or np.isnan(year[i]) or np.isnan(bdrms[i]) or np.isnan(fbath[i]) or np.isnan(hbath[i]) or np.isnan(sf[i]) or np.isnan(res[i]) or np.isnan(condo[i]) or np.isnan(built[i]) or np.isnan(bldg[i]) or np.isnan(land[i]):
        print(i)
        print(lat[i])
        print(lon[i])
        print(year[i])
        print(bdrms[i])
        print(fbath[i])
        print(hbath[i])
        print(sf[i])
        print(res[i])
        print(condo[i])
        print(built[i])
        print(bldg[i])
        print(land[i])

print("no None")


# Only the terms used to predict value
data2015 = pd.concat([lat,lon,year,bdrms,fbath,hbath,sf,res,condo,built], axis = 1)



# Target values. Currently bldg_price, land_price
target = pd.concat([bldg,land],axis=1)

# Normalize data
data_scaler = preprocessing.MinMaxScaler()
target_scaler = preprocessing.MinMaxScaler()

data2015 = data_scaler.fit_transform(data2015.values)
target = target_scaler.fit_transform(target.values)

# Setting seed for reproducibility
environment.reproducible()

# # split data into training and validation
x_train, x_test, y_train, y_test = train_test_split(
    data2015, target, train_size=0.85
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
with open('predict.csv','w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(data_scaler.inverse_transform(x_test))
    wr.writerow(target_scaler.inverse_transform(y_predict))
    wr.writerow(target_scaler.inverse_transform(y_test))
    wr.writerow([error])
