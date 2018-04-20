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

# Import data from file by year
data2015 = pd.read_csv('property-assessment-fy2015.csv')

# TODO: Need to filter all data

# Values to predict on
locations = data2015['Location']

# Prices
bldg_price = data2015['AV_BLDG']
land_price = data2015['AV_LAND']
total_price = data2015['AV_TOTAL']
land_sf = data2015['LAND_SF']
r_bdrms = data2015['R_BDRMS']
r_full_bth = data2015['R_FULL_BTH']
r_half_bth = data2015['R_HALF_BTH']
u_bdrms = data2015['U_BDRMS']
u_full_bth = data2015['U_FULL_BTH']
u_half_bth = data2015['U_HALF_BTH']
cm_id = data2015['CM_ID']

LU = data2015['LU']
LU_keys = ['R1', 'R2', 'R3', 'R4', 'CD']

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


# Find the location of a specific condo id
condo_loc = {}
for i in range(len(cm_id)):
    if LU[i] == 'CM':
        if type(locations[i]) == type(''):
            condo_loc[cm_id[i]] = locations[i]
    else:
        continue


for i in range(len(LU)):

    if LU[i] not in LU_keys:
        continue
    else:
        cur_lat = 0
        cur_lon = 0
        # The only locations that aren't filled in are condos. If it isn't filled, get the location 
        # for the condo from the dictionary.
        if type(locations[i]) == type(0.0) or locations[i] == "0" or locations[i] == "(0.000000000| 0.000000000)":
            if cm_id[i] is not None or np.isnan(cm_id[i]):
                if cm_id[i] in condo_loc.keys():
                    a,b = condo_loc[cm_id[i]].split("|")
                    cur_lat = float(a[1:])
                    cur_lon = float(b[:-1])
                else:
                    continue
            else:
                continue

        # The location is filled.
        else:
            a,b = locations[i].split("|")
            cur_lat = float(a[1:])
            cur_lon = float(b[:-1])

        # No land_sf value. remove stored lat and lon
        if land_sf[i] is None or np.isnan(land_sf[i]):
            continue


        # Condo and residential bedrooms/bathrooms stored separately initially
        # Store if condo or residential
        if LU[i] == 'CD' or LU[i] == 'CM':
            # Make sure a value for bathrooms and bedrooms is provided otherwise pass
            if u_bdrms[i] is not None and u_full_bth[i] is not None and u_half_bth[i] is not None:
                if np.isnan(float(u_bdrms[i])) or np.isnan(float(u_full_bth[i])) or np.isnan(float(u_half_bth[i])):
                    continue
                else:
                    bdrms.append(float(u_bdrms[i]))
                    fbath.append(float(u_full_bth[i]))
                    hbath.append(float(u_half_bth[i]))
                    condo.append(1)
                    res.append(0)
            else:
                continue
        else:
            # Make sure a value for bathrooms and bedrooms is provided otherwise pass
            if r_bdrms[i] is not None and r_full_bth[i] is not None and r_half_bth[i] is not None:
                if np.isnan(float(r_bdrms[i])) or np.isnan(float(r_full_bth[i])) or np.isnan(float(r_half_bth[i])):
                    continue
                else:
                    bdrms.append(float(r_bdrms[i]))
                    fbath.append(float(r_full_bth[i]))
                    hbath.append(float(r_half_bth[i]))
                    condo.append(0)
                    res.append(1)
            else:
                continue
        bldg.append(float(bldg_price[i]))
        land.append(float(land_price[i]))
        sf.append(float(land_sf[i]))
        lat.append(cur_lat)
        lon.append(cur_lon)

year = [2015]*len(lat)
lat = pd.Series(lat,name="latitude")
lon = pd.Series(lon,name="longitude")
year = pd.Series(year,name="year")
bdrms = pd.Series(bdrms,name="bedrooms")
fbath = pd.Series(fbath,name="full_bth")
hbath = pd.Series(hbath,name="half_bth")
sf = pd.Series(sf,name="square_foot")
res = pd.Series(res,name="res")
condo = pd.Series(condo,name="condo")
bldg = pd.Series(bldg,name="bldg_price")
land = pd.Series(land,name="land_price")




# Only the terms used to predict value
data2015 = pd.concat([lat,lon,year,bdrms,fbath,hbath,sf,res,condo], axis = 1)



# Target values. Currently bldg_price, land_price
target = pd.concat([bldg,land],axis=1)

data_scaler = preprocessing.MinMaxScaler()
target_scaler = preprocessing.MinMaxScaler()

data2015 = data_scaler.fit_transform(data2015.values)
target = target_scaler.fit_transform(target.values)

# Setting seed for reproducibility
environment.reproducible()

# # split data2015 into training and validation
x_train, x_test, y_train, y_test = train_test_split(
    data2015, target, train_size=0.85
)

# Creating the neural network
#   connection:
#       Values being trained on. Currently only lat and lon (2)
#       Size of hidden layer. Currently arbitrarily set to 50
#       Size of output values. Currently bldg_price, land_price, and total_price (3)
cgnet = algorithms.ConjugateGradient(
    connection=[
        layers.Input(9),
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
with open('test_180415.csv','w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(data_scaler.inverse_transform(x_test))
    wr.writerow(target_scaler.inverse_transform(y_predict))
    wr.writerow(target_scaler.inverse_transform(y_test))
    wr.writerow([error])
