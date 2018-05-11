# real-estate
Machine Learning project for CS542 Spring 2018.

## Data sets

### [Old data](https://github.com/joewdriver/real-estate/tree/master/old_data)

Data sets pulled directly from [Analyze Boston](https://data.boston.gov/) can be found in the [`old_data`](https://github.com/joewdriver/real-estate/tree/master/old_data) directory. These contain a lot more information than the model uses.

### [New data](https://github.com/joewdriver/real-estate/tree/master/new_data)

Cleaned data sets can be found in the [`new_data`](https://github.com/joewdriver/real-estate/tree/master/new_data). These data sets only contain residential and condo properties. Since our most complete data set was the set from 2015, all other data sets only use properties that are in the 2015 data set as well.

##### Basic data sets

These are the data sets created by directly cleaning the data sets from [`old_data`](https://github.com/joewdriver/real-estate/tree/master/old_data). These take the format `data20**.csv` where the ** is the last 2 digits of the data set's year. These are created by the scripts `realty20**.py` in the main directory. Each year needs a specific script due to differences in format of the data sets in [`old_data`](https://github.com/joewdriver/real-estate/tree/master/old_data).

##### Compiled data sets

These are data sets that combine basic data sets together. `dataAll.csv` is a compilation of every basic data set. `data13-18.csv` is the data sets from 2013 to 2018. This was just to test to see if the market depression in earlier years were strongly affecting our predictions. `data08-15.csv` is a compilation of the years 2008 through 2015. This is for training our neural network to predict the prices of properties in 2018.

##### Other data sets

There are a few data sets that are used in special cases. `sim_08_15.csv` is the set of properties in 2015 that also appear in 2008. This data set is made by the script `sim_08_15.py`. This data set is used in the making of the map that displays the change in price/square foot from 2008 to 2015.

`common.csv` is the list of indices of properties that are common across all basic data sets. Each row is a single property and each column is the index of the property in the corresponding year's data set. This data set was made with the script `common_data.py`.

`future_values.csv` is a list of selected properties that appear in all basic data sets with their years set to 2019, 2020, and 2021. These are for observing the pricing trend across all of the years that we have data for plus 3 predicted years. This data set was made by `generate_predict_values.py`.

## Scripts

### Neural Network

All of the neural network scripts work the same way and depend on what data sets are being trained on and the outputs that are being used. These can be seen in the following table.

|           script            | training values |  predicted values  |   output predictions    |
| --------------------------- | --------------- | ------------------ | ----------------------- |
| `realty0.py`                | `data08-15.csv` | `data2018.csv`     | `predict2018.csv`       |
| `realtyAllPredict.py`       | `dataAll.csv`   | `future_vales.csv` | `predict.csv`           |
| `realtyAllPredict2021.py`   | `dataAll.csv`   | `values2021.csv`   | `predict2021.csv`       |
| `realty13-18Predict.py`     | `data13-18.csv` | `future_vales.csv` | `predict13-18.csv`      |
| `realty13-18Predict2021.py` | `data13-18.csv` | `values2021.csv`   | `predict13-18_2021.csv` |

Since `realty0.py` predicts known values in `data2018.csv`, errors can be calculated. This model gets a RMSLE of approximately 1.24.

### Maps

Since this project deals with realty data, creating a map to visualize data makes sense. This project has 3 scripts that generate map visualizations. `mapHex_compare_15_Sqftvalue.py` maps the price per square foot of properties in 2015. It creates the following map `PerSq_norm_sqftvalue15.jpg`.

![alt text](https://github.com/joewdriver/real-estate/blob/master/images/PerSq_norm_sqftvalue15.jpg "2015 price per square foot")

`mapHex_compare_08-15.py` maps the change in the price per square foot of properties between 2008 and 2015. It creates the following map `PerSq_norm_08-15.jpg`.

![alt text](https://github.com/joewdriver/real-estate/blob/master/images/PerSq_norm_08-15.jpg "Change in price per square foot from 2008 to 2015")

`mapHex_compare_18-21.py` maps the change in the price per square foot of properties between 2018 and the predicted values of 2021. It creates the following map `PerSq_norm_18-21.jpg`.

![alt text](https://github.com/joewdriver/real-estate/blob/master/images/PerSq_norm_18-21.jpg "Change in price per square foot from 2018 to predicted values in 2021")