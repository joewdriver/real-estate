# Finds and outputs the indices of properties in all datasets


import csv
import pandas as pd
import numpy as np


data2008 = pd.read_csv('data2008.csv')
data2009 = pd.read_csv('data2009.csv')
data2010 = pd.read_csv('data2010.csv')
data2011 = pd.read_csv('data2011.csv')
data2013 = pd.read_csv('data2013.csv')
data2015 = pd.read_csv('data2015.csv')
data2018 = pd.read_csv('data2018.csv')


with open('common.csv','w') as myfile:
    wr = csv.writer(myfile,quoting=csv.QUOTE_ALL)
    wr.writerow(["2008","2009","2010","2011","2013","2015","2018"])
    for i in range(len(data2009["latitude"])):
        print(i)
        ind08 = np.where(data2008["address"] == data2009["address"][i])
        ind10 = np.where(data2010["address"] == data2009["address"][i])
        ind11 = np.where(data2011["address"] == data2009["address"][i])
        ind13 = np.where(data2013["address"] == data2009["address"][i])
        ind15 = np.where(data2015["address"] == data2009["address"][i])
        # ind09 = data2015[data2015["address"] == data2015["address"][i]]
        ind18 = np.where(data2018["address"] == data2009["address"][i])
        # print("i= ",i)
        # print(ind08[0].size)
        # print(ind10[0].size)
        # print(ind11[0].size)
        # print(ind13[0].size)
        # print(ind15[0].size)
        # print(ind18)

        if ind08[0].size != 0 and ind10[0].size != 0 and ind11[0].size != 0 and ind13[0].size != 0 and ind15[0].size != 0 and ind18[0].size != 0:
            wr.writerow([ind08[0][0],i,ind10[0][0],ind11[0][0],ind13[0][0],ind15[0][0],ind18[0][0]])