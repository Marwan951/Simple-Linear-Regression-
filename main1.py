from sklearn import linear_model
from sklearn import metrics
import numpy as np
import pandas as pd
import re
data = pd.read_csv(r'C:\3rd_year\second term\pattern recognition\labs\assignment\assignment1_dataset.csv')
# print(data.describe())
X=[]
X = data['transaction date'].str.split('-')
n = len(X)
years = []
for i in range(n):
    years.append(int(X[i][0]))
year = pd.Series(years)
X[1]= data['house age']
X[2] = data['distance to the nearest MRT station']
X[3] = data['number of convenience stores']
X[4] = data['latitude']
X[5] = data['longitude']
Y = data['house price of unit area']
# number of examples
n = float(len(X[0]))
# loop for changing the value of m&c for epochs times
X = None
Y = data['house price of unit area']
tags = [ "transaction date","house age", "distance to the nearest MRT station", "number of convenience stores", "latitude","longitude"]
for i in tags:
    Y = data['house price of unit area']
    if i != "transaction date":
        X = data[i]
    elif i == "transaction date":
        X = year
    cls = linear_model.LinearRegression()
    X = np.expand_dims(X, axis=1)
    Y = np.expand_dims(Y, axis=1)
    cls.fit(X, Y)
    prediction = cls.predict(X)
    print('MSE for',i, metrics.mean_squared_error(Y, prediction))
# print('MSE | transaction date: ', metrics.mean_squared_error(Y, prediction0), "\n")
# print('MSE | house age: ', metrics.mean_squared_error(Y, prediction1), "\n")
# print('MSE | distance to the nearest MRT station: ', metrics.mean_squared_error(Y, prediction2), "\n")
# print('MSE | number of convenience stores: ', metrics.mean_squared_error(Y, prediction3), "\n")
# print('MSE | latitude: ', metrics.mean_squared_error(Y, prediction4), "\n")
# print('MSE | longitude: ', metrics.mean_squared_error(Y, prediction5), "\n")
