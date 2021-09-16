"""
@author: Dhruv Khurana
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import datetime as dt
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("NSE-TATAGLOBAL.csv")
print(df.head())
df = df[['Date','Close']] 
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df['Date']=df['Date'].map(dt.datetime.toordinal)
print(df.head())
X = df.iloc[:, 0].values.reshape(-1, 1)  
Y = df.iloc[:, 1].values.reshape(-1, 1) 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)
y_pred = linear_regressor.predict(x_test)
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=4)
plt.xticks(())
plt.yticks(())
print('Mean squared error: %.2f' % mean_squared_error(y_test,y_pred))
print('Coefficient of determination %.2f'% r2_score(y_test, y_pred))
plt.show()








