"""
Created on Sun Jan  7 15:00:40 2018

@author: henry
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

#Loading dataset to program
dataset = pd.read_csv('Position_Salaries.csv')

# Getting independent variables
X = dataset.iloc[:, 1:2].values
# Getting dependent variables
Y = dataset.iloc[:, 2].values


sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y.reshape(-1,1))

# Fitting the SVR Model to the dataset
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)

# Predicting the new result
y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array(6.5))))

# Visualizing results for the SVR model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.title("Company salaries vs position. SVR, Smoothed")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
