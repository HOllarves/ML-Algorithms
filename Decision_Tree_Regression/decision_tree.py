"""
Created on Sun Jan  7 15:00:40 2018

@author: henry
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

#Loading dataset to program
dataset = pd.read_csv('Position_Salaries.csv')

# Getting independent variables
X = dataset.iloc[:, 1:2].values
# Getting dependent variables
Y = dataset.iloc[:, 2].values

# Fitting the Regression Model to the dataset
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, Y)


# Predicting the new result
y_pred = regressor.predict(6.5)

# Visualizing results for the Polynomial Regression model
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.title("Company salaries vs position. DTR, Smoothed")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
