#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Loading dataset to program
dataset = pd.read_csv('Position_Salaries.csv')

# Getting independent variables
X = dataset.iloc[:, 1:2].values
# Getting dependent variables
Y = dataset.iloc[:, 2].values

# Creating linear regression object
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# Creating polynomial regression object
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

# Creating linear regression object that will use the polynomial model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualizing results for the Linear Regression model

plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Company salaries vs position")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Visualizing results for the Polynomial Regression model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Company salaries vs position. Polynomial Model")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result using Linear Regression
linear_y_pred = lin_reg.predict(6.5)

# Predicting a new result using the Polynomial Regression model
poly_y_pred = lin_reg_2.predict(poly_reg.fit_transform(6.5))