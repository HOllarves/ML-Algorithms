# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


#Loading dataset to program
dataset = pd.read_csv('50_Startups.csv')

#Obtaining X matrix for our model
X = dataset.iloc[:, :-1].values
#Obtaining Y results for our model
Y = dataset.iloc[:, 4].values

# Using Label and OneHot encoders to encode categoric data
# Instanciating Label Encoder class
labeledencoder_X = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = [3])
# Appolying encoding to the state column or index 3
X[:, 3] = labeledencoder_X.fit_transform(X[:,3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:, 1:]

# Splitting dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predigint the test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
# Adding a new column with all 1 values, to be the interceptor or switch.
X = np.append(arr = np.ones([50, 1]).astype(int), values = X, axis = 1)

#Backwards elimination # 1
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# Intersection of rows is not done by default.
# Thats why we added it before.
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Backwards elimination # 2
# Removing state dummy variable
X_opt = X[:, [0, 1, 3, 4, 5]]
# Intersection of rows is not done by default.
# Thats why we added it before.
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Backwards elimination # 3
# Removing second state dummy variable
X_opt = X[:, [0, 1, 4, 5]]
# Intersection of rows is not done by default.
# Thats why we added it before.
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Backwards elimination # 4
# Removing marketing spend
X_opt = X[:, [0, 3, 5]]
# Intersection of rows is not done by default.
# Thats why we added it before.
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

# Backwards elimination # 5
# Removing administration spend
X_opt = X[:, [0, 3]]
# Intersection of rows is not done by default.
# Thats why we added it before.
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

# Optimal pair of values achieved, in this case it's only one
# independent variable = R&D spend

