# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load California housing data, select features and targets, and split into training and
testing sets.
2.Scale both X (features) and Y (targets) using StandardScaler.
3.Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data.
4.Predict on test data, inverse transform the results, and calculate the mean squared
error. 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:  SANJAY A
RegisterNumber:  212224040288

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

np.random.seed(42)
X = np.random.rand(200, 3) * 100
price = 50000 + (X[:, 0] * 1500) + (X[:, 1] * 1000) + np.random.randn(200) * 5000
occupants = 1 + (X[:, 1] // 10) + np.random.randint(0, 3, size=200)
y = np.vstack((price, occupants)).T

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_price = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
model_occupants = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)

model_price.fit(X_train, y_train[:, 0])
model_occupants.fit(X_train, y_train[:, 1])

y_pred_price = model_price.predict(X_test)
y_pred_occupants = model_occupants.predict(X_test)

print("R² Score for Price Prediction:", r2_score(y_test[:, 0], y_pred_price))
print("R² Score for Occupants Prediction:", r2_score(y_test[:, 1], y_pred_occupants))

print("\nSample Predictions:")
for i in range(5):
    print(f"Actual: Price={y_test[i, 0]:.2f}, Occupants={y_test[i, 1]} | "
          f"Predicted: Price={y_pred_price[i]:.2f}, Occupants={y_pred_occupants[i]:.2f}")

*/
```

## Output:

<img width="874" height="588" alt="image" src="https://github.com/user-attachments/assets/fbb6cf73-74b8-4f9f-b1aa-8faf551fac9d" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.


