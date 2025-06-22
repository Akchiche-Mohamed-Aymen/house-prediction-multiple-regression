import json
from multiple_regression import LinearRegression
import pandas as pd
import numpy as np
data = pd.read_csv("house_price.csv")
y = np.array(data["House_Price"]) 
y_train , y_test  = y[:700:] , y[700::]
data = data.drop(columns=['House_Price'])
X = np.array(data.iloc[:700: , ::])
model = LinearRegression(X, y_train , learning_rate=0.1 , iterations = 1500)
model.gradientDescent()

theta_values = model.weights.tolist()  # Convert to list if it's a NumPy array

with open("theta_values.json", "w") as f:
    json.dump({"theta": theta_values , "X_std" : model.X_std.tolist()  , "X_mean" : model.X_mean.tolist() }, f , indent=4)