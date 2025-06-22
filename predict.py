import numpy as np
import json

def normalize(X):
    return  (X - X_mean) / X_std
thetas = None
X_mean = None
X_std = None
with open("theta_values.json", "r") as f:
    data = json.load(f)
    thetas = np.array(data["theta"])
    X_mean = np.array(data["X_mean"])
    X_std = np.array(data["X_std"])
 
def evaluatePrice(row):
    return sum(thetas * [1 , *normalize(np.array(row))])
