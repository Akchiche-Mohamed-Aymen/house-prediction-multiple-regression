import numpy as np

class LinearRegression:
    def __init__(self, X , y, learning_rate=0.1 , iterations  = 1000):
        self.lr = learning_rate
        self.iter  = iterations
        self.weights = None
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X = X
        self.y = y
    def scale(self , X):
        X_scaled = (X - self.X_mean) / self.X_std
        return X_scaled
    def gradientDescent(self):
        n_samples = len(self.X)
        ones = np.ones((n_samples, 1))
        features = np.hstack((ones, self.scale(self.X)))  # shape: (n_samples, n_features+1)
        self.weights = np.zeros(features.shape[1])
        for _ in range(self.iter):
            y_predicted = np.dot(features, self.weights.T)
            error = y_predicted - self.y
            dw = (1 / n_samples) * np.dot(features.T, error)
            self.weights -= self.lr * dw
    def evaluate(self , X):
        X =  self.scale(X)
        return sum(self.weights * [1 , * X])
    def mse(self):
        y_predict  = np.array([self.evaluate(row) for row in self.X])
        error = (y_predict - self.y)**2
        mse = np.mean(error) / 2
        return mse
