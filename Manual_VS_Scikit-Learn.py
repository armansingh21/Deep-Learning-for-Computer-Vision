import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

# Define the manual implementation from the previous step
# test comment
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            y_predicted = self.predict(X)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def compute_cost(self, X, y):
        n_samples = len(y)
        y_predicted = self.predict(X)
        return (1 / (2 * n_samples)) * np.sum((y_predicted - y) ** 2)

# Example data
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([1, 2, 3, 4, 5])

# Train manual implementation
manual_regressor = LinearRegression(learning_rate=0.01, n_iterations=1000)
manual_regressor.fit(X_train, y_train)
manual_predictions = manual_regressor.predict(X_train)
manual_cost = manual_regressor.compute_cost(X_train, y_train)

# Train sklearn implementation
sklearn_regressor = SklearnLinearRegression()
sklearn_regressor.fit(X_train, y_train)
sklearn_predictions = sklearn_regressor.predict(X_train)

# Compute cost for sklearn model using the same cost function
def compute_cost(y_true, y_pred):
    n_samples = len(y_true)
    return (1 / (2 * n_samples)) * np.sum((y_pred - y_true) ** 2)

sklearn_cost = compute_cost(y_train, sklearn_predictions)

# Print results
print("Manual Implementation Predictions:", manual_predictions)
print("Manual Implementation Cost:", manual_cost)
print("Scikit-Learn Predictions:", sklearn_predictions)
print("Scikit-Learn Cost:", sklearn_cost)