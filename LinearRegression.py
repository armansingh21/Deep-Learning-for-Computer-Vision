import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Number of training examples and number of features
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent
        for _ in range(self.n_iterations):
            # Calculate predictions
            y_predicted = self.predict(X)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def compute_cost(self, X, y):
        n_samples = len(y)
        y_predicted = self.predict(X)
        return (1 / (2 * n_samples)) * np.sum((y_predicted - y) ** 2)

# Example usage
if __name__ == "__main__":
    # Example data: (X_train, y_train)
    X_train = np.array([[1], [2], [3], [4], [5]])
    y_train = np.array([1, 2, 3, 4, 5])
    
    # Create and train the model
    regressor = LinearRegression(learning_rate=0.01, n_iterations=1000)
    regressor.fit(X_train, y_train)
    
    # Make predictions
    predictions = regressor.predict(X_train)
    print("Predictions:", predictions)
    
    # Compute the cost
    cost = regressor.compute_cost(X_train, y_train)
    print("Cost:", cost)