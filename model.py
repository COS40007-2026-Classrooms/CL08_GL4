import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate random data
X = np.random.rand(100, 1) * 10
y = 2 * X + 3 + np.random.randn(100, 1)

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Plot results
plt.scatter(X, y, label="Actual")
plt.plot(X, y_pred, label="Predicted")
plt.legend()
plt.title("Simple Regression Model")
plt.savefig("model_results.png")

# Save metrics
mse = np.mean((y - y_pred) ** 2)

with open("metrics.txt", "w") as f:
    f.write(f"Mean Squared Error: {mse}\n")

print("Training complete. Files saved.")