import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate random data
X = np.random.rand(100, 1) * 10  # 100 random points between 0 and 10
y = 2 * X + 3 + np.random.randn(100, 1) * 2  # Linear relation with some noise 

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred, color='red', label='Linear Fit')
plt.legend()
plt.title('Linear Regression Example')
plt.savefig('linear_regression.png')

metric = np.mean((y - y_pred) ** 2)

with open('metrics.txt', 'w') as f:
    f.write(f'Mean Squared Error: {metric}\n')

print(f'Mean Squared Error: {metric}')