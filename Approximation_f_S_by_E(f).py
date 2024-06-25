import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap


np.random.seed(200)
X1 = np.random.normal(0, 1, 100)
X2 = np.random.normal(0, 1, 100)
Y = 2 * X1 - 0.5 * X2 + np.random.normal(0, 1 / 2, 100)
y = Y.reshape(-1, 1)

X = np.column_stack((X1, X2))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model with only X1
model_X1 = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    keras.layers.Dense(1)
])
model_X1.compile(optimizer='adam', loss='mse')
model_X1.fit(X_train[:, 0:1], y_train, epochs=100, verbose=0)

# Model with X1 and X2
model_X1X2 = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    keras.layers.Dense(1)
])
model_X1X2.compile(optimizer='adam', loss='mse')
model_X1X2.fit(X_train, y_train, epochs=100, verbose=0)



x1_range = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100)
x1_range_reshaped = x1_range.reshape(-1, 1)
x2_mean = np.array([0] * 100).reshape(-1, 1)
x1x2_range = np.hstack((x1_range_reshaped, x2_mean))

# Predictions
predictions_X1_line = model_X1.predict(x1_range_reshaped)
predictions_X1X2_line = model_X1X2.predict(x1x2_range)

phi = 1.619


# Graficar los resultados
plt.figure(figsize=(4 * phi, 4))
plt.scatter(X_scaled[:, 0], y, alpha=0.5, label="Linear Sample", s=10, color = shap.plots.colors.red_blue(0.75))
plt.plot(x1_range, predictions_X1_line, color=shap.plots.colors.red_blue(.99), label="Model with $x_1$", linewidth=2)
plt.plot(x1_range, predictions_X1X2_line,color=shap.plots.colors.red_blue(0.5), linewidth=2, label='Model with $x_1$ and fixed $x_2$')
plt.title('Comparative between model functions', fontweight='bold')
plt.xlabel('$x_1$')
plt.xlim(-1,1)
plt.ylim(-5,4.9)
plt.ylabel('$y$')
plt.legend()
plt.grid(False)
plt.show()