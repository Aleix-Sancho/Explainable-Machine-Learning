import numpy as np
import statsmodels.api as sm
from  SHAP_Framework_Implementations import *

np.random.seed(42)
beta0 = 2.0
beta1 = 10
beta2 = -4

num_samples = 100


np.random.seed(42)
x1 = np.random.normal(0, 2, num_samples)
np.random.seed(43)
x2 = np.random.normal(0, 2, num_samples)
np.random.seed(44)
x3 = np.random.normal(0, 2, num_samples)

np.random.seed(45)
error = np.random.normal(0, 1, num_samples)

X = np.column_stack((x1, x2, x3))


y = beta0 + beta1 * x1 + beta2 * x2  + error



import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

keras.utils.set_random_seed(42)

model = Sequential()
model.add(Dense(10, activation='relu', input_dim=X.shape[1]))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=100, batch_size=10, verbose=1)


import numpy as np
from sklearn.metrics import mean_squared_error

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
print(f"MSE: {mse}")



x = np.ones(model.input_shape[1])
x0 = np.zeros(model.input_shape[1])


SHAP_values = get_SHAP_values(model, X, X, x0)

Kernel_SHAP_values = get_Kernel_SHAP_values(model, X, X, x0)

Deep_SHAP_values = get_Deep_SHAP_Values(model, X, x0)

features_names = ["$x_1$", "$x_2$", "$x_3$"]


SHAP_plot(X, SHAP_values, "SHAP Values", features_names)