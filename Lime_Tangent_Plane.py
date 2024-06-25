import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Datos de entrada
np.random.seed(90)
x_sequence_fine = np.arange(-10, 10.1, 0.1)
e_sequence_fine = np.random.normal(0, 0.5, len(x_sequence_fine))
y_sequence_fine = np.sin(x_sequence_fine) + e_sequence_fine

# Definición del modelo
model = Sequential([
    Dense(64, input_dim=1, activation='relu'),  # Capa oculta con 10 neuronas
    Dense(64,  activation='relu'),  # Capa oculta con 10 neuronas
    Dense(1, activation='linear')               # Capa de salida
])

# Compilación del modelo
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# Entrenamiento del modelo
model.fit(x_sequence_fine.reshape(-1, 1), y_sequence_fine, epochs=100, verbose=1)

# Predicciones del modelo
y_pred = model.predict(x_sequence_fine.reshape(-1, 1)).flatten()

# Creando la gráfica con las predicciones de la red neuronal
plt.figure(figsize=(10, 6))
plt.scatter(x_sequence_fine, y_sequence_fine, alpha=0.5, label="y = sin(x) + e", s=10)
plt.plot(x_sequence_fine, np.sin(x_sequence_fine), color='red', label="y = sin(x)", linewidth=2)
plt.plot(x_sequence_fine, y_pred, color='green', label="NN Prediction", linewidth=2)
plt.title("Comparación entre la predicción de la Red Neuronal y y = sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
