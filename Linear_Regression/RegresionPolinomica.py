import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('dataset_polinomico.csv')

X = df[['x']].to_numpy()
y = df[['y']].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Copias en escala original solo para las gráficas (X_train/X_test se expanden y escalan después)
X_train_plot, X_test_plot = X_train.copy(), X_test.copy()
y_train_plot, y_test_plot = y_train.copy(), y_test.copy()

GRADO = 3

def expandir(X, grado):
    
    expansion = np.arange(1, grado + 1)

    X = X ** expansion

    return X


X_train = expandir(X_train, GRADO)
X_test = expandir(X_test, GRADO)
w = np.zeros([GRADO, 1])

from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)

X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test)

b, lr, L, epochs = 0, 0.02, 0.01, 100

historial_de_mejora = []

for _ in range(epochs):
    y_pred = X_train @ w + b
    dw = (2/len(X_train)) * X_train.T @ (y_pred - y_train) + (w * L * 2) # Aplicamos ridge a la derivada
    db = (2/len(X_train)) * np.sum(y_pred - y_train)
    error = np.mean((y_pred - y_train)**2)
    historial_de_mejora.append(error)
    w = w - (lr * dw)
    b = b - (lr * db)


y_pred_test_kg = y_scaler.inverse_transform(X_test @ w + b)
y_test_kg = y_scaler.inverse_transform(y_test)

residuos = y_test_kg - y_pred_test_kg
rmse_kg = np.sqrt(np.mean(residuos**2))
r2 = 1 - np.sum(residuos**2) / np.sum((y_test_kg - y_test_kg.mean())**2)
print(f'RMSE en test: {rmse_kg:.2f} kg | R²: {r2:.3f}')


def predict(x_nuevos):
    
    x_nuevos = expandir(x_nuevos, GRADO)
    x_nuevos = X_scaler.transform(x_nuevos)
    y_predictora = x_nuevos @ w + b
    y_predictora = y_scaler.inverse_transform(y_predictora)
   
    return y_predictora




# ---------- Gráficas ----------
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x_curva = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_train_plot, y_train_plot, label='train')
ax1.scatter(X_test_plot, y_test_plot, label='test', marker='x')
ax1.plot(x_curva, predict(x_curva), color='red', label=f'modelo (grado {GRADO})')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True)

ax2.plot(historial_de_mejora)
ax2.set_xlabel('Época')
ax2.set_ylabel('MSE (escala estandarizada)')
ax2.set_yscale('log')
ax2.set_title('Curva de aprendizaje')
ax2.grid(True)

plt.tight_layout()
plt.show()
