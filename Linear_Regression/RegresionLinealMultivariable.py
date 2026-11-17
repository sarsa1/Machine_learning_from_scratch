import pandas as pd 
import numpy as np

df = pd.read_csv('dataset_multivariable.csv')

X = df.drop(['Peso'], axis=1)
y = df[['Peso']]
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

w = np.zeros([len(X_train.columns), 1])
print(w.shape)

from sklearn.preprocessing import StandardScaler

X_Scaler = StandardScaler()
y_Scaler = StandardScaler()

X_train = X_Scaler.fit_transform(X_train)
y_train = y_Scaler.fit_transform(y_train)

X_test = X_Scaler.transform(X_test)
y_test = y_Scaler.transform(y_test)

b, lr = 0, 0.02

epochs = 500

historial = []

for epoch in range(epochs):
    y_pred = X_train @ w + b
    dw = (2/len(X_train)) * X_train.T @ (y_pred - y_train)
    db = (2/len(X_train)) * np.sum(y_pred - y_train)
    historial.append(np.mean((y_pred - y_train)**2))
    w = w - (lr * dw)
    b = b - (lr * db)

y_pred_test_kg = y_Scaler.inverse_transform(X_test @ w + b)
y_test_kg = y_Scaler.inverse_transform(y_test)

residuos = y_test_kg - y_pred_test_kg
rmse_kg = np.sqrt(np.mean(residuos**2))
r2 = 1 - np.sum(residuos**2) / np.sum((y_test_kg - y_test_kg.mean())**2)
print(f'RMSE en test: {rmse_kg:.2f} kg | R²: {r2:.3f}')

print(f'Peso: {w}, bias: {b}')

def predict(altura, edad, horas):
    parametros = pd.DataFrame({
        'Altura' : [altura],
        'Edad' : [edad],
        'Horas_deporte' : [horas]
        })
    parametros = X_Scaler.transform(parametros)
    peso = parametros @ w + b
    peso = y_Scaler.inverse_transform(peso)
    return peso.item()

altura_persona = float(input('Introduce la altura para predecir el peso de esa persona : '))
edad_persona = float(input('Introduce su edad para predecir el peso de esa persona : '))
horas_deporte_persona = float(input('Introduce las horas de deporte semanal para predecir el peso de esa persona : '))

print(f'El usuario con una altura de {altura_persona}cm, {edad_persona} años y {horas_deporte_persona} horas de deporte pesará unos {predict(altura_persona, edad_persona, horas_deporte_persona):.2f} kg')


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.plot(historial)
plt.xlabel('Época')
plt.ylabel('MSE (escala estandarizada)')
plt.title(f'Curva de aprendizaje (lr={lr})')
plt.yscale('log')   # escala logarítmica: se ven mejor los cambios pequeños al final
plt.grid(True)
plt.show()