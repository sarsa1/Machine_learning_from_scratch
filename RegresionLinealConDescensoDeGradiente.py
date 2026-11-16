import matplotlib 
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('dataset_alturas_pesos.csv')

X = df[['Altura']]
y = df[['Peso']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

print(X_train.shape)
print(y_train.shape)

from sklearn.preprocessing import StandardScaler
X_Scaler = StandardScaler()
y_Scaler = StandardScaler()

X_train = X_Scaler.fit_transform(X_train)
y_train = y_Scaler.fit_transform(y_train)
X_test = X_Scaler.transform(X_test)
y_test = y_Scaler.transform(y_test)

w, b, lr = 0, 0, 0.02

epochs = 100

historial = [] # Para ver cuando esque el modelo empieza a aprender, (cuando veamos valores mas pequeños)



for epoch in range(epochs):
    y_pred = w * X_train + b
    dw = (1/len(X_train)) * 2 * X_train.T @ (y_pred - y_train)
    db = (1/len(X_train)) * 2 * np.sum(y_pred - y_train)
    historial.append(np.mean((y_pred - y_train)**2))
    w = w - (lr * dw)
    b = b - (lr * db)



# Volver a unidades reales (cm y kg) para que los resultados se entiendan
y_pred_test_kg = y_Scaler.inverse_transform(w * X_test + b)
y_test_kg = y_Scaler.inverse_transform(y_test)
X_test_cm = X_Scaler.inverse_transform(X_test)

residuos = y_test_kg - y_pred_test_kg
rmse_kg = np.sqrt(np.mean(residuos**2))
r2 = 1 - np.sum(residuos**2) / np.sum((y_test_kg - y_test_kg.mean())**2)
print(f'RMSE en test: {rmse_kg:.2f} kg | R²: {r2:.3f}')


def predict(altura):
    altura = pd.DataFrame({'Altura' : [altura]})
    altura = X_Scaler.transform(altura)
    peso = w * altura + b
    peso = y_Scaler.inverse_transform(peso)
    return peso.item()

alt = float(input('Introduce una altura para predecir el peso de la personas : '))
print(f'El usuario con una altura de {alt}cm pesará unos {predict(alt):.2f} kg')