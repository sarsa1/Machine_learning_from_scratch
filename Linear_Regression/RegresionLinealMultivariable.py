import pandas as pd 
import numpy as np

df = pd.read_csv('dataset_multivariable.csv')

X = df.drop(['Peso'], axis=1)
y = df[['Peso']]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

w = np.zeros([len(X_train.columns), 1])

L = np.full([1, 1], 0.01)

#Valores límite que se le pueden introducir al modelo para predecir
minimo_valor_permitido_altura = df['Altura'].min()
maximo_valor_permitido_altura = df['Altura'].max()

minimo_valor_permitido_edad = df['Edad'].min()
maximo_valor_permitido_edad = df['Edad'].max()

minimo_valor_permitido_horas_deporte = df['Horas_deporte'].min()
maximo_valor_permitido_horas_deporte = df['Horas_deporte'].max()
#Valores límite que se le pueden introducir al modelo para predecir



from sklearn.preprocessing import StandardScaler

X_Scaler = StandardScaler()
y_Scaler = StandardScaler()

X_train = X_Scaler.fit_transform(X_train)
y_train = y_Scaler.fit_transform(y_train)

X_test = X_Scaler.transform(X_test)
y_test = y_Scaler.transform(y_test)

b, lr = 0, 0.02




epochs = 90

historial = []

for epoch in range(epochs):
    y_pred = X_train @ w + b
    dw = (2/len(X_train)) * X_train.T @ (y_pred - y_train) + (w @ L * 2)
    db = (2/len(X_train)) * np.sum(y_pred - y_train)
    error = np.mean((y_pred - y_train)**2)
    historial.append(error)
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

while True:
    
    try: 

        altura_persona = float(input('Introduce la altura para predecir el peso de esa persona : '))
       
        if altura_persona > maximo_valor_permitido_altura or altura_persona < minimo_valor_permitido_altura:
            raise ValueError(f'La altura {altura_persona}cm es inválida no está en el rango [{minimo_valor_permitido_altura} - {maximo_valor_permitido_altura}] cm')

        edad_persona = float(input('Introduce su edad para predecir el peso de esa persona : '))

        if edad_persona > maximo_valor_permitido_edad or edad_persona < minimo_valor_permitido_edad:
                    raise ValueError(f'La edad {edad_persona} años es inválida no está en el rango [{minimo_valor_permitido_edad} - {maximo_valor_permitido_edad}] años')

        horas_deporte_persona = float(input('Introduce las horas de deporte semanal para predecir el peso de esa persona : '))

        if horas_deporte_persona > maximo_valor_permitido_horas_deporte or horas_deporte_persona < minimo_valor_permitido_horas_deporte:
                    raise ValueError(f'La cantidad de {horas_deporte_persona} horas es inválida no está en el rango [{minimo_valor_permitido_horas_deporte} - {maximo_valor_permitido_horas_deporte}] horas')
        
    except ValueError as e:
        print(e)
        continue
    else:
        break 


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