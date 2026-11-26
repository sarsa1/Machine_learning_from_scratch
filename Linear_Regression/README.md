# Regresión lineal desde cero

Implementación de modelos de regresión lineal con NumPy, sin recurrir a los estimadores de scikit-learn. El objetivo es entender y escribir la matemática de cada modelo: la ecuación normal, el descenso de gradiente, la regularización y la expansión polinómica de características.

scikit-learn se usa solo para la partición train/test (`train_test_split`) y la estandarización (`StandardScaler`).

## Contenido

| Script | Modelo | Método de ajuste | Dataset |
| --- | --- | --- | --- |
| `RegresionLineal.py` | Lineal simple | Ecuación normal | `dataset_alturas_pesos.csv` |
| `RegresionLinealConDescensoDeGradiente.py` | Lineal simple | Descenso de gradiente | `dataset_alturas_pesos.csv` |
| `RegresionLinealMultivariable.py` | Lineal multivariable | Descenso de gradiente con regularización L2 | `dataset_multivariable.csv` |
| `RegresionPolinomica.py` | Polinómica (grado 3) | Descenso de gradiente con regularización L2 | `dataset_polinomico.csv` |
| `generar_dataset_polinomico.py` | Generador de datos sintéticos | — | — |

## Modelos

### 1. Regresión lineal simple: ecuación normal

Predice el peso a partir de la altura. Los parámetros se obtienen en forma cerrada:

$$\theta = (X^\top X)^{-1} X^\top y$$

donde a $X$ se le añade una columna de unos para incluir el término independiente. Muestra la recta ajustada sobre los conjuntos de entrenamiento y test.

### 2. Regresión lineal simple: descenso de gradiente

El mismo problema, resuelto de forma iterativa. Las variables se estandarizan antes de entrenar y en cada época se minimiza el error cuadrático medio:

$$w \leftarrow w - \alpha \frac{\partial \text{MSE}}{\partial w}, \qquad b \leftarrow b - \alpha \frac{\partial \text{MSE}}{\partial b}$$

Al final las predicciones se devuelven a unidades reales (cm y kg). Incluye una función `predict` interactiva.

### 3. Regresión lineal multivariable

Predice el peso a partir de tres variables: altura, edad y horas de deporte semanales. La forma es vectorial ($\hat{y} = Xw + b$) y la regularización L2 (Ridge) se añade al gradiente:

$$\nabla_w = \frac{2}{n} X^\top (\hat{y} - y) + 2\lambda w$$

La entrada interactiva comprueba que cada valor esté dentro del rango observado en el dataset, para no extrapolar. Muestra la curva de aprendizaje en escala logarítmica.

### 4. Regresión polinómica

Ajusta una relación no lineal expandiendo la entrada a $[x, x^2, x^3]$ y aplicando después regresión lineal regularizada sobre esas características. Muestra la curva del modelo sobre los datos y la curva de aprendizaje.

El dataset es sintético y pequeño (30 puntos) a propósito, para que el sobreajuste se note al subir el grado. Se puede regenerar con:

```bash
python generar_dataset_polinomico.py
```

## Evaluación

Todos los modelos se evalúan sobre un conjunto de test del 20 % (`random_state=42`) con:

- **MSE / RMSE**: error en las unidades de la variable objetivo.
- **R²**: proporción de la varianza explicada por el modelo.

Las métricas se calculan a mano con NumPy.

## Datasets

| Archivo | Filas | Variables |
| --- | --- | --- |
| `dataset_alturas_pesos.csv` | 100 | `Altura`, `Peso` |
| `dataset_multivariable.csv` | 500 | `Altura`, `Edad`, `Horas_deporte`, `Peso` |
| `dataset_polinomico.csv` | 30 | `x`, `y` |

## Requisitos

- Python 3.10+
- NumPy
- pandas
- Matplotlib (backend `TkAgg`, requiere Tkinter)
- scikit-learn

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Uso

Los scripts leen los CSV con rutas relativas, así que se ejecutan desde esta carpeta:

```bash
cd Linear_Regression
python RegresionLineal.py
python RegresionLinealConDescensoDeGradiente.py
python RegresionLinealMultivariable.py
python RegresionPolinomica.py
```
