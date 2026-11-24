"""Genera dataset_polinomico.csv: una variable x y una y con relación no lineal + ruido.

Pocos puntos a propósito, para que el sobreajuste con grados altos se note.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 30
RUIDO = 2.0

x = rng.uniform(-3, 3, N)
y = 0.5 * x**3 - 2 * x + 1 + rng.normal(0, RUIDO, N)

df = pd.DataFrame({'x': x.round(3), 'y': y.round(3)})
ruta = Path(__file__).parent / 'dataset_polinomico.csv'
df.to_csv(ruta, index=False)
print(f'Guardado {ruta} ({N} filas)')
