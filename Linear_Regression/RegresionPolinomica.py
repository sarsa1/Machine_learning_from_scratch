import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split



df = pd.read_csv('dataset_multivariable.csv')

X = df.drop(['Peso'], axis=1)
y = df[['Peso']]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

w = np.zeros([len(X_train.columns), 1])


from sklearn.preprocessing import StandardScaler

X_Scaler = StandardScaler()
y_Scaler = StandardScaler()

X_train = X_Scaler.fit_transform(X_train)
y_train = y_Scaler.fit_transform(y_train)

X_test = X_Scaler.transform(X_test)
y_test = y_Scaler.transform(y_test)


b, lr = 0, 0.02

epochs = 200

for _ in range(epochs):
    y_pred = X_train @ w + b