import pandas as pd
from sklearn.linear_model import LinearRegression

# Leer datos
datos = pd.read_csv("datos.csv")

# Variables
X = datos[["publicidad"]]
y = datos["ventas"]

# Crear modelo
modelo = LinearRegression()

# Entrenar modelo
modelo.fit(X, y)

# Predicción
prediccion = modelo.predict([[120]])

print("Predicción:", prediccion[0])