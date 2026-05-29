from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

# Crear aplicación Flask
app = Flask(__name__)

# Leer datos
datos = pd.read_csv("datos.csv")

# Variables
X = datos[["publicidad"]]
y = datos["ventas"]

# Modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Página principal
@app.route("/")
def inicio():
    return render_template("index.html")

# Ruta de predicción
@app.route("/predecir", methods=["POST"])
def predecir():

    publicidad = float(request.form["publicidad"])

    prediccion = modelo.predict([[publicidad]])

    resultado = round(prediccion[0], 2)

    return render_template(
        "index.html",
        resultado=resultado
    )

# Ejecutar servidor
if __name__ == "__main__":
    app.run(debug=True)