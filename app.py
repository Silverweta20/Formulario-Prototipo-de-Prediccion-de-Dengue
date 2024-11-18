from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo preentrenado
modelo = joblib.load('dengue_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resultados', methods=['POST'])
def resultados():
    # Obtener los datos del formulario
    edad = int(request.form['edad'])
    sexo = 1 if request.form['sexo'] == 'Male' else 0
    temperatura = float(request.form['temperatura'])
    altitud = int(request.form['altitud'])
    
    fiebre = 1 if 'fiebre' in request.form else 0
    dolor_muscular = 1 if 'dolor_muscular' in request.form else 0
    dolor_de_cabeza = 1 if 'dolor_de_cabeza' in request.form else 0
    dolor_abdominal = 1 if 'dolor_abdominal' in request.form else 0
    diarrea = 1 if 'diarrea' in request.form else 0
    nauseas = 1 if 'nauseas' in request.form else 0
    fatiga = 1 if 'fatiga' in request.form else 0
    
    # Crear el vector de características
    X = np.array([[edad, sexo, temperatura, altitud, fiebre, dolor_muscular, dolor_de_cabeza, dolor_abdominal, diarrea, nauseas, fatiga]])

    # Realizar la predicción
    prediccion = modelo.predict(X)
    
    # Mostrar el resultado
    resultado = "Positivo" if prediccion[0] == 1 else "Negativo"
    return f"El diagnóstico es: {resultado}"

if __name__ == '__main__':
    app.run(debug=True)