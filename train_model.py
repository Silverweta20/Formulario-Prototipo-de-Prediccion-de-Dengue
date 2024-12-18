import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Cargar los datos
data = pd.read_csv('dengue_data.csv')

# Convertir variables categóricas a numéricas
data['sexo'] = data['sexo'].map({'Male': 1, 'Female': 0})  # Codificando sexo

# Preprocesamiento de los datos
X = data.drop('diagnostico', axis=1)
y = data['diagnostico']

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrenar el modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Guardar el modelo entrenado
pickle.dump(modelo, open('dengue_model.pkl', 'wb'))

# Evaluar el modelo
accuracy = modelo.score(X_test, y_test)
print(f'Accuracy del modelo: {accuracy}')