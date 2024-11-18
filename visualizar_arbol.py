import joblib
from sklearn.tree import export_text

# Cargar el modelo
modelo = joblib.load(r'.\dengue_model.pkl')

# Extraer las reglas del árbol de decisiones
reglas_arbol = export_text(modelo.estimators_[0], 
                           feature_names=['temperatura', 'dolor_abdominal', 'edad', 'nauseas', 'altitud', 'fiebre', 
                                          'dolor_muscular', 'dolor_de_cabeza', 'diarrea', 'fatiga', 'sexo'])

# Mostrar las reglas del árbol
print(reglas_arbol)
