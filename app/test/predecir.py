import pandas as pd
import joblib
from sklearn.metrics import classification_report

# Definir las columnas del DataFrame basado en el conjunto de datos original
columnas = ['atraso', 'vivienda', 'edad', 'dias_lab', 'exp_sf', 'nivel_ahorro', 'ingreso', 'linea_sf', 'deuda_sf', 
            'score', 'zona', 'clasif_sbs', 'nivel_educ']

# Cargar el modelo y el preprocesador
model = joblib.load('modelos/modelo_entrenado.pkl')
preprocessor = model.named_steps['preprocessor']

def predecir_manual(datos, y_true=None):
    """
    Realiza una predicción manual dado un diccionario de datos.

    Parameters:
    datos (dict): Un diccionario donde las llaves son los nombres de las columnas y los valores son los datos correspondientes.
    y_true (list): Una lista con las etiquetas verdaderas opcional para calcular las estadísticas.

    Returns:
    dict: Un diccionario con las predicciones y las estadísticas (si y_true es proporcionado).
    """
    # Convertir el diccionario a un DataFrame
    df = pd.DataFrame([datos])
    
    # Asegurarse de que las columnas están en el orden correcto y todas están presentes
    df = df.reindex(columns=columnas, fill_value=0)
    
    # Asegurarse de que las columnas categóricas están en el tipo correcto
    for col in ['vivienda', 'zona', 'nivel_educ']:
        df[col] = df[col].astype('category')
    
    # Preprocesar los datos
    X_preprocessed = preprocessor.transform(df)
    
    # Realizar la predicción
    prediccion = model.named_steps['classifier'].predict(X_preprocessed)
    
    # Inicializar el resultado
    resultado = {"prediccion": prediccion.tolist()}
    
    # Calcular y agregar las estadísticas si y_true es proporcionado
    if y_true is not None:
        reporte = classification_report(y_true, prediccion, output_dict=True, zero_division=1)
        resultado["estadisticas"] = reporte
    
    # Devolver la predicción y las estadísticas (si las hay)
    return resultado

# Ejemplo de uso de la función con un conjunto más grande
datos_ejemplos = [
    {
        'atraso': 10,
        'vivienda': 'FAMILIAR',
        'edad': 35,
        'dias_lab': 5000,
        'exp_sf': 24,
        'nivel_ahorro': 6,
        'ingreso': 4000,
        'linea_sf': 10000,
        'deuda_sf': 3000,
        'score': 200,
        'zona': 'Lima',
        'clasif_sbs': 0,
        'nivel_educ': 'UNIVERSITARIA'
    },
    {
        'atraso': 20,
        'vivienda': 'ALQUILADA',
        'edad': 50,
        'dias_lab': 7000,
        'exp_sf': 50,
        'nivel_ahorro': 4,
        'ingreso': 7000,
        'linea_sf': 15000,
        'deuda_sf': 5000,
        'score': 150,
        'zona': 'Lima',
        'clasif_sbs': 1,
        'nivel_educ': 'TECNICA'
    }
]

# Etiquetas verdaderas para los ejemplos
y_true_ejemplos = [1, 0]  # Proporciona las etiquetas verdaderas correspondientes aquí

# Realizar predicciones y obtener estadísticas
for datos, y_true in zip(datos_ejemplos, y_true_ejemplos):
    resultado = predecir_manual(datos, y_true=[y_true])
    print("Predicción:", resultado["prediccion"])
    if "estadisticas" in resultado:
        print("Estadísticas:")
        for clase, stats in resultado["estadisticas"].items():
            if isinstance(stats, dict):
                print(f"Clase {clase}:")
                for metric, value in stats.items():
                    print(f"  {metric}: {value}")
            else:
                print(f"{clase}: {stats}")
