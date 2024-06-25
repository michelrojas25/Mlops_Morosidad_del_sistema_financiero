import pandas as pd
import json
import requests

df = pd.read_csv('E:/Proyectos/Project-Mlops-kedro-prefect-mlflow/Mlops-project/app/datos/Morosidad del Sistema Financiero.csv')

# Reordena las columnas en el orden deseado
column_order = [
    "edad", "zona", "ingreso", "nivel_educ", "dias_lab", "exp_sf",
    "nivel_ahorro", "linea_sf", "deuda_sf", "score", "clasif_sbs",
    "atraso", "vivienda"
]

df = df[column_order]
# obtemos las 10 filas aleatorias
df_random_sample = df.sample(n=10)
# Convertir el DataFrame a una lista de diccionarios
data_list = df_random_sample.to_dict(orient='records')

# Crear el payload para la solicitud
payload = {"data": data_list}

# Realizar la solicitud POST
response = requests.post("http://localhost:8000/predict", json=payload)

# Imprimir la respuesta
print(response.json())