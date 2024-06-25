from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List
import psycopg2
from psycopg2.extras import Json

# Cargar el modelo
model = joblib.load('/app/model/modelo_entrenado.pkl')

app = FastAPI()

# Datos del modelo
class Data(BaseModel):
    edad: int
    zona: str
    ingreso: float
    nivel_educ: str
    dias_lab: int
    exp_sf: int
    nivel_ahorro: int
    linea_sf: float
    deuda_sf: float
    score: float
    clasif_sbs: int
    atraso: int
    vivienda: str

class PredictionRequest(BaseModel):
    data: List[Data]

# Conexión a la base de datos
def get_db_connection():
    conn = psycopg2.connect(
        dbname="mlops",
        user="user",
        password="password",
        host="db",
        port="5432"
    )
    return conn

@app.post("/predict")
def predict(request: PredictionRequest):
    # Convertir la solicitud a DataFrame
    df = pd.DataFrame([item.dict() for item in request.data])

    # Realizar la predicción
    predictions = model.predict(df)

    # Guardar predicciones en la base de datos
    conn = get_db_connection()
    cursor = conn.cursor()
    for item, prediction in zip(request.data, predictions):
        cursor.execute(
            """
            INSERT INTO predicciones (input, prediccion) VALUES (%s, %s)
            """,
            (Json(item.dict()), int(prediction))
        )
    conn.commit()
    cursor.close()
    conn.close()

    # Convertir las predicciones a una lista de tipos de datos estándar de Python
    predictions_list = [pred.item() for pred in predictions]
    
    return {"predictions": predictions_list}
