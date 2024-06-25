# Mlops_Morosidad_del_sistema_financiero
# Sistema Predictivo de Morosidad Financiera con Docker y FastAPI

## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un sistema predictivo de morosidad financiera utilizando técnicas de Ciencia de Datos y Machine Learning. El modelo de aprendizaje automático implementado predice la probabilidad de que un cliente incurra en morosidad en el sistema financiero. Para asegurar la eficacia y mantenibilidad del modelo, se ha integrado un enfoque robusto de MLOps que incluye el uso de Docker y FastAPI.

## Arquitectura del Proyecto

La arquitectura del proyecto se puede resumir en el siguiente diagrama:

![Arquitectura del Sistema](https://drive.google.com/drive/folders/1BzFFJumu1Yqc8hW3YTzI4Gkdv0XLzcM6)
<img src="https://drive.google.com/file/d/1ZBghYVORhX1zZzybSYlu8TFoDh3wkjp-/view?usp=drive_link" alt="Descripción opcional" width="300">

### Componentes Principales:

1. **Entrenamiento del Modelo:**
    - **Script:** `train_model.py`
    - **Dependencias:** Scikit-learn, Pandas, Joblib, Imbalanced-learn
    - **Datos:** `/app/datos/Morosidad del Sistema Financiero.csv`
    - **Modelo entrenado:** `/app/modelo/modelo_entrenado.pkl`
    - **Proceso:** Lectura de datos, preprocesamiento, entrenamiento del modelo, y guardado del modelo entrenado.

2. **API de Predicción:**
    - **Framework:** FastAPI
    - **Servidor:** Uvicorn
    - **Script:** `serve_model.py`
    - **Dependencias:** FastAPI, Uvicorn, Joblib, Scikit-learn, Pandas, Imbalanced-learn, Psycopg2-binary
    - **Endpoint:** `/predict`
    - **Proceso:** Carga del modelo, recepción de datos, conversión a DataFrame, predicción, y devolución del resultado.

3. **Base de Datos:**
    - **Tipo:** PostgreSQL
    - **Tabla:** `predicciones`
    - **Columnas:** `id`, `input`, `prediccion`, `timestamp`
    - **Script de inicialización:** `init.sql`

4. **Orquestación y Contenedorización:**
    - **Herramienta:** Docker Compose
    - **Servicios:**
        - `training`: Entrenamiento del modelo
        - `api`: Servidor de la API de predicción
        - `db`: Base de datos PostgreSQL
    - **Archivos:** `Dockerfile`, `docker-compose.yml`

## Ejecución del Proyecto

