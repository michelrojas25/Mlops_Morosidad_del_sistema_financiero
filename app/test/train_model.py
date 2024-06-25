import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

df =pd.read_csv('datos/Morosidad del Sistema Financiero.csv')

# Imputar valores nulos en 'exp_sf' con la mediana
df['exp_sf'].fillna(df['exp_sf'].median(), inplace=True)
# Imputar valores nulos en 'linea_sf' con la mediana
df['linea_sf'].fillna(df['linea_sf'].median(), inplace=True)
# Imputar valores nulos en 'deuda_sf' con la mediana
df['deuda_sf'].fillna(df['deuda_sf'].median(), inplace=True)

# Seleccionar las columnas numéricas y categóricas
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('mora')
categorical_cols = df.select_dtypes(include=['object']).columns

# Dividir el conjunto de datos
X = df.drop('mora', axis=1)
y = df['mora']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir el ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Crear el pipeline con SMOTE y el clasificador
model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Entrenar el modelo
model.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Guardar el modelo en un archivo
joblib.dump(model, 'modelos/modelo_entrenado.pkl')

# Cargar el modelo desde el archivo
modelo_cargado = joblib.load('modelos/modelo_entrenado.pkl')

# Verificar que el modelo cargado funcione correctamente
y_pred_cargado = modelo_cargado.predict(X_test)
print("Accuracy del modelo cargado:", accuracy_score(y_test, y_pred_cargado))