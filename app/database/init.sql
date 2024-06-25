CREATE TABLE predicciones (
    id SERIAL PRIMARY KEY,
    input JSONB,
    prediccion INTEGER,
    descripcion TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
