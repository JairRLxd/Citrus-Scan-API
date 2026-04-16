# Citrus-Scan-API

API académica para clasificación de naranjas y limones con soporte para 3 clasificadores:

- `svm`
- `bayes`
- `perceptron`

La app móvil envía:
- imagen de fruta
- peso
- circunferencia

## Arquitectura

```text
app/
  api/
    routes.py          # Endpoints HTTP
    schemas.py         # Contratos de entrada/salida
  ml/
    dataset_loader.py  # Carga y unión imagen + CSV
    feature_extractor.py
    model_factory.py   # Fábrica de clasificadores
    model_registry.py  # Persistencia de artefactos
    analyzer.py        # Análisis + recomendaciones de preprocesamiento
    service.py         # Casos de uso: train/predict
  core/
    settings.py
  main.py
artifacts/models/      # Modelos entrenados por clasificador
artifacts/preprocessing/ # Pipeline compartido de preprocesamiento
```

## Estructura esperada del dataset

```text
dataset/
  limon/
    limon_001.jpg
    limon_002.jpg
  naranja/
    naranja_001.jpg
    naranja_002.jpg
```

CSV (ejemplo de columnas, se detectan variantes similares):

```csv
imagen,peso,circunferencia
limon_001.jpg,95.4,14.2
naranja_001.jpg,182.1,23.4
```

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar

```bash
uvicorn app.main:app --reload
```

Docs:
- Swagger: http://127.0.0.1:8000/docs

## Endpoints

- `GET /v1/health`
- `GET /v1/models`
- `GET /v1/preprocessing/status`
- `POST /v1/train`
- `POST /v1/predict`
- `POST /v1/preprocessing/recommendation`

### Entrenar un clasificador

```bash
curl -X POST "http://127.0.0.1:8000/v1/train" \
  -H "Content-Type: application/json" \
  -d '{
    "classifier": "svm",
    "dataset_dir": "./dataset",
    "csv_path": "./dataset/CSV_LIM_NAR_unificado.csv",
    "test_size": 0.2,
    "random_state": 42
  }'
```

### Predecir desde la app (imagen + medidas)

```bash
curl -X POST "http://127.0.0.1:8000/v1/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "weight=176.5" \
  -F "circumference=22.1" \
  -F "file=@./ejemplo.jpg"
```

Este endpoint ejecuta internamente los 3 clasificadores (`svm`, `bayes`, `perceptron`)
y devuelve los resultados de cada uno con su `confidence_percent`.

### Obtener recomendaciones de preprocesamiento

```bash
curl -X POST "http://127.0.0.1:8000/v1/preprocessing/recommendation" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_dir": "./dataset",
    "csv_path": "./dataset/CSV_LIM_NAR_unificado.csv"
  }'
```

Nota: el cargador acepta `id` en CSV como identificador de imagen aunque no tenga extensión
(por ejemplo `02_01L`) y lo empata automáticamente con archivos como `02_01L.jpeg` o `04_01_L.jpeg`.

La respuesta de este endpoint incluye:
- `summary`: resumen estadístico y calidad de datos.
- `shared_route`: ruta base recomendada para los 3 clasificadores.
- `recommendations`: ajustes puntuales por clasificador sin cambiar la ruta base.

### Ver estado del pipeline compartido

```bash
curl -X GET "http://127.0.0.1:8000/v1/preprocessing/status"
```

## Como Se Eligio La Ruta De Preprocesamiento

La ruta `shared_citrus_v1` no se eligio de forma arbitraria ni por un solo modelo. Se definio
para ser comun y estable en los 3 clasificadores (`svm`, `bayes`, `perceptron`) a partir de:

- analisis del dataset real (`POST /v1/preprocessing/recommendation`)
- restricciones compartidas de los 3 algoritmos (misma entrada y mismas transformaciones base)
- control de calidad de datos (unidades, outliers, empate imagen-CSV)

Flujo base adoptado:

1. Empate robusto entre imagen y CSV por ID normalizado.
2. Uso exclusivo de muestras con metadata valida (peso/circunferencia numericos).
3. Imagen a RGB, resize `128x128`, normalizacion de pixeles `[0,1]`.
4. Extraccion de features de color (RGB/HSV + histograma de tono).
5. Anexado de `peso` y `circunferencia`.
6. Normalizacion de unidades de peso (si `peso < 5`, se asume kg y se convierte a g).
7. Winsorizacion de variables numericas (`p1-p99`).
8. Split estratificado train/test.
9. Ajuste de transformadores solo en train (sin data leakage).
10. Escalado con `StandardScaler` para una base comun.
11. PCA opcional (apagado por default) para escenarios de alta correlacion.

## Estado Actual (16 Abril 2026)

Lo que ya esta hecho:

- Arquitectura separada por capas (`api`, `ml`, `core`).
- Endpoint de analisis y recomendacion de preprocesamiento.
- Pipeline compartido implementado como artefacto reutilizable:
  - `fit/transform/save/load`
  - mismo preprocesamiento en entrenamiento e inferencia
- Entrenamiento y prediccion funcionales para los 3 clasificadores.
- Persistencia separada:
  - `artifacts/preprocessing/shared_citrus_v1.joblib`
  - `artifacts/models/{svm,bayes,perceptron}.joblib`
- Endpoint de estado:
  - `GET /v1/preprocessing/status`
  - muestra si el pipeline existe, configuracion, fecha de ajuste y modelos que lo usan.

Resultados de entrenamiento reportados actualmente:

- `svm`: `accuracy=0.96512`, `f1_weighted=0.96510`
- `bayes`: `accuracy=0.93023`, `f1_weighted=0.93023`
- `perceptron`: `accuracy=0.95349`, `f1_weighted=0.95349`

Nota importante:
- Estos resultados se documentan como evidencia de que la ruta compartida funciona.
- No se uso esta corrida para "elegir un ganador", sino para validar que los 3 consumen el mismo preprocesamiento.

## Ajuste De Algoritmos

Archivo principal para ajustar hiperparametros:
- `app/ml/model_factory.py`

Reglas para mantener comparabilidad:
- conservar la misma ruta comun en `app/ml/preprocessor.py`
- ajustar solo los modelos en `model_factory.py`
- no romper el flujo `train/predict` de `app/ml/service.py`

Mejoras recomendadas para mejores resultados:

1. SVM:
- probar `kernel=rbf` y `kernel=poly`
- hacer busqueda de `C` y `gamma` con validacion cruzada estratificada
- evaluar `class_weight='balanced'` si cambia el balance del dataset

2. Naive Bayes:
- ajustar `var_smoothing` en escala logaritmica
- probar PCA ligero previo para reducir correlacion entre features

3. Perceptron:
- ajustar `max_iter`, `tol`, `eta0`, `penalty` y `alpha`
- activar early stopping y revisar convergencia por semilla

4. Evaluacion:
- usar validacion cruzada estratificada (k-fold) para estabilidad
- reportar `accuracy`, `f1_weighted` y matriz de confusion
- repetir corridas con varias semillas y reportar promedio/desviacion

5. Datos y features:
- revisar muestras con ruido visual (desenfoque, fondo dominante)
- probar variantes de features de color/texture manteniendo la misma ruta base
- validar consistencia de unidades en peso y circunferencia

Check rapido despues de cambios:

1. Reentrenar los 3 clasificadores con el mismo dataset.
2. Consultar `GET /v1/preprocessing/status` y confirmar `shared_citrus_v1`.
3. Verificar que cualquier mejora venga del algoritmo, no de cambiar el preprocesamiento.
