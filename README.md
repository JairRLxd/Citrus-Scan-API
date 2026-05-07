# Citrus-Scan-API

API de inferencia para clasificacion de naranjas y limones usando 3 modelos preentrenados:

- `svm`
- `bayes`
- `perceptron`

En nube (PythonAnywhere) corre en modo **inferencia-only** por defecto.
No se entrena en servidor.

## Endpoints activos para app movil

- `GET /v1/health`
- `GET /v1/models`
- `GET /v1/preprocessing/status`
- `POST /v1/predict`

Endpoints bloqueados en nube:
- `POST /v1/train` -> `403`
- `POST /v1/preprocessing/recommendation` -> `403`

## Request de prediccion

`POST /v1/predict` (`multipart/form-data`):

- `file`: imagen
- `weight`: numero > 0
- `circumference`: numero > 0

La API ejecuta internamente los 3 clasificadores y devuelve los resultados de todos,
incluyendo `confidence_percent`.

## Requisitos

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Artefactos requeridos

Debes subir al repo (o servidor) los artefactos ya entrenados:

- `artifacts/models/svm.joblib`
- `artifacts/models/bayes.joblib`
- `artifacts/models/perceptron.joblib`
- `artifacts/preprocessing/shared_citrus_v1.joblib`

Sin esos archivos, `/v1/predict` no podra responder correctamente.

## Variables de entorno

- `APP_MODE`:
  - `inference` (default): bloquea entrenamiento y analisis de dataset.
  - `full`: habilita entrenamiento/analisis (uso local).

## Despliegue en PythonAnywhere

1. Crear virtualenv e instalar dependencias.
2. Configurar WSGI con `a2wsgi`:

```python
import sys
project_home = "/home/<USER>/Citrus-Scan-API"
if project_home not in sys.path:
    sys.path.insert(0, project_home)

from a2wsgi import ASGIMiddleware
from app.main import app

application = ASGIMiddleware(app)
```

3. En Web app, configurar variable de entorno:
- `APP_MODE=inference`

4. Reload y probar:

```bash
curl https://<USER>.pythonanywhere.com/v1/health
```

