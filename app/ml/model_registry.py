from __future__ import annotations

from pathlib import Path

import joblib

from app.core.settings import ARTIFACTS_DIR, PREPROCESSING_ARTIFACTS_DIR
from app.ml.types import ClassifierName


class ModelRegistryError(Exception):
    pass


def _artifact_path(classifier: ClassifierName) -> Path:
    return ARTIFACTS_DIR / f"{classifier.value}.joblib"


def _preprocessing_artifact_path(route_name: str) -> Path:
    return PREPROCESSING_ARTIFACTS_DIR / f"{route_name}.joblib"


def save_artifact(classifier: ClassifierName, artifact: dict) -> Path:
    path = _artifact_path(classifier)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_artifact(classifier: ClassifierName) -> dict:
    path = _artifact_path(classifier)
    if not path.exists():
        raise ModelRegistryError(
            f"No existe un modelo entrenado para '{classifier.value}'. Ejecuta /v1/train primero."
        )
    return joblib.load(path)


def save_preprocessing_artifact(route_name: str, artifact: dict) -> Path:
    path = _preprocessing_artifact_path(route_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_preprocessing_artifact(route_name: str) -> dict:
    path = _preprocessing_artifact_path(route_name)
    if not path.exists():
        raise ModelRegistryError(
            f"No existe un artefacto de preprocesamiento para '{route_name}'. Ejecuta /v1/train primero."
        )
    return joblib.load(path)


def list_available_models() -> list[str]:
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted(path.stem for path in ARTIFACTS_DIR.glob("*.joblib"))


def preprocessing_artifact_path(route_name: str) -> Path:
    return _preprocessing_artifact_path(route_name)


def list_models_using_preprocessing(route_name: str) -> list[str]:
    if not ARTIFACTS_DIR.exists():
        return []

    model_names: list[str] = []
    for path in sorted(ARTIFACTS_DIR.glob("*.joblib")):
        try:
            artifact = joblib.load(path)
        except Exception:
            continue
        if str(artifact.get("preprocessing_route", "")) == route_name:
            model_names.append(path.stem)

    return model_names
