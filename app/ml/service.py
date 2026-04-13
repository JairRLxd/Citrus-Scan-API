from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from app.ml.dataset_loader import DatasetError, DatasetRecord, load_dataset_records
from app.ml.feature_extractor import build_feature_vector, read_image_from_bytes
from app.ml.model_factory import make_classifier, to_probabilities
from app.ml.model_registry import (
    load_artifact,
    load_preprocessing_artifact,
    save_artifact,
    save_preprocessing_artifact,
)
from app.ml.preprocessor import SharedPreprocessor
from app.ml.types import ClassifierName


class TrainingError(Exception):
    pass


def _records_to_matrix(records: list[DatasetRecord]) -> tuple[np.ndarray, np.ndarray]:
    X_rows: list[np.ndarray] = []
    y: list[str] = []

    for record in records:
        with Image.open(record.image_path) as image:
            features = build_feature_vector(image, record.weight, record.circumference)
        X_rows.append(features)
        y.append(record.label)

    return np.vstack(X_rows), np.asarray(y)


def train(
    classifier: ClassifierName,
    dataset_dir: str,
    csv_path: str,
    test_size: float,
    random_state: int,
) -> dict:
    try:
        records = load_dataset_records(dataset_dir=dataset_dir, csv_path=csv_path)
    except DatasetError:
        raise
    except Exception as exc:
        raise TrainingError(f"Error cargando dataset: {exc}") from exc

    y_all = np.asarray([record.label for record in records])
    train_records, test_records = train_test_split(
        records,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all,
    )

    X_train_raw, y_train = _records_to_matrix(train_records)
    X_test_raw, y_test = _records_to_matrix(test_records)

    preprocessor = SharedPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    model = make_classifier(classifier=classifier, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))

    classes = sorted({str(label) for label in y_all})
    preprocessing_route = preprocessor.config.route_name
    preprocessing_artifact_path = save_preprocessing_artifact(
        preprocessing_route, preprocessor.to_artifact()
    )

    artifact = {
        "classifier": classifier.value,
        "model": model,
        "classes": classes,
        "feature_version": "v2_shared_preprocessing_pipeline",
        "preprocessing_route": preprocessing_route,
        "preprocessing_artifact_path": str(preprocessing_artifact_path),
    }
    artifact_path = save_artifact(classifier, artifact)

    return {
        "classifier": classifier.value,
        "artifact_path": str(artifact_path),
        "classes": classes,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "accuracy": round(accuracy, 5),
        "f1_weighted": round(f1, 5),
    }


def predict(
    classifier: ClassifierName,
    image_bytes: bytes,
    weight: float,
    circumference: float,
) -> dict:
    artifact = load_artifact(classifier)
    model = artifact["model"]
    classes: list[str] = artifact["classes"]
    preprocessing_route = str(artifact.get("preprocessing_route", "shared_citrus_v1"))
    preprocessing_artifact = load_preprocessing_artifact(preprocessing_route)
    preprocessor = SharedPreprocessor.from_artifact(preprocessing_artifact)

    image = read_image_from_bytes(image_bytes)
    vector_raw = build_feature_vector(image, weight, circumference).reshape(1, -1)
    vector = preprocessor.transform(vector_raw)

    probs = to_probabilities(model, vector, classes)
    best_idx = int(np.argmax(probs))

    return {
        "classifier": classifier.value,
        "predicted_label": classes[best_idx],
        "confidence": float(probs[best_idx]),
        "class_probabilities": {
            class_name: float(prob) for class_name, prob in zip(classes, probs)
        },
    }
