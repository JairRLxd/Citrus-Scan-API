from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev

import numpy as np

from app.ml.dataset_loader import DatasetRecord
from app.ml.types import ClassifierName


def _ratio(values: list[int]) -> float:
    min_v = min(values)
    max_v = max(values)
    return float(max_v / min_v) if min_v else float("inf")


def _iqr_outlier_count(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = q3 - q1
    lower = float(q1 - 1.5 * iqr)
    upper = float(q3 + 1.5 * iqr)
    count = int(((arr < lower) | (arr > upper)).sum())
    return lower, upper, count


def analyze_records(records: list[DatasetRecord]) -> dict:
    labels = [r.label for r in records]
    weights = [r.weight for r in records]
    circumferences = [r.circumference for r in records]

    distribution = Counter(labels)
    imbalance_ratio = _ratio(list(distribution.values()))

    w_low, w_high, w_outliers = _iqr_outlier_count(weights)
    c_low, c_high, c_outliers = _iqr_outlier_count(circumferences)

    # Regla académica simple: pesos menores a 5 suelen venir en kg y no en gramos.
    possible_kg_entries = int(np.sum(np.asarray(weights, dtype=np.float64) < 5.0))

    return {
        "sample_count": len(records),
        "class_distribution": dict(distribution),
        "imbalance_ratio": round(imbalance_ratio, 3),
        "weight": {
            "mean": round(mean(weights), 4),
            "std": round(pstdev(weights), 4),
            "min": round(min(weights), 4),
            "max": round(max(weights), 4),
        },
        "circumference": {
            "mean": round(mean(circumferences), 4),
            "std": round(pstdev(circumferences), 4),
            "min": round(min(circumferences), 4),
            "max": round(max(circumferences), 4),
        },
        "data_quality": {
            "possible_kg_entries": possible_kg_entries,
            "weight_iqr_bounds": [round(w_low, 4), round(w_high, 4)],
            "circumference_iqr_bounds": [round(c_low, 4), round(c_high, 4)],
            "weight_outliers_iqr": w_outliers,
            "circumference_outliers_iqr": c_outliers,
        },
    }


def _shared_route_steps(summary: dict) -> list[str]:
    quality = summary["data_quality"]

    steps = [
        "1) Unificar IDs de imagen (quitar guiones bajos/extensión y usar minúsculas) para empatar imagen↔CSV sin pérdida.",
        "2) Mantener solo muestras con metadata válida (peso + circunferencia numéricos).",
        "3) Convertir imagen a RGB, redimensionar a 128x128 y normalizar pixeles a [0,1].",
        "4) Extraer descriptores de color globales (RGB/HSV + histograma de tono) y anexar peso/circunferencia.",
    ]

    if quality["possible_kg_entries"] > 0:
        steps.append(
            "5) Estandarizar unidades de peso antes de entrenar (si peso < 5, convertir kg→g multiplicando por 1000 y registrar el cambio)."
        )
    else:
        steps.append("5) Verificar consistencia de unidades de peso (todo en gramos).")

    steps.extend(
        [
            "6) Tratar outliers numéricos con winsorización por percentiles (p1-p99) antes del escalado.",
            "7) Separar train/test con estratificación por clase.",
            "8) Ajustar transformadores solo con train (evitar fuga de información).",
            "9) Escalar features con StandardScaler para usar una base homogénea en SVM, Bayes y Perceptron.",
            "10) Opcional recomendado: PCA (95%-98% varianza) si se detecta alta correlación entre features.",
        ]
    )

    return steps


def build_shared_preprocessing_route(records: list[DatasetRecord]) -> dict:
    summary = analyze_records(records)

    return {
        "route_name": "shared_citrus_v1",
        "objective": "Ruta única de preprocesamiento compatible con SVM, Naive Bayes y Perceptron.",
        "steps": _shared_route_steps(summary),
        "validation_checks": [
            "Comparar distribución de peso/circunferencia antes vs después de winsorización.",
            "Verificar que el split estratificado conserve proporciones de clase.",
            "Medir estabilidad con validación cruzada estratificada (mismo preprocesamiento para los 3 algoritmos).",
        ],
    }


def preprocessing_recommendations(records: list[DatasetRecord]) -> dict[str, list[str]]:
    analysis = analyze_records(records)
    imbalance_ratio = analysis["imbalance_ratio"]

    recommendations: dict[str, list[str]] = {}

    for classifier in ClassifierName:
        steps: list[str] = [
            "Usar la misma ruta compartida de limpieza, extracción de features y escalado para mantener comparabilidad.",
            "Asegurar que todo preprocesamiento se ajuste solo con train y luego se aplique a test/inferencia.",
        ]

        if imbalance_ratio > 1.5:
            steps.append(
                "Aplicar balanceo de clases (class_weight='balanced' u over/under-sampling)."
            )

        if classifier == ClassifierName.svm:
            steps.append("Después del preprocesamiento compartido, ajustar C y gamma con CV.")

        if classifier == ClassifierName.bayes:
            steps.append(
                "Si hay alta correlación entre features, activar PCA ligero (95%-98% varianza) tras el escalado."
            )

        if classifier == ClassifierName.perceptron:
            steps.append(
                "Mantener early stopping y revisar convergencia (max_iter/tol) con el mismo preprocesamiento base."
            )

        recommendations[classifier.value] = steps

    return recommendations
