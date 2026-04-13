from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron

from app.ml.types import ClassifierName


def make_classifier(classifier: ClassifierName, random_state: int) -> BaseEstimator:
    if classifier == ClassifierName.svm:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=2.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if classifier == ClassifierName.bayes:
        return GaussianNB()

    if classifier == ClassifierName.perceptron:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    Perceptron(
                        max_iter=3000,
                        tol=1e-3,
                        eta0=0.01,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    raise ValueError(f"Clasificador no soportado: {classifier}")


def to_probabilities(model: BaseEstimator, x: np.ndarray, classes: list[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        return probs[0]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        scores = np.atleast_2d(scores)

        if scores.shape[1] == 1 and len(classes) == 2:
            score = scores[0, 0]
            p1 = 1.0 / (1.0 + np.exp(-score))
            return np.array([1.0 - p1, p1], dtype=np.float64)

        row = scores[0]
        row = row - np.max(row)
        exp = np.exp(row)
        return exp / np.sum(exp)

    predictions = model.predict(x)
    hard = np.zeros(len(classes), dtype=np.float64)
    idx = classes.index(str(predictions[0]))
    hard[idx] = 1.0
    return hard
