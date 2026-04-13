from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PreprocessingConfig:
    route_name: str = "shared_citrus_v1"
    winsor_lower_q: float = 0.01
    winsor_upper_q: float = 0.99
    enable_pca: bool = False
    pca_variance: float = 0.98


class SharedPreprocessor:
    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.clip_lower: np.ndarray | None = None
        self.clip_upper: np.ndarray | None = None
        self.fitted_at: str | None = None

    @staticmethod
    def _normalize_units(x: np.ndarray) -> np.ndarray:
        x_norm = np.asarray(x, dtype=np.float64).copy()
        # Se asume que las dos últimas features son peso y circunferencia.
        weight_idx = x_norm.shape[1] - 2
        kg_mask = x_norm[:, weight_idx] < 5.0
        x_norm[kg_mask, weight_idx] = x_norm[kg_mask, weight_idx] * 1000.0
        return x_norm

    def fit(self, x: np.ndarray) -> "SharedPreprocessor":
        if x.ndim != 2:
            raise ValueError("fit espera una matriz 2D")

        x_norm = self._normalize_units(x)
        self.clip_lower = np.quantile(x_norm, self.config.winsor_lower_q, axis=0)
        self.clip_upper = np.quantile(x_norm, self.config.winsor_upper_q, axis=0)

        x_clip = np.clip(x_norm, self.clip_lower, self.clip_upper)
        self.scaler.fit(x_clip)

        if self.config.enable_pca:
            self.pca = PCA(n_components=self.config.pca_variance, svd_solver="full")
            self.pca.fit(self.scaler.transform(x_clip))

        self.fitted_at = datetime.now(UTC).isoformat()
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.clip_lower is None or self.clip_upper is None:
            raise ValueError("El preprocesador no está entrenado. Ejecuta fit primero.")
        if x.ndim != 2:
            raise ValueError("transform espera una matriz 2D")

        x_norm = self._normalize_units(x)
        x_clip = np.clip(x_norm, self.clip_lower, self.clip_upper)
        x_scaled = self.scaler.transform(x_clip)

        if self.pca is not None:
            return self.pca.transform(x_scaled)

        return x_scaled

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def to_artifact(self) -> dict:
        if self.clip_lower is None or self.clip_upper is None:
            raise ValueError("No se puede serializar un preprocesador sin entrenar.")

        return {
            "route_name": self.config.route_name,
            "config": {
                "winsor_lower_q": self.config.winsor_lower_q,
                "winsor_upper_q": self.config.winsor_upper_q,
                "enable_pca": self.config.enable_pca,
                "pca_variance": self.config.pca_variance,
            },
            "clip_lower": self.clip_lower,
            "clip_upper": self.clip_upper,
            "scaler": self.scaler,
            "pca": self.pca,
            "fitted_at": self.fitted_at,
        }

    @classmethod
    def from_artifact(cls, artifact: dict) -> "SharedPreprocessor":
        config = PreprocessingConfig(
            route_name=str(artifact.get("route_name", "shared_citrus_v1")),
            winsor_lower_q=float(artifact.get("config", {}).get("winsor_lower_q", 0.01)),
            winsor_upper_q=float(artifact.get("config", {}).get("winsor_upper_q", 0.99)),
            enable_pca=bool(artifact.get("config", {}).get("enable_pca", False)),
            pca_variance=float(artifact.get("config", {}).get("pca_variance", 0.98)),
        )
        instance = cls(config=config)
        instance.clip_lower = artifact["clip_lower"]
        instance.clip_upper = artifact["clip_upper"]
        instance.scaler = artifact["scaler"]
        instance.pca = artifact.get("pca")
        instance.fitted_at = artifact.get("fitted_at")
        return instance
