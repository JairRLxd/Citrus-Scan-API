from pydantic import BaseModel, Field

from app.ml.types import ClassifierName


class TrainRequest(BaseModel):
    classifier: ClassifierName
    dataset_dir: str = Field(..., description="Ruta local del dataset de imágenes")
    csv_path: str = Field(..., description="Ruta del CSV con peso y circunferencia")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = Field(default=42)


class TrainResponse(BaseModel):
    message: str
    classifier: str
    artifact_path: str
    classes: list[str]
    train_samples: int
    test_samples: int
    accuracy: float
    f1_weighted: float


class PredictItemResponse(BaseModel):
    classifier: str
    status: str
    predicted_label: str | None = None
    confidence: float | None = None
    confidence_percent: float | None = None
    class_probabilities: dict[str, float] | None = None
    detail: str | None = None


class PredictResponse(BaseModel):
    results: list[PredictItemResponse]
    best_classifier: str | None = None
    best_label: str | None = None


class DatasetAnalysisRequest(BaseModel):
    dataset_dir: str
    csv_path: str


class DatasetAnalysisResponse(BaseModel):
    summary: dict
    shared_route: dict
    recommendations: dict[str, list[str]]


class PreprocessingStatusResponse(BaseModel):
    route_name: str
    exists: bool
    artifact_path: str
    fitted_at: str | None
    config: dict
    models_using_route: list[str]
