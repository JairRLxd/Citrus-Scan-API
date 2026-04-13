from __future__ import annotations

import threading

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.schemas import (
    DatasetAnalysisRequest,
    DatasetAnalysisResponse,
    PredictResponse,
    PreprocessingStatusResponse,
    TrainRequest,
    TrainResponse,
)
from app.ml.analyzer import (
    analyze_records,
    build_shared_preprocessing_route,
    preprocessing_recommendations,
)
from app.ml.dataset_loader import DatasetError, load_dataset_records
from app.ml.model_registry import (
    ModelRegistryError,
    list_available_models,
    list_models_using_preprocessing,
    load_preprocessing_artifact,
    preprocessing_artifact_path,
)
from app.ml.preprocessor import PreprocessingConfig
from app.ml.service import TrainingError, predict, train
from app.ml.types import ClassifierName

router = APIRouter(prefix="/v1", tags=["citrus-api"])
_lock = threading.Lock()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/models")
def models() -> dict:
    return {"available_models": list_available_models()}


@router.get("/preprocessing/status", response_model=PreprocessingStatusResponse)
def preprocessing_status() -> PreprocessingStatusResponse:
    route_name = PreprocessingConfig().route_name
    artifact_path = preprocessing_artifact_path(route_name)
    exists = artifact_path.exists()
    models_using_route = list_models_using_preprocessing(route_name)

    if not exists:
        return PreprocessingStatusResponse(
            route_name=route_name,
            exists=False,
            artifact_path=str(artifact_path),
            fitted_at=None,
            config={
                "winsor_lower_q": PreprocessingConfig().winsor_lower_q,
                "winsor_upper_q": PreprocessingConfig().winsor_upper_q,
                "enable_pca": PreprocessingConfig().enable_pca,
                "pca_variance": PreprocessingConfig().pca_variance,
            },
            models_using_route=models_using_route,
        )

    artifact = load_preprocessing_artifact(route_name)
    return PreprocessingStatusResponse(
        route_name=route_name,
        exists=True,
        artifact_path=str(artifact_path),
        fitted_at=artifact.get("fitted_at"),
        config=artifact.get("config", {}),
        models_using_route=models_using_route,
    )


@router.post("/train", response_model=TrainResponse)
def train_model(payload: TrainRequest) -> TrainResponse:
    with _lock:
        try:
            metrics = train(
                classifier=payload.classifier,
                dataset_dir=payload.dataset_dir,
                csv_path=payload.csv_path,
                test_size=payload.test_size,
                random_state=payload.random_state,
            )
        except DatasetError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except TrainingError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error inesperado en train: {exc}") from exc

    return TrainResponse(message="Modelo entrenado correctamente", **metrics)


@router.post("/predict", response_model=PredictResponse)
async def predict_model(
    classifier: ClassifierName = Form(...),
    weight: float = Form(..., ge=0),
    circumference: float = Form(..., ge=0),
    file: UploadFile = File(...),
) -> PredictResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="La imagen está vacía.")

    with _lock:
        try:
            result = predict(
                classifier=classifier,
                image_bytes=image_bytes,
                weight=weight,
                circumference=circumference,
            )
        except ModelRegistryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error en predict: {exc}") from exc

    return PredictResponse(**result)


@router.post("/preprocessing/recommendation", response_model=DatasetAnalysisResponse)
def preprocessing_recommendation(payload: DatasetAnalysisRequest) -> DatasetAnalysisResponse:
    try:
        records = load_dataset_records(payload.dataset_dir, payload.csv_path)
        summary = analyze_records(records)
        shared_route = build_shared_preprocessing_route(records)
        recommendations = preprocessing_recommendations(records)
    except DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error analizando dataset: {exc}") from exc

    return DatasetAnalysisResponse(
        summary=summary,
        shared_route=shared_route,
        recommendations=recommendations,
    )
