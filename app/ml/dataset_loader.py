from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.core.settings import VALID_IMAGE_EXTENSIONS


class DatasetError(Exception):
    pass


@dataclass(frozen=True)
class DatasetRecord:
    image_path: Path
    label: str
    weight: float
    circumference: float


_IMAGE_KEYS = ("id", "image", "imagen", "file", "filename", "archivo", "img", "photo")
_WEIGHT_KEYS = ("peso", "weight")
_CIRC_KEYS = ("circunferencia", "circumference", "perimetro", "perimeter", "tamano", "tamaño")


def _normalize(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _normalize_image_key(value: str) -> str:
    stem = Path(value).stem.strip().lower()
    return "".join(ch for ch in stem if ch.isalnum())


def _normalize_label_from_dir(dirname: str) -> str:
    normalized = _normalize(dirname)
    if "limon" in normalized:
        return "limon"
    if "naranja" in normalized:
        return "naranja"
    return normalized


def _infer_column(columns: list[str], options: tuple[str, ...], human_name: str) -> str:
    normalized = {_normalize(col): col for col in columns}
    for option in options:
        if option in normalized:
            return normalized[option]

    for norm_name, raw_name in normalized.items():
        if any(option in norm_name for option in options):
            return raw_name

    raise DatasetError(
        f"No pude inferir la columna de '{human_name}' en el CSV. Columnas detectadas: {columns}"
    )


def _collect_images(dataset_dir: Path) -> list[tuple[Path, str]]:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise DatasetError(f"La carpeta '{dataset_dir}' no existe o no es un directorio.")

    pairs: list[tuple[Path, str]] = []
    class_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise DatasetError("No hay carpetas de clase dentro del dataset.")

    for class_dir in class_dirs:
        class_label = _normalize_label_from_dir(class_dir.name)
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                pairs.append((image_path, class_label))

    if not pairs:
        raise DatasetError("No se encontraron imágenes válidas en las carpetas de clase.")

    return pairs


def load_dataset_records(dataset_dir: str, csv_path: str) -> list[DatasetRecord]:
    image_pairs = _collect_images(Path(dataset_dir))

    csv_file = Path(csv_path)
    if not csv_file.exists() or not csv_file.is_file():
        raise DatasetError(f"El archivo CSV '{csv_path}' no existe.")

    df = pd.read_csv(csv_file)
    if df.empty:
        raise DatasetError("El CSV está vacío.")

    columns = list(df.columns)
    image_col = _infer_column(columns, _IMAGE_KEYS, "nombre de imagen")
    weight_col = _infer_column(columns, _WEIGHT_KEYS, "peso")
    circ_col = _infer_column(columns, _CIRC_KEYS, "circunferencia")

    df = df.copy()
    df[image_col] = df[image_col].astype(str).map(_normalize_image_key)

    metadata_by_name: dict[str, dict] = {}
    for row in df.to_dict(orient="records"):
        key = str(row.get(image_col, "")).strip()
        if not key:
            continue
        # Si hay IDs repetidos en el CSV, conservamos la primera ocurrencia.
        metadata_by_name.setdefault(key, row)

    records: list[DatasetRecord] = []
    missing_metadata: list[str] = []

    for image_path, label in image_pairs:
        key = _normalize_image_key(image_path.name)
        row = metadata_by_name.get(key)

        if row is None:
            missing_metadata.append(image_path.name)
            continue

        try:
            weight = float(row[weight_col])
            circumference = float(row[circ_col])
        except (TypeError, ValueError) as exc:
            raise DatasetError(
                f"Valores inválidos de peso/circunferencia para '{image_path.name}'"
            ) from exc

        records.append(
            DatasetRecord(
                image_path=image_path,
                label=label,
                weight=weight,
                circumference=circumference,
            )
        )

    if missing_metadata and not records:
        preview = ", ".join(missing_metadata[:10])
        raise DatasetError(
            "No se pudo empatar ninguna imagen contra el CSV. "
            f"Ejemplos faltantes: {preview}"
        )

    if len({record.label for record in records}) < 2:
        raise DatasetError("Se requieren al menos 2 clases para entrenar.")

    return records
