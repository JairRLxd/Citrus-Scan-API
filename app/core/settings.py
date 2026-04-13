from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "models"
PREPROCESSING_ARTIFACTS_DIR = BASE_DIR / "artifacts" / "preprocessing"
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
