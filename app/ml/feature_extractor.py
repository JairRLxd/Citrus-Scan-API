from __future__ import annotations

import io

import numpy as np
from PIL import Image


def read_image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def extract_image_features(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((128, 128))
    rgb = np.asarray(image, dtype=np.float32) / 255.0

    rgb_mean = rgb.mean(axis=(0, 1))
    rgb_std = rgb.std(axis=(0, 1))

    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    hsv_mean = hsv.mean(axis=(0, 1))
    hsv_std = hsv.std(axis=(0, 1))

    hue_channel = hsv[:, :, 0]
    hue_hist, _ = np.histogram(hue_channel, bins=8, range=(0.0, 1.0), density=True)

    return np.concatenate([rgb_mean, rgb_std, hsv_mean, hsv_std, hue_hist])


def build_feature_vector(image: Image.Image, weight: float, circumference: float) -> np.ndarray:
    image_features = extract_image_features(image)
    physical_features = np.array([float(weight), float(circumference)], dtype=np.float32)
    return np.concatenate([image_features, physical_features])
