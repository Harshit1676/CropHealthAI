
from __future__ import annotations

import logging
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "crop_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "crophealth.db"

DATASET_DIR = BASE_DIR / "dataset"

IMAGE_SIZE = (224, 224)
SEED = 42

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

KNOWN_CROPS = [
    "Cashew",
    "Cassava",
    "Maize",
    "Tomato",
]

RISK_LEVELS = [
    (0, 29, "Low"),
    (30, 59, "Medium"),
    (60, 79, "High"),
    (80, 100, "Critical"),
]


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


DATA_DIR.mkdir(parents=True, exist_ok=True)

HEALTHY_KEYWORDS = ["healthy"]

    
PEST_KEYWORDS = [
    "fall armyworm",
    "grasshoper",
    "leaf beetle",
    "leaf miner",
    "green mite",
]
