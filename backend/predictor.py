
"""
Model loading and inference for CropHealth AI.

Loads the trained EfficientNetB0 model and class_names.json lazily.
Provides real predictions, top-3 results, confidence, uncertainty
assessment, and warnings for known visually-confused disease classes.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from ml.model_utils import parse_class_name

logger = config.get_logger(__name__)


class ModelNotTrainedError(Exception):
    """Raised when a real prediction is requested but no trained model exists."""


class InvalidImageError(Exception):
    """Raised when the uploaded file cannot be read as an image."""


class Predictor:
    """Singleton-style wrapper around the trained Keras model."""

    _instance: Optional["Predictor"] = None

    # Confidence below this level is treated as uncertain.
    CONFIDENCE_THRESHOLD = 0.60

    # If the top two classes are too close, prediction is uncertain.
    MARGIN_THRESHOLD = 0.15

    # Pairs/classes that showed meaningful confusion during our
    # real-image sanity testing.
    KNOWN_CONFUSIONS = {
        frozenset({"Cassava bacterial blight", "Cassava brown spot"}),
        frozenset({"Maize leaf blight", "Maize streak virus"}),
        frozenset({"Maize leaf spot", "Maize leaf blight"}),
        frozenset({"Maize leaf spot", "Maize healthy"}),
        frozenset({"Tomato leaf blight", "Tomato verticulium wilt"}),
        frozenset({"Tomato leaf blight", "Tomato leaf curl"}),
        frozenset({"Tomato leaf curl", "Tomato verticulium wilt"}),
        frozenset({"Tomato septoria leaf spot", "Tomato leaf curl"}),
        frozenset({"Tomato septoria leaf spot", "Tomato verticulium wilt"}),
    }

    def __init__(self) -> None:
        self._model = None
        self._class_names: list[str] = []
        self._load_error: Optional[str] = None
        self._try_load()

    @classmethod
    def instance(cls) -> "Predictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _try_load(self) -> None:
        if not config.MODEL_PATH.exists() or not config.CLASS_NAMES_PATH.exists():
            self._load_error = (
                "Model not trained. Run the training script: "
                "python -m ml.train --epochs 10 --batch-size 32"
            )
            logger.info(self._load_error)
            return

        try:
            import tensorflow as tf
            import json

            self._model = tf.keras.models.load_model(
                str(config.MODEL_PATH)
            )

            with open(config.CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                self._class_names = json.load(f)

            logger.info(
                "Loaded model with %d classes from %s",
                len(self._class_names),
                config.MODEL_PATH,
            )

        except Exception as exc:
            self._load_error = f"Failed to load model: {exc}"
            logger.error(self._load_error)
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def reload(self) -> None:
        self._model = None
        self._class_names = []
        self._load_error = None
        self._try_load()

    def predict(self, image_bytes: bytes, top_k: int = 3) -> dict:
        """
        Run inference on raw image bytes.
        Returns a plain dictionary for FastAPI and internal scripts.
        """

        image_array = self._decode_image(image_bytes)

        if config.DEMO_MODE and not self.is_loaded:
            return self._demo_predict()

        if not self.is_loaded:
            return {
                "success": False,
                "demo_mode": False,
                "message": self._load_error or "Model not trained.",
            }

        import numpy as np

        preds = self._model.predict(
            np.expand_dims(image_array, axis=0),
            verbose=0
        )[0]

        return self._format_prediction(
            preds,
            demo_mode=False,
            top_k=top_k,
        )

    def _decode_image(self, image_bytes: bytes):
        from PIL import Image, UnidentifiedImageError
        import numpy as np

        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = img.convert("RGB")
            img = img.resize(config.IMAGE_SIZE)

        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError(
                f"Could not read the uploaded file as an image: {exc}"
            ) from exc

        # EfficientNetB0 training uses raw pixel values.
        # EfficientNet performs its own input preprocessing.
        return np.array(img, dtype=np.float32)

    def _format_prediction(
        self,
        preds,
        demo_mode: bool,
        top_k: int = 3,
    ) -> dict:

        import numpy as np

        order = np.argsort(preds)[::-1]
        top_indices = order[:min(top_k, len(order))]

        top_predictions = []

        for idx in top_indices:
            raw_name = self._class_names[idx]
            meta = parse_class_name(raw_name)

            top_predictions.append({
                "label": meta["label"],
                "crop": meta["crop"],
                "condition": meta["condition"],
                "category": meta["category"],
                "confidence": round(float(preds[idx]) * 100, 2),
                "raw_class": raw_name,
            })

        best = top_predictions[0]

        top1_confidence = float(preds[top_indices[0]])

        if len(top_indices) >= 2:
            top2_confidence = float(preds[top_indices[1]])
        else:
            top2_confidence = 0.0

        margin = top1_confidence - top2_confidence

        top1_raw = self._class_names[top_indices[0]]

        top2_raw = (
            self._class_names[top_indices[1]]
            if len(top_indices) >= 2
            else None
        )

        # Basic uncertainty checks.
        low_confidence = top1_confidence < self.CONFIDENCE_THRESHOLD
        small_margin = margin < self.MARGIN_THRESHOLD

        # Known visual confusion.
        confusion_warning = False

        if top2_raw is not None:
            pair = frozenset({top1_raw, top2_raw})

            if pair in self.KNOWN_CONFUSIONS:
                confusion_warning = True

        if low_confidence or small_margin:
            status = "uncertain"

            message = (
                "The image is difficult to classify reliably. "
                "Please upload a clearer image showing the affected "
                "leaf or plant."
            )

        elif confusion_warning:
            status = "caution"

            message = (
                "This prediction belongs to a visually similar disease "
                "group. Please verify the result with a clearer image "
                "or expert assessment."
            )

        else:
            status = "confident"
            message = "Prediction completed successfully."

        return {
            "success": True,

            "crop": best["crop"],
            "condition": best["condition"],
            "category": best["category"],
            "label": best["label"],

            "confidence": best["confidence"],

            "status": status,

            "top1_confidence": round(top1_confidence * 100, 2),
            "top2_confidence": round(top2_confidence * 100, 2),
            "margin": round(margin * 100, 2),

            "uncertain": status == "uncertain",
            "confusion_warning": confusion_warning,

            "top_predictions": top_predictions,

            "demo_mode": demo_mode,
            "message": message,
        }

    def _demo_predict(self) -> dict:
        """
        Explicit synthetic demo prediction.
        Never silently used for real inference.
        """

        import random

        demo_classes = [
            "Tomato___Leaf Curl",
            "Tomato___Healthy",
            "Maize___Fall Armyworm",
            "Cashew___Anthracnose",
            "Cassava___Mosaic",
        ]

        random.shuffle(demo_classes)

        raw_scores = sorted(
            [
                random.uniform(50, 96),
                random.uniform(1, 20),
                random.uniform(0.5, 10),
            ],
            reverse=True,
        )

        top_predictions = []

        for raw_name, score in zip(
            demo_classes[:3],
            raw_scores
        ):
            meta = parse_class_name(raw_name)

            top_predictions.append({
                "label": meta["label"],
                "crop": meta["crop"],
                "condition": meta["condition"],
                "category": meta["category"],
                "confidence": round(score, 2),
                "raw_class": raw_name,
            })

        best = top_predictions[0]

        return {
            "success": True,
            "crop": best["crop"],
            "condition": best["condition"],
            "category": best["category"],
            "label": best["label"],
            "confidence": best["confidence"],
            "status": "demo",
            "top1_confidence": best["confidence"],
            "top2_confidence": (
                top_predictions[1]["confidence"]
                if len(top_predictions) > 1
                else 0
            ),
            "margin": (
                best["confidence"] - top_predictions[1]["confidence"]
                if len(top_predictions) > 1
                else best["confidence"]
            ),
            "uncertain": False,
            "confusion_warning": False,
            "top_predictions": top_predictions,
            "demo_mode": True,
            "message": (
                "DEMO MODE: this is a synthetic demo prediction, "
                "not real inference. Train the model and disable "
                "DEMO_MODE for real results."
            ),
        }


def get_predictor() -> Predictor:
    return Predictor.instance()
