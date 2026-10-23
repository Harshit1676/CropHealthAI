"""
Shared model-building and class-name parsing utilities.
EfficientNetB0 version for CropHealth AI.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = config.get_logger(__name__)


def build_model(
    num_classes: int,
    image_size: tuple[int, int] = config.IMAGE_SIZE,
    fine_tune: bool = False,
):
    """
    Build an EfficientNetB0 transfer-learning classifier.

    EfficientNetB0 includes its own input rescaling, so raw 0-255
    image tensors can be passed directly to the model.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB0

    inputs = tf.keras.Input(shape=(*image_size, 3))

    # Data augmentation.
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.15)(x)
    x = layers.RandomContrast(0.1)(x)

    base_model = EfficientNetB0(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = fine_tune

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(
        inputs,
        outputs,
        name="crophealth_efficientnetb0",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def parse_class_name(raw_class_name: str) -> dict:
    """
    Parse a raw class-folder name into a structured record.
    """
    name = raw_class_name.replace("___", "_").strip()
    parts = re.split(r"[_\s]+", name, maxsplit=1)

    crop = "Unknown"
    condition = name

    for known_crop in config.KNOWN_CROPS:
        if name.lower().startswith(known_crop.lower()):
            crop = known_crop
            condition = name[len(known_crop):].strip(" _-")
            break
    else:
        if len(parts) >= 2:
            crop = parts[0].capitalize()
            condition = parts[1].replace("_", " ").strip()

    condition_clean = condition.replace("_", " ").strip()

    if not condition_clean:
        condition_clean = "Unknown"

    is_healthy = any(
        k in condition_clean.lower()
        for k in config.HEALTHY_KEYWORDS
    )

    is_pest = (
        not is_healthy
        and any(k in condition_clean.lower()
                for k in config.PEST_KEYWORDS)
    )

    category = (
        "healthy"
        if is_healthy
        else ("pest" if is_pest else "disease")
    )

    display_condition = " ".join(
        w.capitalize() if not w.isupper() else w
        for w in condition_clean.split(" ")
    )

    return {
        "raw": raw_class_name,
        "crop": crop,
        "condition": display_condition,
        "category": category,
        "label": f"{crop} — {display_condition}",
    }


def build_class_metadata(class_names: list[str]) -> list[dict]:
    return [parse_class_name(c) for c in class_names]
