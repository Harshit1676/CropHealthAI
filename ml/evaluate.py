"""
Evaluate a trained model and produce real metrics: accuracy, precision,
recall, F1-score, confusion matrix and a full classification report.

Usage:
    python -m ml.evaluate
    python -m ml.evaluate --data-dir dataset --model models/crop_disease_model.keras
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from ml import preprocess

logger = config.get_logger(__name__)


def evaluate_model(model=None, val_ds=None, class_names: list[str] | None = None,
                    data_dir: str | None = None, model_path: str | None = None) -> dict:
    """
    Evaluate `model` on `val_ds`. If they are not supplied (e.g. run as a
    standalone script), load the saved model and rebuild the validation
    split from the dataset directory.
    """
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report,
    )

    if model is None:
        model_path = model_path or str(config.MODEL_PATH)
        if not Path(model_path).exists():
            print(f"ERROR: Model not found at {model_path}. Run training first: "
                  f"python -m ml.train")
            return {}
        model = tf.keras.models.load_model(model_path)

    if val_ds is None:
        dataset_root = Path(data_dir) if data_dir else preprocess.find_dataset_root()
        if dataset_root is None:
            print("ERROR: Dataset not found; cannot rebuild validation split.")
            return {}
        flat_root = preprocess.flatten_dataset_if_nested(dataset_root)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            flat_root,
            validation_split=config.VALIDATION_SPLIT,
            subset="validation",
            seed=config.SEED,
            image_size=config.IMAGE_SIZE,
            batch_size=32,
            label_mode="categorical",
        )
        class_names = val_ds.class_names

    if class_names is None:
        class_names = preprocess.load_class_names()

    y_true, y_pred = [], []
    for batch_x, batch_y in val_ds:
        preds = model.predict(batch_x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_true.extend(np.argmax(batch_y.numpy(), axis=1).tolist())

    if not y_true:
        print("ERROR: Validation set is empty; cannot evaluate.")
        return {}

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )

    print("\n===== Evaluation Results =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    result = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "class_names": class_names,
        "classification_report": report,
    }
    with open(config.EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull evaluation report saved to {config.EVAL_REPORT_PATH}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the trained CropHealth AI model")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=str(config.MODEL_PATH))
    args = parser.parse_args()
    result = evaluate_model(data_dir=args.data_dir, model_path=args.model)
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
