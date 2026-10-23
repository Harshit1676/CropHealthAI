"""
Train an EfficientNetB0 transfer-learning classifier.

Training:
    Stage 1: frozen EfficientNetB0 backbone
    Stage 2: fine-tune last 40 non-BatchNorm layers

Uses class weighting and a memory-safe tf.data pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from ml import preprocess
from ml.model_utils import build_model

logger = config.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CropHealth AI EfficientNetB0 classifier"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS_DEFAULT,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE_DEFAULT,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--model-output",
        type=str,
        default=str(config.MODEL_PATH),
    )

    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.LEARNING_RATE,
    )

    return parser.parse_args()


def calculate_class_weights(summary: dict, class_names: list[str]) -> dict:
    """
    Compute balanced class weights:

        total_images / (number_of_classes * class_count)
    """
    total_images = sum(summary.get(name, 0) for name in class_names)
    num_classes = len(class_names)

    weights = {}

    for index, name in enumerate(class_names):
        count = summary.get(name, 0)

        if count > 0:
            weights[index] = total_images / (num_classes * count)
        else:
            weights[index] = 1.0

    return weights


def main() -> int:
    args = parse_args()

    dataset_root = (
        Path(args.data_dir)
        if args.data_dir
        else preprocess.find_dataset_root()
    )

    if dataset_root is None:
        print("ERROR: Dataset not found.")
        return 1

    flat_root = preprocess.flatten_dataset_if_nested(dataset_root)
    class_names = preprocess.get_class_names(dataset_root)

    if len(class_names) < 2:
        print(
            f"ERROR: Found only {len(class_names)} classes."
        )
        return 1

    summary = preprocess.dataset_summary(dataset_root)

    total_images = sum(summary.values())

    print(f"Dataset root : {dataset_root}")
    print(f"Classes found: {len(class_names)}")
    print(f"Total images : {total_images}")

    import tensorflow as tf

    tf.random.set_seed(config.SEED)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    train_ds = tf.keras.utils.image_dataset_from_directory(
        flat_root,
        validation_split=config.VALIDATION_SPLIT,
        subset="training",
        seed=config.SEED,
        image_size=config.IMAGE_SIZE,
        batch_size=args.batch_size,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        flat_root,
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",
        seed=config.SEED,
        image_size=config.IMAGE_SIZE,
        batch_size=args.batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names

    print("\nClasses:")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")

    # ---------------------------------------------------------
    # MEMORY-SAFE PIPELINE
    # ---------------------------------------------------------

    # IMPORTANT:
    # No cache()
    # Small shuffle buffer
    # Small prefetch
    train_ds = train_ds.shuffle(
        32,
        seed=config.SEED,
        reshuffle_each_iteration=True,
    ).prefetch(1)

    val_ds = val_ds.prefetch(1)

    # ---------------------------------------------------------
    # CLASS WEIGHTS
    # ---------------------------------------------------------

    class_weights = calculate_class_weights(
        summary,
        class_names,
    )

    print("\nClass weights:")
    for index, weight in class_weights.items():
        print(
            f"{index:2d} | "
            f"{class_names[index]:30s} | "
            f"{weight:.4f}"
        )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    model, base_model = build_model(
        num_classes=len(class_names),
        fine_tune=False,
    )

    model.summary()

    config.MODELS_DIR.mkdir(
        exist_ok=True,
        parents=True,
    )

    checkpoint_path = str(args.model_output)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ---------------------------------------------------------
    # STAGE 1 — FROZEN BACKBONE
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("STAGE 1: TRAINING CLASSIFICATION HEAD")
    print("=" * 70)

    start = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # ---------------------------------------------------------
    # STAGE 2 — FINE TUNING
    # ---------------------------------------------------------

    if args.fine_tune_epochs > 0:

        print("\n" + "=" * 70)
        print(
            f"STAGE 2: FINE-TUNING LAST 40 LAYERS "
            f"FOR {args.fine_tune_epochs} EPOCH(S)"
        )
        print("=" * 70)

        # Freeze everything first.
        base_model.trainable = False

        # Unfreeze only the last 40 layers.
        # BatchNormalization layers remain frozen for stability.
        for layer in base_model.layers[-40:]:
            if not isinstance(
                layer,
                tf.keras.layers.BatchNormalization
            ):
                layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-5
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        ft_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
            class_weight=class_weights,
        )

        for key in history.history:
            history.history[key].extend(
                ft_history.history.get(key, [])
            )

    elapsed = time.time() - start

    print(
        f"\nTraining completed in "
        f"{elapsed / 60:.1f} minutes"
    )

    # ---------------------------------------------------------
    # LOAD BEST CHECKPOINT
    # ---------------------------------------------------------

    print("\nLoading best validation checkpoint...")

    best_model = tf.keras.models.load_model(
        checkpoint_path
    )

    # Save class names.
    preprocess.save_class_names(class_names)

    # Save training history.
    with open(
        config.TRAINING_HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                k: [float(v) for v in vals]
                for k, vals in history.history.items()
            },
            f,
            indent=2,
        )

    print(f"Model saved to: {checkpoint_path}")
    print(
        f"Class names saved to: "
        f"{config.CLASS_NAMES_PATH}"
    )

    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------

    from ml.evaluate import evaluate_model

    evaluate_model(
        model=best_model,
        val_ds=val_ds,
        class_names=class_names,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
