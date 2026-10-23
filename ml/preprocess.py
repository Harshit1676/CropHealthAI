"""
Dataset discovery and preprocessing utilities for CropHealth AI.

The Kaggle dataset "crop-pest-and-disease-detection" is distributed with a
nested folder structure that varies depending on how it was extracted. This
module searches a set of candidate locations, finds the actual directory
that contains class subfolders (e.g. "Tomato_Leaf Curl", "Cashew_healthy",
etc.), and builds a normalized class list plus a Keras-ready dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = config.get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _dir_has_images(path: Path) -> bool:
    try:
        for f in path.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                return True
    except (PermissionError, OSError):
        return False
    return False


def find_dataset_root(search_paths: Optional[list[Path]] = None) -> Optional[Path]:
    """
    Search known candidate locations for a directory whose immediate
    subdirectories are class folders containing images. Handles datasets
    extracted with an extra top-level wrapper folder by descending one
    level when necessary.
    """
    candidates = search_paths or config.CANDIDATE_DATASET_DIRS

    def class_dirs_of(root: Path) -> list[Path]:
        if not root.exists() or not root.is_dir():
            return []
        subdirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        return [d for d in subdirs if _dir_has_images(d)]

    for root in candidates:
        found = class_dirs_of(root)
        if found:
            logger.info("Dataset root found at %s (%d class folders)", root, len(found))
            return root

        # Try one level deeper (common with Kaggle zip wrapper folders, or
        # crop-level folders each containing their own disease subfolders,
        # e.g. dataset/Cashew/anthracnose, dataset/Cassava/healthy, ...).
        if root.exists() and root.is_dir():
            for sub in root.iterdir():
                if sub.is_dir():
                    nested = class_dirs_of(sub)
                    if nested:
                        logger.info("Dataset root found at %s (%d class folders)", sub, len(nested))
                        return sub
                    # two crop-per-folder layout: root/Crop/Class/*.jpg
                    grandchildren = [g for g in sub.iterdir() if g.is_dir()]
                    if grandchildren and any(_dir_has_images(g) for g in grandchildren):
                        logger.info("Dataset uses crop-nested layout, rooted at %s", root)
                        return root
    return None


def discover_classes(dataset_root: Path) -> list[str]:
    """
    Build a sorted list of class names from the dataset directory.

    Supports two layouts:
      1. Flat: dataset_root/<Crop_Class>/*.jpg
      2. Nested: dataset_root/<Crop>/<Class>/*.jpg
    Returned class names are normalized to "Crop___Class" form for layout 2,
    or used as-is (after cleanup) for layout 1.
    """
    classes = []
    for item in sorted(dataset_root.iterdir()):
        if not item.is_dir():
            continue
        if _dir_has_images(item):
            classes.append(item.name)
        else:
            # nested crop/class layout
            for sub in sorted(item.iterdir()):
                if sub.is_dir() and _dir_has_images(sub):
                    classes.append(f"{item.name}___{sub.name}")
    return classes


def flatten_dataset_if_nested(dataset_root: Path) -> Path:
    """
    Keras' image_dataset_from_directory expects `root/class_name/*.jpg`.
    If the dataset uses a nested crop/class layout, create a flattened
    view using symlinks under dataset/_flattened so training scripts can
    use the standard directory API without duplicating image data on disk.
    """
    classes = discover_classes(dataset_root)
    is_nested = any("___" in c for c in classes)
    if not is_nested:
        return dataset_root

    flat_root = dataset_root.parent / "_flattened_dataset"
    flat_root.mkdir(exist_ok=True)

    for crop_dir in sorted(dataset_root.iterdir()):
        if not crop_dir.is_dir():
            continue
        for class_dir in sorted(crop_dir.iterdir()):
            if not class_dir.is_dir() or not _dir_has_images(class_dir):
                continue
            flat_name = f"{crop_dir.name}___{class_dir.name}"
            link_path = flat_root / flat_name
            if not link_path.exists():
                try:
                    link_path.symlink_to(class_dir.resolve(), target_is_directory=True)
                except OSError:
                    # Symlinks may be unavailable (e.g. restricted Windows
                    # accounts) — fall back to a real copy.
                    import shutil
                    shutil.copytree(class_dir, link_path)
    logger.info("Flattened nested dataset into %s", flat_root)
    return flat_root


def get_class_names(dataset_root: Path) -> list[str]:
    flat_root = flatten_dataset_if_nested(dataset_root)
    return sorted([d.name for d in flat_root.iterdir() if d.is_dir() and _dir_has_images(d)])


def save_class_names(class_names: list[str], path: Path = config.CLASS_NAMES_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    logger.info("Saved %d class names to %s", len(class_names), path)


def load_class_names(path: Path = config.CLASS_NAMES_PATH) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"class_names.json not found at {path}. Train the model first: "
            f"python -m ml.train"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dataset_summary(dataset_root: Path) -> dict:
    """Return counts per class for quick sanity checks / dashboard use."""
    flat_root = flatten_dataset_if_nested(dataset_root)
    summary = {}
    for class_dir in sorted(flat_root.iterdir()):
        if class_dir.is_dir():
            count = sum(1 for f in class_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
            if count > 0:
                summary[class_dir.name] = count
    return summary


if __name__ == "__main__":
    root = find_dataset_root()
    if root is None:
        print("Dataset not found. Place the extracted Kaggle dataset into one of:")
        for c in config.CANDIDATE_DATASET_DIRS:
            print(f"  - {c}")
    else:
        print(f"Dataset root: {root}")
        summary = dataset_summary(root)
        print(f"Found {len(summary)} classes, {sum(summary.values())} images total")
        for k, v in summary.items():
            print(f"  {k}: {v}")
