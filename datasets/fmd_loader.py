"""
CS 7180, SpTp. Advanced Perception
Lucas Dunker, 10/18/25

Recognizing Materials Using Perceptually Inspired Features

FMD (Flickr Material Database) dataset loading utilities.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def load_fmd_dataset(
    fmd_path: str, categories: Optional[List[str]] = None
) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray]]:
    """
    Load the FMD dataset.

    Args:
        fmd_path: Path to FMD dataset root directory
        categories: List of category names to load (default: all 10 categories)

    Returns:
        Tuple of (images, labels, masks)
    """
    if categories is None:
        categories = [
            "fabric",
            "foliage",
            "glass",
            "leather",
            "metal",
            "paper",
            "plastic",
            "stone",
            "water",
            "wood",
        ]

    images = []
    labels = []
    masks = []

    for cat_idx, category in enumerate(categories):
        cat_path = Path(fmd_path) / "image" / category
        mask_path = Path(fmd_path) / "mask" / category

        img_files = sorted(cat_path.glob("*.jpg"))

        for img_file in img_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            images.append(img)
            labels.append(cat_idx)

            mask_file = mask_path / img_file.name
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            else:
                # Create full mask if no mask file exists
                mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
            masks.append(mask)

    return images, np.array(labels, dtype=np.int32), masks


def split_dataset_per_category(
    images: List[np.ndarray],
    labels: np.ndarray,
    masks: List[np.ndarray] | List[None],
    train_per_category: int = 50,
    random_seed: int = 43,
) -> Tuple:
    """
    Split dataset with an exact number of samples per category.

    Args:
        images: List of images
        labels: Array of category labels
        masks: List of masks
        train_per_category: Number of training samples per category
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_images, train_labels, train_masks,
                  test_images, test_labels, test_masks)
    """
    train_images = []
    train_labels = []
    train_masks = []
    test_images = []
    test_labels = []
    test_masks = []

    np.random.seed(random_seed)

    for cat in np.unique(labels):
        cat_idx = np.where(labels == cat)[0]
        np.random.shuffle(cat_idx)

        for i, idx in enumerate(cat_idx):
            if i < train_per_category:
                train_images.append(images[idx])
                train_labels.append(labels[idx])
                train_masks.append(masks[idx])
            else:
                test_images.append(images[idx])
                test_labels.append(labels[idx])
                test_masks.append(masks[idx])

    return (
        train_images,
        np.array(train_labels),
        train_masks,
        test_images,
        np.array(test_labels),
        test_masks,
    )


def get_category_names() -> List[str]:
    """Get list of FMD category names."""
    return [
        "fabric",
        "foliage",
        "glass",
        "leather",
        "metal",
        "paper",
        "plastic",
        "stone",
        "water",
        "wood",
    ]


def print_dataset_info(
    images: List[np.ndarray], labels: np.ndarray, set_name: str = "Dataset"
):
    """
    Print dataset statistics.

    Args:
        images: List of images
        labels: Array of category labels
        set_name: Name of the dataset (for display)
    """
    print(f"\n{set_name} Statistics:")
    print(f"  Total images: {len(images)}")
    print(f"  Categories: {len(np.unique(labels))}")

    categories = get_category_names()
    for cat_idx in range(len(categories)):
        count = np.sum(labels == cat_idx)
        print(f"    {categories[cat_idx]:10s}: {count} images")
