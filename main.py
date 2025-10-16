"""
CS 7180, SpTp. Advanced Perception
Lucas Dunker, 10/18/25

Recognizing Materials Using Perceptually Inspired Features

A computer vision system for recognizing material categories from images,
based on "Recognizing Materials Using Perceptually Inspired Features"
(Sharan et al., 2013).
"""

import time
import argparse
import numpy as np
import cv2

from typing import Optional
from models.classifier import MaterialRecognitionSystem
from features.feature_pipeline import FeaturePipeline
from datasets.fmd_loader import (
    load_fmd_dataset,
    split_dataset_per_category,
    print_dataset_info,
)
from config import FMD_CATEGORIES, TRAIN_PER_CATEGORY, PLOT_DIR


def train_material_recognizer(
    fmd_path: str, model_name: str = "material_recognition", use_masks: bool = True
):
    """
    Trains a material recognition system on a provided image dataset. Using the FMD dataset
    as per the reference paper.

    Args:
        fmd_path: Path to dataset
        model_name: Name for saved model
        use_masks: Whether to use binary masks
    """
    system = MaterialRecognitionSystem()
    pipeline = FeaturePipeline()

    if system.model_exists(model_name):
        print(f"Model '{model_name}' already exists. Loading...")
        system.load_model(model_name)
        return system

    print("Training new model...")

    print("Loading FMD dataset...")
    images, labels, masks = load_fmd_dataset(fmd_path, FMD_CATEGORIES)
    print(f"Loaded {len(images)} images from {len(FMD_CATEGORIES)} categories")

    if not use_masks:
        masks = [None] * len(images)

    X_train, y_train, masks_train, X_test, y_test, masks_test = (
        split_dataset_per_category(images, labels, masks, TRAIN_PER_CATEGORY)
    )

    print_dataset_info(X_train, y_train, "Training Set")

    print("\nExtracting features from training images...")
    start_time = time.time()

    features_by_type = pipeline.collect_features_by_type(X_train, masks_train)

    feature_time = time.time() - start_time
    print(f"Feature extraction completed in {feature_time:.1f} seconds")

    system.build_dictionaries(features_by_type)

    print("\nComputing bag-of-words histograms...")
    train_features = pipeline.extract_batch(X_train, masks_train)
    train_histograms = []
    for features in train_features:
        hist = system.compute_bow_histogram(features)
        train_histograms.append(hist)
    train_histograms = np.array(train_histograms)

    print("\nTraining SVM...")
    system.train_svm(train_histograms, y_train)

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f} seconds")

    system.save_model(model_name)

    return system


def evaluate_model(
    system: MaterialRecognitionSystem,
    fmd_path: str,
    use_masks: bool = True,
):
    """
    Evaluate the trained model on the test dataset.

    Args:
        system: Trained MaterialRecognitionSystem
        fmd_path: Path to FMD dataset
        use_masks: Whether to use binary masks
    """
    pipeline = FeaturePipeline()

    print("\nLoading test data...")
    images, labels, masks = load_fmd_dataset(fmd_path, FMD_CATEGORIES)

    if not use_masks:
        masks = [None] * len(images)

    _, _, _, X_test, y_test, masks_test = split_dataset_per_category(
        images, labels, masks, TRAIN_PER_CATEGORY
    )
    print_dataset_info(X_test, y_test, "Test Set")

    print("Extracting features from test images...")
    test_features = pipeline.extract_batch(X_test, masks_test)

    print("Evaluating on test set...")
    accuracy, predictions = system.evaluate(test_features, y_test)

    print(f"\nOverall accuracy: {accuracy:.2%}")

    print("\nPer-category accuracy:")
    for cat_idx, category in enumerate(FMD_CATEGORIES):
        cat_mask = y_test == cat_idx
        if cat_mask.sum() > 0:
            cat_acc = (predictions[cat_mask] == y_test[cat_mask]).mean()
            print(f"  {category:10s}: {cat_acc:.2%}")

    np.save(PLOT_DIR / "predictions.npy", predictions)
    np.save(PLOT_DIR / "true_labels.npy", y_test)
    print(
        "\nPredictions and true labels saved to 'predictions.npy' and 'true_labels.npy'"
    )

    return accuracy, predictions


def predict_single_image(
    system: MaterialRecognitionSystem, image_path: str, mask_path: Optional[str] = None
):
    """
    Loads an image and its corresponding mask, extracts image features, then
    predicts the image's material category.

    Args:
        system: Trained MaterialRecognitionSystem
        image_path: Path to input image
        mask_path: Optional path to mask image

    Returns:
        Predicted category name
    """
    pipeline = FeaturePipeline()

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image from {image_path}")

    mask = None
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    features = pipeline.extract_all_features(img, mask)

    label = system.predict(features)
    category = FMD_CATEGORIES[label]

    print(f"Predicted material: {category}")
    return category


def main():
    parser = argparse.ArgumentParser(description="Material Recognition System")
    parser.add_argument(
        "--fmd_path", type=str, default="./datasets/FMD", help="Path to FMD dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="material_recognition",
        help="Name for saved model",
    )
    parser.add_argument(
        "--no_masks", action="store_true", help="Don't use binary masks"
    )
    parser.add_argument(
        "--predict", type=str, help="Path to single image for prediction"
    )
    parser.add_argument(
        "--mask", type=str, help="Path to mask for single image prediction"
    )
    parser.add_argument(
        "--evaluate_only", action="store_true", help="Only evaluate existing model"
    )

    args = parser.parse_args()

    if args.predict:
        # Single image prediction
        system = MaterialRecognitionSystem()
        system.load_model(args.model_name)
        predict_single_image(system, args.predict, args.mask)
    else:
        # Training and evaluation
        if args.evaluate_only:
            system = MaterialRecognitionSystem()
            system.load_model(args.model_name)
        else:
            system = train_material_recognizer(
                args.fmd_path, args.model_name, not args.no_masks
            )

        # Evaluation
        accuracy, _ = evaluate_model(system, args.fmd_path, not args.no_masks)
        print(f"\nFinal test accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
