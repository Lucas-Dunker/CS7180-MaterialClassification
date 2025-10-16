"""
Main script to load saved predictions and generate all plots.
"""

import numpy as np
import os
from accuracy_plots import (
    plot_confusion_matrix,
    plot_per_category_accuracy,
    plot_classification_report,
)
from analysis_plots import (
    plot_error_analysis,
)

def load_results(predictions_path: str, labels_path: str):
    """Load saved predictions and labels."""
    y_pred = np.load(predictions_path)
    y_true = np.load(labels_path)
    return y_true, y_pred

def generate_all_plots(y_true, y_pred, output_dir: str = "./plotting"):
    """Generate all visualization plots."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

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

    print("Generating plots...")

    # 1. Confusion Matrix
    print("1. Creating confusion matrix...")
    plot_confusion_matrix(
        y_true,
        y_pred,
        categories=categories,
        normalize=True,
        save_path=f"{output_dir}/confusion_matrix.png",
    )

    # 2. Per-category accuracy
    print("2. Creating per-category accuracy plot...")
    plot_per_category_accuracy(
        y_true,
        y_pred,
        categories=categories,
        save_path=f"{output_dir}/per_category_accuracy.png",
    )

    # 3. Classification report
    print("3. Creating classification report...")
    plot_classification_report(
        y_true,
        y_pred,
        categories=categories,
        save_path=f"{output_dir}/classification_report.png",
    )

    # 4. Error analysis
    print("4. Creating error analysis...")
    plot_error_analysis(
        y_true,
        y_pred,
        categories=categories,
        save_path=f"{output_dir}/error_analysis.png",
    )

    print(f"\nAll plots saved to {output_dir}/")

    # Print summary statistics
    accuracy = (y_true == y_pred).mean()
    print(f"\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Total Samples: {len(y_true)}")
    print(f"Correct Predictions: {np.sum(y_true == y_pred)}")
    print(f"Incorrect Predictions: {np.sum(y_true != y_pred)}")

    # Per-category breakdown
    print(f"\nPer-Category Accuracy:")
    for i, cat in enumerate(categories):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean()
            print(f"  {cat:10s}: {acc:.2%} ({mask.sum()} samples)")


if __name__ == "__main__":
    # Load your saved predictions and labels
    # Update these paths to match your saved files
    y_true, y_pred = load_results(
        predictions_path="./plotting/predictions.npy",
        labels_path="./plotting/true_labels.npy",
    )

    # Generate all plots
    generate_all_plots(y_true, y_pred)
