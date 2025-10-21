"""
CS 7180, SpTp. Advanced Perception
Lucas Dunker, 10/18/25

Recognizing Materials Using Perceptually Inspired Features

Accuracy visualization for material recognition results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from typing import List
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.ticker import FuncFormatter


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    categories: List[str] = [],
    figsize: tuple = (10, 8),
    save_path: str = "confusion_matrix.png",
    normalize: bool = True,
):
    """
    Plot confusion matrix for material recognition results.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        categories: List of category names
        figsize: Figure size
        save_path: Path to save figure
        normalize: Whether to normalize the confusion matrix
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

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        cm_display = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    accuracy = np.sum(y_true == y_pred) / len(y_true)
    print(f"Overall Accuracy: {accuracy:.2%}")

    return fig, cm


def plot_per_category_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    categories: List[str] = [],
    figsize: tuple = (12, 6),
    save_path: str = "per_category_accuracy.png",
):
    """
    Plot per-category accuracy as a bar chart.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        categories: List of category names
        figsize: Figure size
        save_path: Path to save figure
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

    accuracies = []
    for i in range(len(categories)):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean()
            accuracies.append(acc)
        else:
            accuracies.append(0.0)

    # Sort by accuracy for better visualization
    sorted_indices = np.argsort(accuracies)[::-1]
    sorted_categories = [categories[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=figsize)

    viridis = plt.cm.get_cmap("viridis")
    bars = ax.bar(
        sorted_categories,
        sorted_accuracies,
        color=viridis(np.linspace(0.3, 0.9, len(categories))),
    )

    for i, (bar, acc) in enumerate(zip(bars, sorted_accuracies)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{acc:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Add overall accuracy line
    overall_acc = (y_true == y_pred).mean()
    ax.axhline(
        y=overall_acc,
        color="red",
        linestyle="--",
        label=f"Overall Accuracy: {overall_acc:.1%}",
        linewidth=2,
    )

    ax.set_xlabel("Material Category", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Category Recognition Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, accuracies


def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    categories: List[str] = [],
    figsize: tuple = (10, 6),
    save_path: str = "classification_report.png",
):
    """
    Plot classification report as a heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        categories: List of category names
        figsize: Figure size
        save_path: Path to save figure
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

    report = classification_report(
        y_true, y_pred, target_names=categories, output_dict=True
    )

    df_report = pd.DataFrame(report).transpose()
    metrics_df = df_report[["precision", "recall", "f1-score"]].iloc[:-3]
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        metrics_df.T,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Score"},
        ax=ax,
    )

    ax.set_xlabel("Material Category", fontsize=12)
    ax.set_ylabel("Metric", fontsize=12)
    ax.set_title("Classification Report", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, df_report
