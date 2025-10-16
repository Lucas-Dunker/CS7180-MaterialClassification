"""
Advanced analysis plots for material recognition.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import FuncFormatter


def plot_error_analysis(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        categories: List[str] = [],
                        figsize: tuple = (12, 8),
                        save_path: str = ""):
    """
    Analyze and visualize common misclassifications.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        categories: List of category names
        figsize: Figure size
        save_path: Path to save figure
    """
    if categories is None:
        categories = [
            "fabric", "foliage", "glass", "leather", "metal",
            "paper", "plastic", "stone", "water", "wood"
        ]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Find top misclassifications
    misclassifications = []
    for i in range(len(categories)):
        for j in range(len(categories)):
            if i != j and cm[i, j] > 0:
                misclassifications.append({
                    'true': categories[i],
                    'pred': categories[j],
                    'count': cm[i, j],
                    'percent': cm[i, j] / cm[i].sum() * 100
                })
    
    # Sort by count
    misclassifications.sort(key=lambda x: x['count'], reverse=True)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Top misclassification pairs
    top_n = min(10, len(misclassifications))
    labels = [f"{m['true']} → {m['pred']}" for m in misclassifications[:top_n]]
    counts = [m['count'] for m in misclassifications[:top_n]]
    percentages = [m['percent'] for m in misclassifications[:top_n]]
    
    bars = ax1.barh(range(top_n), counts, color='coral')
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Number of Misclassifications')
    ax1.set_title('Top 10 Misclassification Pairs', fontweight='bold')
    ax1.invert_yaxis()
    
    # Add count labels
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        ax1.text(count + 0.5, bar.get_y() + bar.get_height()/2,
                f'{count} ({pct:.1f}%)', va='center')
    
    # Plot 2: Per-category error rate
    error_rates = []
    for i in range(len(categories)):
        total = cm[i].sum()
        if total > 0:
            error_rate = (total - cm[i, i]) / total
            error_rates.append(error_rate)
        else:
            error_rates.append(0)
    
    sorted_indices = np.argsort(error_rates)[::-1]
    sorted_categories = [categories[i] for i in sorted_indices]
    sorted_error_rates = [error_rates[i] for i in sorted_indices]
    
    bars2 = ax2.bar(range(len(categories)), sorted_error_rates,
                    color=plt.cm.get_cmap('Reds')(np.array(sorted_error_rates)))
    ax2.set_xlabel('Material Category')
    ax2.set_ylabel('Error Rate')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(sorted_categories, rotation=45, ha='right')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.grid(axis='y', alpha=0.3)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for bar, rate in zip(bars2, sorted_error_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Error Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, misclassifications


def plot_accuracy_comparison(results_dict: Dict[str, tuple],
                            figsize: tuple = (10, 6),
                            save_path: str = ""):
    """
    Compare accuracy across different models or configurations.
    
    Args:
        results_dict: Dictionary mapping model names to (y_true, y_pred) tuples
        figsize: Figure size
        save_path: Path to save figure
    """
    model_names = list(results_dict.keys())
    accuracies = []
    
    for name, (y_true, y_pred) in results_dict.items():
        acc = (y_true == y_pred).mean()
        accuracies.append(acc)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    bars = ax.bar(range(len(model_names)), accuracies,
                  color=plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(model_names))))
    
    # Add percentage labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Model/Configuration', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim(0, max(accuracies) * 1.1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(axis='y', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, accuracies