"""
Plotting utilities for material recognition results.
"""

from .accuracy_plots import plot_confusion_matrix, plot_per_category_accuracy
from .analysis_plots import plot_accuracy_comparison, plot_error_analysis

__all__ = [
    "plot_confusion_matrix",
    "plot_per_category_accuracy",
    "plot_accuracy_comparison",
    "plot_error_analysis",
]
