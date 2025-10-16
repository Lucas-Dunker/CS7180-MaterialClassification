"""
Configuration settings and constants for the material recognition system.
"""

from pathlib import Path

FEATURE_CONFIG = {
    "n_clusters": {
        "color": 150,
        "jet": 200,
        "sift": 250,
        "micro_jet": 200,
        "micro_sift": 250,
        "curvature": 100,
        "edge_slice": 200,
        "edge_ribbon": 200,
    },
    "grid_step": 5,  # Sampling step for features
    "bilateral_filter": {"d": 9, "sigma_color": 50, "sigma_space": 5},
    "edge_detection": {"threshold_ratio": 0.9, "lower_ratio": 0.4},
    "gabor_filter": {
        "scales": [0.6, 1.2, 2.0, 3.0],
        "n_orientations": 8,
        "kernel_size": (25, 25),
    },
}

FMD_CATEGORIES = [
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

TRAIN_PER_CATEGORY = 95

MODEL_DIR = Path("./models")
PLOT_DIR = Path("./plotting")
CACHE_SIZE = 100  # For bilateral filter cache
