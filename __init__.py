"""
Material Recognition System

A computer vision system for recognizing material categories from images,
based on "Recognizing Materials Using Perceptually Inspired Features"
(Sharan et al., 2013).
"""

from .models.classifier import MaterialRecognitionSystem
from .features.feature_pipeline import FeaturePipeline
from .datasets.fmd_loader import load_fmd_dataset

__all__ = ["MaterialRecognitionSystem", "FeaturePipeline", "load_fmd_dataset"]
