"""
CS 7180, SpTp. Advanced Perception
Lucas Dunker, 10/18/25

Recognizing Materials Using Perceptually Inspired Features
"""

from .models.classifier import MaterialRecognitionSystem
from .features.feature_pipeline import FeaturePipeline
from .datasets.fmd_loader import load_fmd_dataset

__all__ = ["MaterialRecognitionSystem", "FeaturePipeline", "load_fmd_dataset"]
