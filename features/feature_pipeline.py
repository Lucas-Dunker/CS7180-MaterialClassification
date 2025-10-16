"""
Complete feature extraction pipeline for material recognition.
"""

import numpy as np

from typing import Dict, Optional, List
from .extractors import FeatureExtractor
from utils.image_processing import ImageProcessor

class FeaturePipeline:
    """Complete pipeline for extracting all features from an image."""
    
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.processor = ImageProcessor()
    
    def extract_all_features(self, img: np.ndarray, 
                            mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Extract all features from an image.
        
        Args:
            img: Input image (BGR format)
            mask: Optional binary mask for region of interest
        
        Returns:
            Dictionary mapping feature names to feature arrays
        """
        features = {}
        
        base, residual = self.processor.bilateral_filter(img)
        residual_normalized = np.clip(residual + 128, 0, 255).astype(np.uint8)
        
        # Extract color and texture features from original image
        features["color"] = self.extractor.extract_color_features(img, mask)
        features["jet"] = self.extractor.extract_jet_features(img, mask)
        features["sift"] = self.extractor.extract_sift_features(img, mask)
        
        # Extract micro-texture features from residual
        features["micro_jet"] = self.extractor.extract_jet_features(
            residual_normalized, mask
        )
        features["micro_sift"] = self.extractor.extract_sift_features(
            residual_normalized, mask
        )
        
        # Extract shape features
        edges = self.processor.detect_edges(base)
        contours = self.processor.extract_contours(edges)
        features["curvature"] = self.extractor.extract_curvature_features(
            contours, mask=mask
        )
        
        # Extract edge-based HOG features
        _, _, magnitude, orientation = self.processor.compute_gradients(base)
        edge_slice, edge_ribbon = self._extract_edge_hog_features(
            edges, magnitude, orientation, mask
        )
        features["edge_slice"] = edge_slice
        features["edge_ribbon"] = edge_ribbon
        
        return features
    
    def _extract_edge_hog_features(self, edges: np.ndarray,
                                   magnitude: np.ndarray,
                                   orientation: np.ndarray,
                                   mask: Optional[np.ndarray] = None) -> tuple:
        """
        Extract HOG features along and perpendicular to edges.
        
        Returns:
            Tuple of (edge_slice_features, edge_ribbon_features)
        """
        edge_points = np.column_stack(np.where(edges > 0))
        
        edge_slice_features = []
        edge_ribbon_features = []
        
        h, w = edges.shape
        for i in range(0, len(edge_points), 2):
            y, x = edge_points[i]
            
            # Skip points too close to boundaries
            if x < 10 or x >= w - 10 or y < 10 or y >= h - 10:
                continue
            
            if mask is not None and mask[y, x] == 0:
                continue
            
            angle = float(orientation[y, x])
            
            # Perpendicular slice (edge-slice)
            perp_features = self.extractor.extract_hog_slice(
                magnitude, orientation, x, y, 
                angle + np.pi / 2, 18, 3
            )
            if perp_features is not None and len(perp_features) == 72:
                edge_slice_features.append(perp_features)
            
            # Along edge slice (edge-ribbon)
            along_features = self.extractor.extract_hog_slice(
                magnitude, orientation, x, y, 
                angle, 18, 3
            )
            if along_features is not None and len(along_features) == 72:
                edge_ribbon_features.append(along_features)
        
        edge_slice = (
            np.array(edge_slice_features, dtype=np.float32)
            if edge_slice_features
            else np.array([]).reshape(0, 72).astype(np.float32)
        )
        edge_ribbon = (
            np.array(edge_ribbon_features, dtype=np.float32)
            if edge_ribbon_features
            else np.array([]).reshape(0, 72).astype(np.float32)
        )
        
        return edge_slice, edge_ribbon
    
    def extract_batch(self, images: List[np.ndarray],
                     masks: Optional[List[np.ndarray]] = None) -> List[Dict[str, np.ndarray]]:
        """
        Extract features from multiple images.
        
        Args:
            images: List of images
            masks: Optional list of masks
        
        Returns:
            List of feature dictionaries
        """
        features_list = []
        for i, img in enumerate(images):
            mask = masks[i] if masks is not None else None
            features = self.extract_all_features(img, mask)
            features_list.append(features)
        
        return features_list

    def collect_features_by_type(self, 
                                images: List[np.ndarray],
                                masks: Optional[List[np.ndarray]] = None) -> Dict[str, List[np.ndarray]]:
        """
        Extract features and organize by feature type.
        
        This is useful for building dictionaries where we need all instances
        of each feature type across all images.
        
        Args:
            images: List of images
            masks: Optional list of masks
        
        Returns:
            Dictionary mapping feature names to lists of feature arrays
        """
        # Initialize dictionary with empty lists for each feature type
        features_by_type = {
            "color": [],
            "jet": [],
            "sift": [],
            "micro_jet": [],
            "micro_sift": [],
            "curvature": [],
            "edge_slice": [],
            "edge_ribbon": []
        }
        
        # Extract features from each image
        for i, img in enumerate(images):
            mask = masks[i] if masks is not None else None
            features = self.extract_all_features(img, mask)
            
            # Add each feature to the corresponding list
            for feat_name, feat_array in features.items():
                features_by_type[feat_name].append(feat_array)
        
        return features_by_type