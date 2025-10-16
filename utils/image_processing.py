"""
Image processing utilities for material recognition.
"""

import cv2
import numpy as np
from typing import Tuple, Dict
from config import FEATURE_CONFIG, CACHE_SIZE

class ImageProcessor:
    """Handles image preprocessing operations."""
    
    def __init__(self):
        self.bilateral_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.cache_size = CACHE_SIZE
    
    def bilateral_filter(self, img: np.ndarray, 
                        d: int = FEATURE_CONFIG["bilateral_filter"]["d"], 
                        sigma_color: float = FEATURE_CONFIG["bilateral_filter"]["sigma_color"], 
                        sigma_space: float = FEATURE_CONFIG['bilateral_filter']["sigma_space"]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies bilateral filtering with caching.
        
        Returns:
            Tuple of (base image, residual image)
        """
        img_key = id(img)
        
        if img_key not in self.bilateral_cache:
            base = cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            residual = img.astype(np.float32) - base.astype(np.float32)
            self.bilateral_cache[img_key] = (base, residual)
            
            # Manage cache size
            if len(self.bilateral_cache) > self.cache_size:
                self.bilateral_cache.pop(next(iter(self.bilateral_cache)))
        
        return self.bilateral_cache[img_key]
    
    def detect_edges(self, img: np.ndarray, 
                    threshold_ratio: float = FEATURE_CONFIG["edge_detection"]["threshold_ratio"],
                    lower_ratio: float = FEATURE_CONFIG["edge_detection"]["lower_ratio"]) -> np.ndarray:
        """
        Detects edges using Canny edge detection.
        
        Args:
            img: Input image (can be color or grayscale)
            threshold_ratio: Ratio for upper threshold
            lower_ratio: Ratio of lower to upper threshold
        
        Returns:
            Binary edge map
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Calculate thresholds based on gradient magnitudes
        gradients = cv2.Sobel(gray, cv2.CV_32F, 1, 1)
        upper_threshold = np.percentile(np.abs(gradients), threshold_ratio * 100)
        lower_threshold = upper_threshold * lower_ratio
        
        edges = cv2.Canny(gray, float(lower_threshold), float(upper_threshold))
        return edges
    
    def extract_contours(self, edge_map: np.ndarray, 
                        min_length: int = 10) -> list:
        """
        Extract contours from an edge map.
        
        Args:
            edge_map: Binary edge map
            min_length: Minimum contour length to keep
        
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Filter out short contours
        filtered_contours = [c for c in contours if len(c) >= min_length]
        return filtered_contours
    
    @staticmethod
    def compute_gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute image gradients.
        
        Returns:
            Tuple of (gradient_x, gradient_y, magnitude, orientation)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)
        
        return gx, gy, magnitude, orientation