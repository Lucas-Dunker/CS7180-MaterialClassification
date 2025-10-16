"""
CS 7180, SpTp. Advanced Perception
Lucas Dunker, 10/18/25

Recognizing Materials Using Perceptually Inspired Features

Feature extraction methods for material recognition.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from config import FEATURE_CONFIG


@dataclass
class GaborFilterBank:
    """Gabor filter bank for texture analysis."""

    def __init__(
        self,
        scales: List[float] = FEATURE_CONFIG["gabor_filter"]["scales"],
        n_orientations: int = FEATURE_CONFIG["gabor_filter"]["n_orientations"],
    ):
        self.scales = scales or [0.6, 1.2, 2.0, 3.0]
        self.n_orientations = n_orientations
        self.filters = self._create_filterbank()

    def _create_filterbank(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create Gabor filter bank with multiple scales and orientations."""
        filters = []
        orientations = np.linspace(0, np.pi, self.n_orientations, endpoint=False)

        for scale in self.scales:
            for theta in orientations:
                kernel_real = cv2.getGaborKernel(
                    FEATURE_CONFIG["gabor_filter"]["kernel_size"],
                    sigma=scale * 2,
                    theta=float(theta),
                    lambd=scale * 4,
                    gamma=0.5,
                    psi=0,
                )
                kernel_imag = cv2.getGaborKernel(
                    FEATURE_CONFIG["gabor_filter"]["kernel_size"],
                    sigma=scale * 2,
                    theta=float(theta),
                    lambd=scale * 4,
                    gamma=0.5,
                    psi=np.pi / 2,
                )
                filters.append((kernel_real, kernel_imag))

        return filters

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply Gabor filters to image and return response stack."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        responses = []
        for kernel_real, kernel_imag in self.filters:
            resp_real = cv2.filter2D(gray, cv2.CV_32F, kernel_real)
            resp_imag = cv2.filter2D(gray, cv2.CV_32F, kernel_imag)
            responses.append(resp_real)
            responses.append(resp_imag)

        return np.stack(responses, axis=2)


class FeatureExtractor:
    """Base class for feature extraction."""

    def __init__(self):
        self.gabor_bank = None

    def extract_color_features(
        self, img: np.ndarray, mask: Optional[np.ndarray] = None, grid_step: int = 5
    ) -> np.ndarray:
        """Extract 3x3 RGB patches as color features."""
        h, w = img.shape[:2]
        y_coords = np.arange(1, h - 1, grid_step)
        x_coords = np.arange(1, w - 1, grid_step)

        features = []
        for y in y_coords:
            for x in x_coords:
                if mask is not None and mask[int(y), int(x)] == 0:
                    continue
                patch = img[int(y - 1) : int(y + 2), int(x - 1) : int(x + 2)]
                if patch.shape == (3, 3, 3):
                    features.append(patch.flatten())

        return (
            np.array(features, dtype=np.float32)
            if features
            else np.array([]).reshape(0, 27).astype(np.float32)
        )

    def extract_jet_features(
        self, img: np.ndarray, mask: Optional[np.ndarray] = None, grid_step: int = 5
    ) -> np.ndarray:
        """Extract Gabor jet features."""
        if self.gabor_bank is None:
            self.gabor_bank = GaborFilterBank()

        response_stack = self.gabor_bank.apply(img)
        h, w = img.shape[:2]

        y_coords = np.arange(12, h - 12, grid_step)
        x_coords = np.arange(12, w - 12, grid_step)

        features = []
        for y in y_coords:
            for x in x_coords:
                if mask is not None and mask[int(y), int(x)] == 0:
                    continue
                features.append(response_stack[int(y), int(x), :])

        return (
            np.array(features, dtype=np.float32)
            if features
            else np.array([]).reshape(0, 64).astype(np.float32)
        )

    def extract_sift_features(
        self,
        img: np.ndarray,
        mask: Optional[np.ndarray] = None,
        grid_step: int = FEATURE_CONFIG["grid_step"],
    ) -> np.ndarray:
        """Extract SIFT features on dense grid."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape

        sift = cv2.SIFT.create()
        keypoints = []

        for y in range(8, h - 8, grid_step):
            for x in range(8, w - 8, grid_step):
                if mask is not None and mask[y, x] == 0:
                    continue
                kp = cv2.KeyPoint(float(x), float(y), 16)
                keypoints.append(kp)

        if keypoints:
            _, descriptors = sift.compute(gray, keypoints)
            return (
                descriptors.astype(np.float32)
                if descriptors is not None
                else np.array([]).reshape(0, 128).astype(np.float32)
            )
        else:
            return np.array([]).reshape(0, 128).astype(np.float32)

    def extract_curvature_features(
        self,
        contours: List[np.ndarray],
        scales: Optional[List[int]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extract curvature features at multiple scales from contours."""
        if scales is None:
            scales = [2, 8, 16]

        features = []

        for contour in contours:
            if len(contour) < 10:
                continue

            for i in range(0, len(contour), 3):
                if mask is not None:
                    pt = contour[i][0]
                    if mask[pt[1], pt[0]] == 0:
                        continue

                curvatures = []
                for scale in scales:
                    start = max(0, i - scale)
                    end = min(len(contour), i + scale + 1)

                    if end - start >= 3:
                        segment = contour[start:end]
                        if len(segment) >= 3:
                            v1 = segment[-1][0].astype(np.float32) - segment[
                                len(segment) // 2
                            ][0].astype(np.float32)
                            v2 = segment[len(segment) // 2][0].astype(
                                np.float32
                            ) - segment[0][0].astype(np.float32)

                            angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0])
                            angle = np.arctan2(np.sin(angle), np.cos(angle))
                            curvatures.append(angle / scale)
                        else:
                            curvatures.append(0.0)
                    else:
                        curvatures.append(0.0)

                if len(curvatures) == len(scales):
                    features.append(curvatures)

        return (
            np.array(features, dtype=np.float32)
            if features
            else np.array([]).reshape(0, 3).astype(np.float32)
        )

    def extract_hog_slice(
        self,
        magnitude: np.ndarray,
        orientation: np.ndarray,
        cx: int,
        cy: int,
        angle: float,
        length: int = 18,
        width: int = 3,
    ) -> Optional[np.ndarray]:
        """Extract HOG descriptor from an oriented slice of the image."""
        h, w = magnitude.shape

        slice_mags = []
        slice_oris = []

        for i in range(length):
            for j in range(width):
                i_centered = i - length // 2
                j_centered = j - width // 2

                x = int(cx + i_centered * np.cos(angle) - j_centered * np.sin(angle))
                y = int(cy + i_centered * np.sin(angle) + j_centered * np.cos(angle))

                if 0 <= x < w and 0 <= y < h:
                    slice_mags.append(magnitude[y, x])
                    slice_oris.append(orientation[y, x])
                else:
                    slice_mags.append(0.0)
                    slice_oris.append(0.0)

        if len(slice_mags) != length * width:
            return None

        slice_mags = np.array(slice_mags).reshape(length, width)
        slice_oris = np.array(slice_oris).reshape(length, width)

        # Divide into 6 cells
        cell_size = length // 6
        hog_features = []

        for i in range(6):
            start_row = i * cell_size
            end_row = start_row + cell_size

            cell_mag = slice_mags[start_row:end_row, :].flatten()
            cell_ori = slice_oris[start_row:end_row, :].flatten()

            if len(cell_mag) > 0:
                hist, _ = np.histogram(
                    cell_ori, bins=12, range=(-np.pi, np.pi), weights=cell_mag
                )
            else:
                hist = np.zeros(12)

            hog_features.extend(hist)

        return np.array(hog_features, dtype=np.float32)
