import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from pathlib import Path

class MaterialRecognitionSystem:
    def __init__(self, n_clusters_per_feature=None):
        """
        Initialize the material recognition system
        
        Args:
            n_clusters_per_feature: Dict mapping feature names to number of clusters
                                   (from Table 2 in the paper)
        """
        if n_clusters_per_feature is None:
            self.n_clusters = {
                'color': 150,
                'jet': 200,
                'sift': 250,
                'micro_jet': 200,
                'micro_sift': 250,
                'curvature': 100,
                'edge_slice': 200,
                'edge_ribbon': 200
            }
        else:
            self.n_clusters = n_clusters_per_feature
            
        self.dictionaries = {}
        self.svm_classifier = None
        
    def extract_color_features(self, img, mask=None, grid_step=5):
        """
        Extract 3x3 RGB patches as color features
        """
        h, w = img.shape[:2]
        features = []
        
        # Sample on a coarse grid
        for y in range(1, h-1, grid_step):
            for x in range(1, w-1, grid_step):
                if mask is not None and mask[y, x] == 0:
                    continue
                    
                # Extract 3x3 patch
                patch = img[y-1:y+2, x-1:x+2].reshape(-1)
                features.append(patch)
                
        return np.array(features) if features else np.array([]).reshape(0, 27)
    
    def create_gabor_filterbank(self):
        """
        Create Gabor filter bank with 4 scales and 8 orientations
        """
        filters = []
        scales = [0.6, 1.2, 2.0, 3.0]
        orientations = np.linspace(0, np.pi, 8, endpoint=False)
        
        for scale in scales:
            for theta in orientations:
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    (25, 25), 
                    sigma=scale*2, 
                    theta=theta,
                    lambd=scale*4,
                    gamma=0.5
                )
                filters.append(kernel)
                
        return filters
    
    def extract_jet_features(self, img, mask=None, grid_step=5):
        """
        Extract jet features (Gabor filter responses)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        filters = self.create_gabor_filterbank()
        features = []
        h, w = gray.shape
        
        # Apply each filter
        responses = []
        for kernel in filters:
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            responses.append(filtered)
        
        # Sample responses on grid
        for y in range(12, h-12, grid_step):
            for x in range(12, w-12, grid_step):
                if mask is not None and mask[y, x] == 0:
                    continue
                    
                feature = []
                for response in responses:
                    # Get both real (cosine) and imaginary (sine) parts
                    feature.append(response[y, x])
                    
                features.append(feature)
                
        return np.array(features) if features else np.array([]).reshape(0, 64)
    
    def extract_sift_features(self, img, mask=None, grid_step=5):
        """
        Extract SIFT descriptors on a grid
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        sift = cv2.SIFT_create()
        h, w = gray.shape
        features = []
        
        # Create keypoints on a grid
        keypoints = []
        for y in range(8, h-8, grid_step):
            for x in range(8, w-8, grid_step):
                if mask is not None and mask[y, x] == 0:
                    continue
                kp = cv2.KeyPoint(x, y, 16)
                keypoints.append(kp)
        
        # Compute descriptors
        if keypoints:
            _, descriptors = sift.compute(gray, keypoints)
            if descriptors is not None:
                features = descriptors
                
        return np.array(features) if len(features) > 0 else np.array([]).reshape(0, 128)
    
    def bilateral_filter_decomposition(self, img):
        """
        Decompose image into base and residual using bilateral filtering
        """
        # Apply bilateral filter (base image)
        base = cv2.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=5)
        
        # Compute residual (micro-texture)
        residual = img.astype(np.float32) - base.astype(np.float32)
        
        return base, residual
    
    def extract_micro_features(self, img, mask=None):
        """
        Extract micro-jet and micro-SIFT from residual image
        """
        _, residual = self.bilateral_filter_decomposition(img)
        
        # Normalize residual to valid image range
        residual_normalized = np.clip(residual + 128, 0, 255).astype(np.uint8)
        
        # Extract jet and SIFT from residual
        micro_jet = self.extract_jet_features(residual_normalized, mask)
        micro_sift = self.extract_sift_features(residual_normalized, mask)
        
        return micro_jet, micro_sift
    
    def extract_curvature_features(self, img, mask=None):
        """
        Extract curvature features from edges
        """
        base, _ = self.bilateral_filter_decomposition(img)
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if len(base.shape) == 3 else base
        
        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        features = []
        scales = [2, 8, 16]
        
        for contour in contours:
            if len(contour) < 10:
                continue
                
            # Sample every third point
            for i in range(0, len(contour), 3):
                pt = contour[i][0]
                
                if mask is not None and mask[pt[1], pt[0]] == 0:
                    continue
                
                curvatures = []
                for scale in scales:
                    # Compute curvature at different scales
                    start = max(0, i - scale)
                    end = min(len(contour), i + scale + 1)
                    
                    if end - start < 3:
                        curvatures.append(0)
                        continue
                    
                    # Simple curvature estimation
                    segment = contour[start:end]
                    if len(segment) >= 3:
                        # Use angle change as curvature measure
                        v1 = segment[-1][0] - segment[len(segment)//2][0]
                        v2 = segment[len(segment)//2][0] - segment[0][0]
                        
                        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0])
                        curvatures.append(angle)
                    else:
                        curvatures.append(0)
                
                features.append(curvatures)
                
        return np.array(features) if features else np.array([]).reshape(0, 3)
    
    def extract_edge_hog_features(self, img, mask=None):
        """
        Extract edge-slice and edge-ribbon features
        """
        base, _ = self.bilateral_filter_decomposition(img)
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if len(base.shape) == 3 else base
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Get edge points
        edge_points = np.argwhere(edges > 0)
        
        edge_slice_features = []
        edge_ribbon_features = []
        
        # Sample every other edge point
        for i in range(0, len(edge_points), 2):
            y, x = edge_points[i]
            
            if mask is not None and mask[y, x] == 0:
                continue
            
            # Compute edge orientation using Sobel
            dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            
            angle = np.arctan2(dy[y, x], dx[y, x])
            
            # Extract slices perpendicular and along edge
            slice_perpendicular = self._extract_slice(gray, x, y, angle + np.pi/2, 18, 3)
            slice_along = self._extract_slice(gray, x, y, angle, 18, 3)
            
            # Compute HOG for each slice
            if slice_perpendicular is not None:
                hog_perp = self._compute_hog_descriptor(slice_perpendicular)
                edge_slice_features.append(hog_perp)
                
            if slice_along is not None:
                hog_along = self._compute_hog_descriptor(slice_along)
                edge_ribbon_features.append(hog_along)
                
        edge_slice = np.array(edge_slice_features) if edge_slice_features else np.array([]).reshape(0, 72)
        edge_ribbon = np.array(edge_ribbon_features) if edge_ribbon_features else np.array([]).reshape(0, 72)
        
        return edge_slice, edge_ribbon
    
    def _extract_slice(self, img, x, y, angle, length, width):
        """
        Extract a slice of pixels along a given angle
        """
        h, w = img.shape
        slice_pixels = []
        
        for i in range(-length//2, length//2):
            for j in range(-width//2, width//2 + 1):
                # Rotate coordinates
                px = int(x + i * np.cos(angle) - j * np.sin(angle))
                py = int(y + i * np.sin(angle) + j * np.cos(angle))
                
                if 0 <= px < w and 0 <= py < h:
                    slice_pixels.append(img[py, px])
                else:
                    slice_pixels.append(0)
                    
        return np.array(slice_pixels).reshape(length, width) if slice_pixels else None
    
    def _compute_hog_descriptor(self, patch):
        """
        Compute simplified HOG descriptor for a patch
        """
        if patch is None:
            return np.zeros(72)
            
        # Divide into 6 cells of 3x3
        cells = []
        cell_size = 3
        n_bins = 12
        
        for i in range(6):
            cell_start = i * cell_size
            cell_end = min(cell_start + cell_size, patch.shape[0])
            
            cell = patch[cell_start:cell_end, :]
            
            # Compute gradients
            if cell.size > 0:
                gx = np.gradient(cell, axis=1)
                gy = np.gradient(cell, axis=0)
                
                # Compute orientation and magnitude
                angles = np.arctan2(gy, gx)
                magnitudes = np.sqrt(gx**2 + gy**2)
                
                # Create histogram
                hist, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi), weights=magnitudes)
                cells.append(hist)
            else:
                cells.append(np.zeros(n_bins))
                
        return np.concatenate(cells)
    
    def extract_all_features(self, img, mask=None):
        """
        Extract all features from an image
        """
        features = {}
        
        # Basic features
        features['color'] = self.extract_color_features(img, mask)
        features['jet'] = self.extract_jet_features(img, mask)
        features['sift'] = self.extract_sift_features(img, mask)
        
        # Micro features
        micro_jet, micro_sift = self.extract_micro_features(img, mask)
        features['micro_jet'] = micro_jet
        features['micro_sift'] = micro_sift
        
        # Shape features
        features['curvature'] = self.extract_curvature_features(img, mask)
        
        # Edge features
        edge_slice, edge_ribbon = self.extract_edge_hog_features(img, mask)
        features['edge_slice'] = edge_slice
        features['edge_ribbon'] = edge_ribbon
        
        return features
    
    def build_dictionaries(self, training_images, training_masks=None):
        """
        Build visual word dictionaries using k-means clustering
        """
        print("Building dictionaries...")
        
        for feat_name in self.n_clusters.keys():
            print(f"Processing {feat_name}...")
            all_features = []
            
            for idx, img in enumerate(training_images):
                mask = training_masks[idx] if training_masks else None
                features = self.extract_all_features(img, mask)
                
                if feat_name in features and len(features[feat_name]) > 0:
                    all_features.append(features[feat_name])
            
            if all_features:
                all_features = np.vstack(all_features)
                
                # Perform k-means clustering
                kmeans = KMeans(n_clusters=self.n_clusters[feat_name], random_state=42, n_init=10)
                kmeans.fit(all_features)
                self.dictionaries[feat_name] = kmeans
                print(f"  Created dictionary with {self.n_clusters[feat_name]} words")
    
    def compute_bow_histogram(self, img, mask=None):
        """
        Compute bag-of-words histogram for an image
        """
        features = self.extract_all_features(img, mask)
        histograms = []
        
        for feat_name in self.n_clusters.keys():
            if feat_name not in self.dictionaries:
                continue
                
            if feat_name in features and len(features[feat_name]) > 0:
                # Predict cluster assignments
                assignments = self.dictionaries[feat_name].predict(features[feat_name])
                
                # Create histogram
                hist, _ = np.histogram(assignments, bins=self.n_clusters[feat_name], 
                                      range=(0, self.n_clusters[feat_name]))
                hist = hist.astype(np.float32)
                
                # Normalize
                if hist.sum() > 0:
                    hist = hist / hist.sum()
            else:
                hist = np.zeros(self.n_clusters[feat_name])
                
            histograms.append(hist)
            
        # Concatenate all histograms
        return np.concatenate(histograms)
    
    def train_svm(self, training_images, training_labels, training_masks=None):
        """
        Train SVM classifier with histogram intersection kernel
        """
        print("Training SVM classifier...")
        
        # Compute BoW histograms for all training images
        X_train = []
        for idx, img in enumerate(training_images):
            mask = training_masks[idx] if training_masks else None
            hist = self.compute_bow_histogram(img, mask)
            X_train.append(hist)
            
        X_train = np.array(X_train)
        
        # Train SVM with custom kernel
        self.svm_classifier = SVC(kernel=self.histogram_intersection_kernel, C=1.0)
        self.svm_classifier.fit(X_train, training_labels)
        
        print("SVM training completed")
        
    @staticmethod
    def histogram_intersection_kernel(X, Y):
        """
        Histogram intersection kernel for SVM
        """
        return np.array([[np.minimum(x, y).sum() for y in Y] for x in X])
    
    def predict(self, img, mask=None):
        """
        Predict material category for an image
        """
        if self.svm_classifier is None:
            raise ValueError("Classifier not trained yet")
            
        hist = self.compute_bow_histogram(img, mask)
        return self.svm_classifier.predict([hist])[0]
    
    def evaluate(self, test_images, test_labels, test_masks=None):
        """
        Evaluate classifier on test set
        """
        predictions = []
        for idx, img in enumerate(test_images):
            mask = test_masks[idx] if test_masks else None
            pred = self.predict(img, mask)
            predictions.append(pred)
            
        predictions = np.array(predictions)
        accuracy = (predictions == test_labels).mean()
        
        return accuracy, predictions


# Usage example with FMD dataset
def load_fmd_dataset(fmd_path, categories=None):
    """
    Load FMD dataset
    
    Args:
        fmd_path: Path to FMD dataset root
        categories: List of category names to load (default: all 10)
    """
    if categories is None:
        categories = ['fabric', 'foliage', 'glass', 'leather', 'metal', 
                     'paper', 'plastic', 'stone', 'water', 'wood']
    
    images = []
    labels = []
    masks = []
    
    for cat_idx, category in enumerate(categories):
        cat_path = Path(fmd_path) / 'image' / category
        mask_path = Path(fmd_path) / 'mask' / category
        
        for img_file in sorted(cat_path.glob('*.jpg')):
            # Load image
            img = cv2.imread(str(img_file))
            images.append(img)
            labels.append(cat_idx)
            
            # Load corresponding mask if exists
            mask_file = mask_path / img_file.name
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                masks.append(mask)
            else:
                # Create default mask (all ones)
                masks.append(np.ones(img.shape[:2], dtype=np.uint8) * 255)
    
    return images, np.array(labels), masks


# Example training script
if __name__ == "__main__":
    # Set path to your FMD dataset
    FMD_PATH = "/path/to/FMD/dataset"
    
    # Load dataset
    print("Loading FMD dataset...")
    images, labels, masks = load_fmd_dataset(FMD_PATH)
    
    # Split into train/test (50/50 as in paper)
    n_samples = len(images)
    n_train = n_samples // 2
    
    # Random split
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_images = [images[i] for i in train_idx]
    train_labels = labels[train_idx]
    train_masks = [masks[i] for i in train_idx]
    
    test_images = [images[i] for i in test_idx]
    test_labels = labels[test_idx]
    test_masks = [masks[i] for i in test_idx]
    
    # Initialize system
    system = MaterialRecognitionSystem()
    
    # Build dictionaries
    system.build_dictionaries(train_images[:100], train_masks[:100])  # Use subset for faster testing
    
    # Train classifier
    system.train_svm(train_images, train_labels, train_masks)
    
    # Evaluate
    accuracy, predictions = system.evaluate(test_images, test_labels, test_masks)
    print(f"Test accuracy: {accuracy:.2%}")
    
    # Per-category accuracy
    categories = ['fabric', 'foliage', 'glass', 'leather', 'metal', 
                  'paper', 'plastic', 'stone', 'water', 'wood']
    
    for cat_idx, category in enumerate(categories):
        cat_mask = test_labels == cat_idx
        if cat_mask.sum() > 0:
            cat_acc = (predictions[cat_mask] == test_labels[cat_mask]).mean()
            print(f"  {category}: {cat_acc:.2%}")