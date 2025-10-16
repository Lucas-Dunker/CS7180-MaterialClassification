import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
import multiprocessing as mp
import joblib
import pickle
from pathlib import Path
import time


class MaterialRecognitionSystem:
    def __init__(self, n_clusters_per_feature=None, n_jobs=-1, model_dir="./models"):
        """
        Material recognition system matching paper specifications
        """
        if n_clusters_per_feature is None:
            self.n_clusters = {
                "color": 150,
                "jet": 200,
                "sift": 250,
                "micro_jet": 200,
                "micro_sift": 250,
                "curvature": 100,
                "edge_slice": 200,
                "edge_ribbon": 200,
            }
        else:
            self.n_clusters = n_clusters_per_feature

        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.dictionaries = {}
        self.svm_classifier = None
        self.X_train_stored = None
        self.gabor_filters = self._create_gabor_filterbank()
        self.bilateral_cache = {}

        print(f"Initialized with {self.n_jobs} CPU cores")

    def _create_gabor_filterbank(self):
        """Create Gabor filter bank"""
        filters = []
        scales = [0.6, 1.2, 2.0, 3.0]
        orientations = np.linspace(0, np.pi, 8, endpoint=False)

        for scale in scales:
            for theta in orientations:
                kernel_real = cv2.getGaborKernel(
                    (25, 25),
                    sigma=scale * 2,
                    theta=float(theta),
                    lambd=scale * 4,
                    gamma=0.5,
                    psi=0,
                )
                kernel_imag = cv2.getGaborKernel(
                    (25, 25),
                    sigma=scale * 2,
                    theta=float(theta),
                    lambd=scale * 4,
                    gamma=0.5,
                    psi=np.pi / 2,
                )
                filters.append((kernel_real, kernel_imag))

        return filters

    def extract_color_features(self, img, mask=None, grid_step=5):
        """Extract 3x3 RGB patches"""
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

    def extract_jet_features(self, img, mask=None, grid_step=5):
        """Extract Gabor jet features"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape

        responses = []
        for kernel_real, kernel_imag in self.gabor_filters:
            resp_real = cv2.filter2D(gray, cv2.CV_32F, kernel_real)
            resp_imag = cv2.filter2D(gray, cv2.CV_32F, kernel_imag)
            responses.append(resp_real)
            responses.append(resp_imag)

        response_stack = np.stack(responses, axis=2)

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

    def extract_sift_features(self, img, mask=None, grid_step=5):
        """Extract SIFT features on dense grid"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape

        sift = cv2.SIFT_create()
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

    def bilateral_filter(self, img):
        """Bilateral filtering with caching"""
        img_key = id(img)

        if img_key not in self.bilateral_cache:
            base = cv2.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=5)
            residual = img.astype(np.float32) - base.astype(np.float32)
            self.bilateral_cache[img_key] = (base, residual)

            if len(self.bilateral_cache) > 100:
                self.bilateral_cache.pop(next(iter(self.bilateral_cache)))

        return self.bilateral_cache[img_key]

    def extract_curvature_features(self, img, mask=None):
        """Extract curvature at three scales"""
        base, _ = self.bilateral_filter(img)
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if len(base.shape) == 3 else base

        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        features = []
        scales = [2, 8, 16]

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

                if len(curvatures) == 3:
                    features.append(curvatures)

        return (
            np.array(features, dtype=np.float32)
            if features
            else np.array([]).reshape(0, 3).astype(np.float32)
        )

    def extract_edge_hog_features(self, img, mask=None):
        """Extract HOG features along edges"""
        base, _ = self.bilateral_filter(img)
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if len(base.shape) == 3 else base

        edges = cv2.Canny(gray, 50, 150)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)

        edge_points = np.column_stack(np.where(edges > 0))

        edge_slice_features = []
        edge_ribbon_features = []

        # Only process edge points that are far enough from image boundaries
        h, w = gray.shape
        for i in range(0, len(edge_points), 2):
            y, x = edge_points[i]

            # Skip points too close to boundaries
            if x < 10 or x >= w - 10 or y < 10 or y >= h - 10:
                continue

            if mask is not None and mask[y, x] == 0:
                continue

            angle = orientation[y, x]

            # Perpendicular slice
            perp_features = self._extract_hog_slice(
                magnitude, orientation, x, y, angle + np.pi / 2, 18, 3
            )
            if perp_features is not None and len(perp_features) == 72:
                edge_slice_features.append(perp_features)

            # Along edge slice
            along_features = self._extract_hog_slice(
                magnitude, orientation, x, y, angle, 18, 3
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

    def _extract_hog_slice(self, magnitude, orientation, cx, cy, angle, length, width):
        """Extract HOG descriptor from a slice"""
        h, w = magnitude.shape

        # Collect pixels in the slice
        slice_mags = []
        slice_oris = []

        # Create a grid of points for the slice
        for i in range(length):
            for j in range(width):
                # Center the slice
                i_centered = i - length // 2
                j_centered = j - width // 2

                # Rotate coordinates
                x = int(cx + i_centered * np.cos(angle) - j_centered * np.sin(angle))
                y = int(cy + i_centered * np.sin(angle) + j_centered * np.cos(angle))

                if 0 <= x < w and 0 <= y < h:
                    slice_mags.append(magnitude[y, x])
                    slice_oris.append(orientation[y, x])
                else:
                    # Pad with zeros if outside image
                    slice_mags.append(0.0)
                    slice_oris.append(0.0)

        # We should have exactly length*width pixels now
        if len(slice_mags) != length * width:
            return None

        # Reshape to length x width
        slice_mags = np.array(slice_mags).reshape(length, width)
        slice_oris = np.array(slice_oris).reshape(length, width)

        # Divide into 6 cells (6 cells along the length, each of size 3x3)
        cell_size = length // 6  # Should be 3
        hog_features = []

        for i in range(6):
            start_row = i * cell_size
            end_row = start_row + cell_size

            # Extract cell
            cell_mag = slice_mags[start_row:end_row, :].flatten()
            cell_ori = slice_oris[start_row:end_row, :].flatten()

            # Compute 12-bin histogram
            if len(cell_mag) > 0:
                hist, _ = np.histogram(
                    cell_ori, bins=12, range=(-np.pi, np.pi), weights=cell_mag
                )
            else:
                hist = np.zeros(12)

            hog_features.extend(hist)

        return np.array(hog_features, dtype=np.float32)

    def extract_all_features(self, img, mask=None):
        """Extract all features from an image"""
        features = {}

        base, residual = self.bilateral_filter(img)
        residual_normalized = np.clip(residual + 128, 0, 255).astype(np.uint8)

        features["color"] = self.extract_color_features(img, mask)
        features["jet"] = self.extract_jet_features(img, mask)
        features["sift"] = self.extract_sift_features(img, mask)

        features["micro_jet"] = self.extract_jet_features(residual_normalized, mask)
        features["micro_sift"] = self.extract_sift_features(residual_normalized, mask)

        features["curvature"] = self.extract_curvature_features(img, mask)
        edge_slice, edge_ribbon = self.extract_edge_hog_features(img, mask)
        features["edge_slice"] = edge_slice
        features["edge_ribbon"] = edge_ribbon

        return features

    def build_dictionaries(self, training_images, training_masks=None):
        """Build visual word dictionaries"""
        
        print(f"Building dictionaries using {self.n_jobs} cores...")

        if training_masks is None:
            training_masks = [None] * len(training_images)

        # Process sequentially to avoid pickle issues
        all_image_features = []
        for img, mask in zip(training_images, training_masks):
            features = self.extract_all_features(img, mask)
            all_image_features.append(features)

        for feat_name in self.n_clusters.keys():
            print(f"Building dictionary for {feat_name}...")
            all_features = []

            for features in all_image_features:
                if feat_name in features and len(features[feat_name]) > 0:
                    all_features.append(features[feat_name])

            if all_features:
                all_features = np.vstack(all_features).astype(np.float32)
                print(f"  {len(all_features)} features collected")

                kmeans = MiniBatchKMeans(
                    n_clusters=self.n_clusters[feat_name],
                    random_state=42,
                    batch_size=min(1000, len(all_features)),
                    n_init=10,
                    max_iter=300,
                )
                kmeans.fit(all_features)
                self.dictionaries[feat_name] = kmeans
                print(f"  Created dictionary with {self.n_clusters[feat_name]} words")

    def compute_bow_histogram(self, img, mask=None):
        """Compute bag-of-words histogram"""
        features = self.extract_all_features(img, mask)
        histograms = []

        for feat_name in self.n_clusters.keys():
            if feat_name not in self.dictionaries:
                continue

            if feat_name in features and len(features[feat_name]) > 0:
                assignments = self.dictionaries[feat_name].predict(features[feat_name])
                hist, _ = np.histogram(
                    assignments,
                    bins=self.n_clusters[feat_name],
                    range=(0, self.n_clusters[feat_name]),
                )
                hist = hist.astype(np.float32)
                if hist.sum() > 0:
                    hist = hist / hist.sum()
            else:
                hist = np.zeros(self.n_clusters[feat_name], dtype=np.float32)

            histograms.append(hist)

        return np.concatenate(histograms).astype(np.float32)

    @staticmethod
    def histogram_intersection_kernel(X, Y):
        """Histogram intersection kernel"""
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        return np.array(
            [[np.minimum(x, y).sum() for y in Y] for x in X], dtype=np.float32
        )

    def train_svm(self, training_images, training_labels, training_masks=None):
        """Train SVM with histogram intersection kernel"""
        print(f"Computing histograms for {len(training_images)} training images...")

        if training_masks is None:
            training_masks = [None] * len(training_images)

        X_train = []
        for img, mask in zip(training_images, training_masks):
            hist = self.compute_bow_histogram(img, mask)
            X_train.append(hist)

        X_train = np.array(X_train, dtype=np.float32)
        self.X_train_stored = X_train

        print("Training SVM with histogram intersection kernel...")
        K_train = self.histogram_intersection_kernel(X_train, X_train)

        self.svm_classifier = SVC(kernel="precomputed", C=1.0, random_state=42)
        self.svm_classifier.fit(K_train, training_labels)
        self.training_labels = training_labels
        print("SVM training completed")

    def predict(self, img, mask=None):
        """Predict material category"""
        if self.svm_classifier is None or self.X_train_stored is None:
            raise ValueError("Classifier not trained yet")

        hist = self.compute_bow_histogram(img, mask)
        K_test = self.histogram_intersection_kernel([hist], self.X_train_stored)
        return self.svm_classifier.predict(K_test)[0]

    def predict_batch(self, images, masks=None):
        """Predict multiple images"""
        if masks is None:
            masks = [None] * len(images)

        histograms = []
        for img, mask in zip(images, masks):
            hist = self.compute_bow_histogram(img, mask)
            histograms.append(hist)

        X_test = np.array(histograms, dtype=np.float32)
        K_test = self.histogram_intersection_kernel(X_test, self.X_train_stored)
        return self.svm_classifier.predict(K_test)

    def evaluate(self, test_images, test_labels, test_masks=None):
        """Evaluate accuracy"""
        predictions = self.predict_batch(test_images, test_masks)
        test_labels = np.array(test_labels)
        accuracy = (predictions == test_labels).mean()
        return accuracy, predictions

    def save_model(self, name="material_recognition"):
        """Save model to disk"""
        save_path = self.model_dir / name
        save_path.mkdir(exist_ok=True)

        for feat_name, dictionary in self.dictionaries.items():
            joblib.dump(dictionary, save_path / f"dict_{feat_name}.pkl")

        if self.svm_classifier is not None:
            joblib.dump(self.svm_classifier, save_path / "svm_classifier.pkl")
            np.save(save_path / "X_train.npy", self.X_train_stored)
            np.save(save_path / "training_labels.npy", self.training_labels)

        config = {"n_clusters": self.n_clusters, "n_jobs": self.n_jobs}
        with open(save_path / "config.pkl", "wb") as f:
            pickle.dump(config, f)

        print(f"Model saved to {save_path}")

    def load_model(self, name="material_recognition"):
        """Load model from disk"""
        load_path = self.model_dir / name

        if not load_path.exists():
            raise FileNotFoundError(f"Model not found at {load_path}")

        with open(load_path / "config.pkl", "rb") as f:
            config = pickle.load(f)
            self.n_clusters = config["n_clusters"]

        self.dictionaries = {}
        for feat_name in self.n_clusters.keys():
            dict_path = load_path / f"dict_{feat_name}.pkl"
            if dict_path.exists():
                self.dictionaries[feat_name] = joblib.load(dict_path)

        svm_path = load_path / "svm_classifier.pkl"
        if svm_path.exists():
            self.svm_classifier = joblib.load(svm_path)
            self.X_train_stored = np.load(load_path / "X_train.npy")
            self.training_labels = np.load(load_path / "training_labels.npy")

        print(f"Model loaded from {load_path}")

    def model_exists(self, name="material_recognition"):
        """Check if model exists"""
        load_path = self.model_dir / name
        return load_path.exists() and (load_path / "svm_classifier.pkl").exists()


def load_fmd_dataset(fmd_path, categories=None):
    """Load FMD dataset"""
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

    images = []
    labels = []
    masks = []

    for cat_idx, category in enumerate(categories):
        cat_path = Path(fmd_path) / "image" / category
        mask_path = Path(fmd_path) / "mask" / category

        img_files = sorted(cat_path.glob("*.jpg"))

        for img_file in img_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            images.append(img)
            labels.append(cat_idx)

            mask_file = mask_path / img_file.name
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            else:
                mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
            masks.append(mask)

    return images, np.array(labels, dtype=np.int32), masks


def split_dataset_per_category(images, labels, masks, train_per_category=50):
    """Split dataset with exact number per category"""
    train_images = []
    train_labels = []
    train_masks = []
    test_images = []
    test_labels = []
    test_masks = []

    for cat in np.unique(labels):
        cat_idx = np.where(labels == cat)[0]
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(cat_idx)

        for i, idx in enumerate(cat_idx):
            if i < train_per_category:
                train_images.append(images[idx])
                train_labels.append(labels[idx])
                train_masks.append(masks[idx])
            else:
                test_images.append(images[idx])
                test_labels.append(labels[idx])
                test_masks.append(masks[idx])

    return (
        train_images,
        np.array(train_labels),
        train_masks,
        test_images,
        np.array(test_labels),
        test_masks,
    )


def main(fmd_path, model_name="material_recognition"):
    """Main training and evaluation pipeline"""
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

    system = MaterialRecognitionSystem()

    if system.model_exists(model_name):
        print(f"Loading existing model '{model_name}'...")
        system.load_model(model_name)

        images, labels, masks = load_fmd_dataset(fmd_path, categories)
        _, _, _, X_test, y_test, masks_test = split_dataset_per_category(
            images, labels, masks, train_per_category=50
        )
    else:
        print(f"Training new model...")

        print("Loading FMD dataset...")
        images, labels, masks = load_fmd_dataset(fmd_path, categories)
        print(f"Loaded {len(images)} images from {len(categories)} categories")

        X_train, y_train, masks_train, X_test, y_test, masks_test = (
            split_dataset_per_category(images, labels, masks, train_per_category=50)
        )

        print(f"Training set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")

        start_time = time.time()

        system.build_dictionaries(X_train, masks_train)
        system.train_svm(X_train, y_train, masks_train)

        train_time = time.time() - start_time
        print(f"Total training time: {train_time:.1f} seconds")

        system.save_model(model_name)

    print("\nEvaluating on test set...")
    accuracy, predictions = system.evaluate(X_test, y_test, masks_test)
    print(f"Overall accuracy: {accuracy:.2%}")

    print("\nPer-category accuracy:")
    for cat_idx, category in enumerate(categories):
        cat_mask = y_test == cat_idx
        if cat_mask.sum() > 0:
            cat_acc = (predictions[cat_mask] == y_test[cat_mask]).mean()
            print(f"  {category:10s}: {cat_acc:.2%}")

    return system, accuracy

if __name__ == "__main__":
    FMD_PATH = "./FMD"
    MODEL_NAME = "material_recognition"
    
    system, accuracy = main(FMD_PATH, MODEL_NAME)
    print(f"\nFinal test accuracy: {accuracy:.2%}")
