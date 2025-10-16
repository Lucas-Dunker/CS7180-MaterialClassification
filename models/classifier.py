"""
CS 7180, SpTp. Advanced Perception
Lucas Dunker, 10/18/25

Recognizing Materials Using Perceptually Inspired Features

Main material recognition system classifier.
"""

import numpy as np
import joblib
import pickle

from typing import Dict, List, Optional, Tuple
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from config import FEATURE_CONFIG, MODEL_DIR


class MaterialRecognitionSystem:
    """Material recognition system matching paper specifications."""

    def __init__(self, n_clusters_per_feature: Optional[Dict[str, int]] = None):
        """
        Initialize the material recognition system.

        Args:
            n_clusters_per_feature: Dictionary mapping feature names to cluster counts
        """
        if n_clusters_per_feature is None:
            self.n_clusters = FEATURE_CONFIG["n_clusters"]
        else:
            self.n_clusters = n_clusters_per_feature

        self.model_dir = MODEL_DIR
        self.model_dir.mkdir(exist_ok=True)

        self.dictionaries = {}
        self.svm_classifier = None
        self.X_train_stored = None
        self.training_labels = None

    def build_dictionaries(self, features_dict: Dict[str, List[np.ndarray]]):
        """
        Build visual word dictionaries for each feature type.

        Args:
            features_dict: Dictionary mapping feature names to lists of feature arrays
                          for all training images
        """
        print(f"Building dictionaries...")

        for feat_name in self.n_clusters.keys():
            if feat_name not in features_dict:
                continue

            print(f"Building dictionary for {feat_name}...")
            all_features = []

            for img_features in features_dict[feat_name]:
                if len(img_features) > 0:
                    all_features.append(img_features)

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

    def compute_bow_histogram(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute bag-of-words histogram from features.

        Args:
            features: Dictionary mapping feature names to feature arrays

        Returns:
            Concatenated histogram of visual words
        """
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
    def histogram_intersection_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute histogram intersection kernel."""
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        return np.array(
            [[np.minimum(x, y).sum() for y in Y] for x in X], dtype=np.float32
        )

    def train_svm(self, histograms: np.ndarray, labels: np.ndarray):
        """
        Train SVM classifier with histogram intersection kernel.

        Args:
            histograms: Array of bag-of-words histograms for training images
            labels: Array of category labels
        """
        self.X_train_stored = histograms
        self.training_labels = labels

        print("Training SVM with histogram intersection kernel...")
        K_train = self.histogram_intersection_kernel(histograms, histograms)

        self.svm_classifier = SVC(kernel="precomputed", C=1.0, random_state=42)
        self.svm_classifier.fit(K_train, labels)
        print("SVM training completed")

    def predict(self, features: Dict[str, np.ndarray]) -> int:
        """
        Predict material category for a single image.

        Args:
            features: Dictionary of extracted features

        Returns:
            Predicted category label
        """
        if self.svm_classifier is None or self.X_train_stored is None:
            raise ValueError("Classifier not trained yet")

        hist = self.compute_bow_histogram(features)
        K_test = self.histogram_intersection_kernel(
            np.array([hist]), self.X_train_stored
        )
        return self.svm_classifier.predict(K_test)[0]

    def predict_batch(self, features_list: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Predict material categories for multiple images.

        Args:
            features_list: List of feature dictionaries

        Returns:
            Array of predicted labels
        """
        if self.svm_classifier is None or self.X_train_stored is None:
            raise ValueError("Classifier not trained yet")

        histograms = []
        for features in features_list:
            hist = self.compute_bow_histogram(features)
            histograms.append(hist)

        X_test = np.array(histograms, dtype=np.float32)
        K_test = self.histogram_intersection_kernel(X_test, self.X_train_stored)
        return self.svm_classifier.predict(K_test)

    def evaluate(
        self, features_list: List[Dict[str, np.ndarray]], test_labels: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Evaluate accuracy on test set.

        Args:
            features_list: List of feature dictionaries for test images
            test_labels: True labels for test images

        Returns:
            Tuple of (accuracy, predictions)
        """
        predictions = self.predict_batch(features_list)
        test_labels = np.array(test_labels)
        accuracy = (predictions == test_labels).mean()
        return accuracy, predictions

    def save_model(self, name: str = "material_recognition"):
        """Save model to disk."""
        save_path = self.model_dir / name
        save_path.mkdir(exist_ok=True)

        # Save dictionaries
        for feat_name, dictionary in self.dictionaries.items():
            joblib.dump(dictionary, save_path / f"dict_{feat_name}.pkl")

        # Save SVM and training data
        if self.svm_classifier is not None:
            joblib.dump(self.svm_classifier, save_path / "svm_classifier.pkl")
            if self.X_train_stored is not None:
                np.save(save_path / "X_train.npy", self.X_train_stored)
            if self.training_labels is not None:
                np.save(save_path / "training_labels.npy", self.training_labels)

        # Save configuration
        config = {"n_clusters": self.n_clusters}
        with open(save_path / "config.pkl", "wb") as f:
            pickle.dump(config, f)

        print(f"Model saved to {save_path}")

    def load_model(self, name: str = "material_recognition"):
        """Load model from disk."""
        load_path = self.model_dir / name

        if not load_path.exists():
            raise FileNotFoundError(f"Model not found at {load_path}")

        # Load configuration
        with open(load_path / "config.pkl", "rb") as f:
            config = pickle.load(f)
            self.n_clusters = config["n_clusters"]

        # Load dictionaries
        self.dictionaries = {}
        for feat_name in self.n_clusters.keys():
            dict_path = load_path / f"dict_{feat_name}.pkl"
            if dict_path.exists():
                self.dictionaries[feat_name] = joblib.load(dict_path)

        # Load SVM
        svm_path = load_path / "svm_classifier.pkl"
        if svm_path.exists():
            self.svm_classifier = joblib.load(svm_path)
            self.X_train_stored = np.load(load_path / "X_train.npy")
            self.training_labels = np.load(load_path / "training_labels.npy")

        print(f"Model loaded from {load_path}")

    def model_exists(self, name: str = "material_recognition") -> bool:
        """Check if a model at the given path exists."""
        load_path = self.model_dir / name
        return load_path.exists() and (load_path / "svm_classifier.pkl").exists()
