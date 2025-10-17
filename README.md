# CS7180 Assignment 2 - Material Recognition System

A computer vision system for recognizing material categories from images, implementing the approach described in "Recognizing Materials Using Perceptually Inspired Features" (Sharan et al., IJCV 2013).

## Overview

This system can classify images into 10 material categories (fabric, foliage, glass, leather, metal, paper, plastic, stone, water, wood) using a combination of perceptually-inspired features and machine learning. It achieves approximately 54% accuracy on the Flickr Material Database (FMD) over the average of several iterations, significantly outperforming previous texture recognition methods at the time of this paper's publication.

This project was created over the span of 1-2 weeks on both Window 11 and MacOS devices. Due to the project being written in Python3, it is inherently machine-agnostic. To generate the plots and model weights used in the project report, please follow the usage instructions outlined below.

## Features

- 8 Feature Extraction Types: Color, texture (Jet, SIFT), micro-texture, curvature, and edge-based features
- Bag-of-Words Model: Visual vocabulary construction using K-means clustering
- SVM Classification: Using histogram intersection kernel for optimal performance
- Bilateral Filtering for separating base structure from micro-texture
- Parallel Processing: Multi-core support for faster feature extraction
- Comprehensive Visualization: Confusion matrices, per-category accuracy, and error analysis
- Caching System: Efficient bilateral filter caching for faster processing
- Modular Design: Clean separation between feature extraction, classification, and data loading

## Requirements

- Python 3.7+
- OpenCV 4.x
- NumPy
- scikit-learn
- joblib
- matplotlib
- seaborn
- pandas
- tqdm

## Installation

Clone this repository:
```bash
git clone https://github.com/yourusername/material-recognition.git
cd material-recognition
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install numpy opencv-python scikit-learn joblib matplotlib seaborn pandas tqdm
```

Download the FMD dataset:
- Download from: [FMD Dataset](http://people.csail.mit.edu/lavanya/fmd.html)
- Extract to a directory (e.g., `./datasets/FMD`)
- Expected structure:
```
FMD/
├── image/
│   ├── fabric/
│   ├── foliage/
│   ├── glass/
│   └── ...
└── mask/
    ├── fabric/
    ├── foliage/
    └── ...
```

## Usage

### Quickstart - Evaluation and Training
For quick evaluation and training on a new device (using all defaults):
```bash
python main.py
```

### Training a Model

Train a new model on the FMD dataset:
```bash
python main.py --fmd_path ./datasets/FMD --model_name my_model
```

### Evaluating a Model

Evaluate an existing model:
```bash
python main.py --fmd_path ./datasets/FMD --model_name my_model --evaluate_only
```

### Single Image Prediction

Predict the material category of a single image:
```bash
python main.py --predict path/to/image.jpg --model_name my_model
```

With image masks:
```bash
python main.py --predict path/to/image.jpg --mask path/to/mask.png --model_name my_model
```

### Training Without Masks

To train without using the provided masks:
```bash
python main.py --fmd_path ./FMD --no_masks
```

### Visualizing Results

After training and evaluation, visualize the results:
```python
from plotting.visualize_results import generate_all_plots

# Load saved predictions and labels
import numpy as np
y_true = np.load("true_labels.npy")
y_pred = np.load("predictions.npy")

# Generate all visualization plots
generate_all_plots(y_true, y_pred, output_dir="plots")
```

Or run the visualization script directly:
```bash
cd plotting
python visualize_results.py
```

## Python API
```python
from material_recognition import MaterialRecognitionSystem, FeaturePipeline
from material_recognition.datasets import load_fmd_dataset

# Initialize system with parallel processing
system = MaterialRecognitionSystem()
pipeline = FeaturePipeline(n_jobs=-1)  # Use all CPU cores

# Load pre-trained model
system.load_model("material_recognition")

# Extract features from an image
import cv2
img = cv2.imread("path/to/image.jpg")
features = pipeline.extract_all_features(img)

# Predict material
label = system.predict(features)
print(f"Predicted material: {label}")

# Visualize results
from plotting.accuracy_plots import plot_confusion_matrix
plot_confusion_matrix(y_true, y_pred, normalize=True)
```

## Project Structure
```
material_recognition/
├── __init__.py              # Package initialization
├── main.py                  # Main entry point and CLI
├── config.py               # Configuration and constants
├── features/               # Feature extraction modules
│   ├── __init__.py
│   ├── extractors.py       # Individual feature extractors
│   └── feature_pipeline.py # Complete feature pipeline
├── models/                 # Machine learning models
│   ├── __init__.py
│   └── classifier.py       # Material recognition system
├── utils/                  # Utility functions
│   ├── __init__.py
│   └── image_processing.py # Image preprocessing utilities
├── datasets/              # Data loading utilities
│   ├── __init__.py
│   └── fmd_loader.py      # FMD dataset loader
└── plotting/              # Visualization tools
    ├── __init__.py
    ├── accuracy_plots.py   # Accuracy visualizations
    ├── analysis_plots.py   # Error analysis plots
    └── visualize_results.py # Main plotting script
```

## Feature Details

The system extracts 8 types of features:

1. **Color** (3×3 RGB patches): Captures color distribution
2. **Jet** (Gabor filters): Multi-scale texture analysis
3. **SIFT**: Scale-invariant local gradients
4. **Micro-Jet**: Texture in high-frequency residual
5. **Micro-SIFT**: Fine-scale texture patterns
6. **Curvature**: Edge curvature at multiple scales
7. **Edge-Slice**: HOG perpendicular to edges
8. **Edge-Ribbon**: HOG along edges

## Performance

Expected performance on FMD dataset:
- Overall accuracy: ~54% (with masks)
- Overall accuracy: ~52% (without masks)
- Training + evaluation time: ~30-40 minutes with parallel processing (depending on CPU cores)
- Prediction time: ~0.5-1 second per image

Per-category accuracy typically ranges from 40-80%, with natural materials (foliage, wood) performing better than manufactured materials (plastic, metal).

## Visualization Outputs

The plotting module generates:
- **Confusion Matrix**: Normalized and raw confusion matrices showing classification patterns
- **Per-Category Accuracy**: Bar chart showing accuracy for each material category
- **Classification Report**: Heatmap of precision, recall, and F1-scores
- **Error Analysis**: Top misclassification pairs and error rates by category

All plots are saved as high-resolution PNGs in the `plots/` directory.

## Command-Line Arguments

- `--fmd_path`: Path to FMD dataset directory (default: `./FMD`)
- `--model_name`: Name for saved/loaded model (default: `material_recognition`)
- `--no_masks`: Don't use binary masks during training/evaluation
- `--predict`: Path to single image for prediction
- `--mask`: Path to mask for single image prediction
- `--evaluate_only`: Only evaluate existing model without training
- `--n_jobs`: Number of CPU cores to use for parallel processing (default: -1 for all cores)

## Model Storage

Trained models are saved in the `./models/` directory with the following structure:
```
models/
└── model_name/
    ├── dict_*.pkl          # Visual dictionaries for each feature
    ├── svm_classifier.pkl  # Trained SVM classifier
    ├── X_train.npy        # Training histograms
    ├── training_labels.npy # Training labels
    └── config.pkl         # Model configuration
```

## Customization

### Modifying Feature Parameters

Edit config.py to adjust:
- Number of clusters per feature type
- Grid sampling step
- Bilateral filter parameters
- Gabor filter scales and orientations
- Train/Test image category sizes

### Adding New Features

1. Implement extractor in `features/extractors.py`
2. Add to pipeline in `features/feature_pipeline.py`
3. Update dictionary sizes in `config.py`

### Using Different Classifiers

The modular design allows easy swapping of classifiers. Modify `models/classifier.py` to implement alternative classification methods.

## Time Travel Days

I will be using 4 of my time travel days for this assignment. Thank you!