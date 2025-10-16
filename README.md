# CS7180 Assignment 2 - Material Recognition System

A computer vision system for recognizing material categories from images, implementing the approach described in "Recognizing Materials Using Perceptually Inspired Features" (Sharan et al., IJCV 2013).

## Overview

This system can classify images into 10 material categories (fabric, foliage, glass, leather, metal, paper, plastic, stone, water, wood) using a combination of perceptually-inspired features and machine learning. It achieves approximately 57% accuracy on the Flickr Material Database (FMD), significantly outperforming previous texture recognition methods on this challenging dataset.

## Features

8 Feature Types: Color, texture (Jet, SIFT), micro-texture, curvature, and edge-based features
Bag-of-Words Model: Visual vocabulary construction using K-means clustering
SVM Classification: Using histogram intersection kernel for optimal performance
Bilateral Filtering: For separating base structure from micro-texture
Edge-based HOG: Novel features measuring reflectance properties along edges
Caching System: Efficient bilateral filter caching for faster processing
Modular Design: Clean separation between feature extraction, classification, and data loading

## Requirements

Python 3.7+
OpenCV 4.x
NumPy
scikit-learn
joblib

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

pip install numpy opencv-python scikit-learn joblib
```

3. Download the FMD dataset:
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

### Training a Model

Train a new model on the FMD dataset:
```bash

python main.py --fmd_path ./FMD --model_name my_model
```

## Evaluating a Model

Evaluate an existing model:
```bash

python main.py --fmd_path ./FMD --model_name my_model --evaluate_only
```

## Single Image Prediction

Predict the material category of a single image:
```bash

python main.py --predict path/to/image.jpg --model_name my_model
```

With mask:
```bash

python main.py --predict path/to/image.jpg --mask path/to/mask.png --model_name my_model
```

### Training Without Masks

To train without using the provided masks:
```bash

python main.py --fmd_path ./FMD --no_masks
```

## Python API
```python

from material_recognition import MaterialRecognitionSystem, FeaturePipeline
from material_recognition.datasets import load_fmd_dataset

# Initialize system
system = MaterialRecognitionSystem()
pipeline = FeaturePipeline()

# Load pre-trained model
system.load_model("material_recognition")

# Extract features from an image
import cv2
img = cv2.imread("path/to/image.jpg")
features = pipeline.extract_all_features(img)

# Predict material
label = system.predict(features)
print(f"Predicted material: {label}")
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
└── datasets/              # Data loading utilities
    ├── __init__.py
    └── fmd_loader.py      # FMD dataset loader
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
- Overall accuracy: ~57% (with masks)
- Overall accuracy: ~55% (without masks)
- Training time: ~5-10 minutes (depending on hardware)
- Prediction time: ~0.5-1 second per image

Per-category accuracy typically ranges from 40-80%, with natural materials (foliage, wood) performing better than manufactured materials (plastic, metal).

## Command-Line Arguments

- `--fmd_path`: Path to FMD dataset directory (default: `./FMD`)
- `--model_name`: Name for saved/loaded model (default: `material_recognition`)
- `--no_masks`: Don't use binary masks during training/evaluation
- `--predict`: Path to single image for prediction
- `--mask`: Path to mask for single image prediction
- `--evaluate_only`: Only evaluate existing model without training

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

### Adding New Features

- Implement extractor in features/extractors.py
- Add to pipeline in features/feature_pipeline.py
- Update dictionary sizes in config.py

### Using Different Classifiers

The modular design allows easy swapping of classifiers. Modify models/classifier.py to implement alternative classification methods.

## Citation

If you use this code in your research, please cite the original paper:
```bibtex

@article{sharan2013recognizing,
  title={Recognizing materials using perceptually inspired features},
  author={Sharan, Lavanya and Liu, Ce and Rosenholtz, Ruth and Adelson, Edward H},
  journal={International Journal of Computer Vision},
  volume={103},
  number={3},
  pages={348--371},
  year={2013},
  publisher={Springer}
}
```

## Troubleshooting

### Import Errors

Ensure you're running from the project root directory or have installed the package properly.

### Memory Issues

For large datasets, consider:

- Reducing batch sizes
- Using fewer clusters in dictionaries
- Processing images sequentially rather than in parallel

### Low Accuracy

Verify dataset is loaded correctly
Ensure masks align with images
Check that all feature extractors are working
Consider training with more data per category

### License

This implementation is provided for research and educational purposes. Please refer to the original paper for any commercial use considerations.

### Acknowledgments

This implementation is based on the research paper "Recognizing Materials Using Perceptually Inspired Features" by Sharan et al. The Flickr Material Database (FMD) was created by the same authors.


