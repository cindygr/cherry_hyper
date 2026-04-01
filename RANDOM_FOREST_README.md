# Random Forest Signature Classifier

A complete machine learning pipeline for classifying signatures using Random Forest classifier with training, testing, and comprehensive evaluation metrics.

## Features

- **Load signatures** from organized directory structure
- **Feature extraction** with automatic flattening of multi-dimensional data
- **Data normalization** using StandardScaler
- **Train/test split** with stratification for balanced classes
- **Random Forest training** with configurable hyperparameters
- **Comprehensive evaluation** including:
  - Accuracy, Precision, Recall, F1 Score
  - Confusion Matrix
  - Detailed Classification Report
  - Feature Importance Analysis
  - ROC-AUC Score (for binary classification)
- **Visualization** of results (confusion matrix, feature importance)
- **Model persistence** (save/load trained models)
- **Batch and single predictions** on new data

## Installation

All required packages are already in your virtual environment:
- scikit-learn
- numpy
- matplotlib

## Usage

### Data Organization

Organize your signature files as follows:

```
your_data_directory/
├── class_1_name/
│   ├── signature_001.npy
│   ├── signature_002.npy
│   └── ...
├── class_2_name/
│   ├── signature_001.npy
│   ├── signature_002.npy
│   └── ...
└── ...
```

**Note**: Each `.npy` file should contain a numpy array representing a signature. Multi-dimensional arrays will be automatically flattened.

### Quick Start

#### Option 1: Using the Classifier Class (Recommended)

```python
from random_forest import SignatureClassifier

# Initialize
classifier = SignatureClassifier(
    n_estimators=100,
    random_state=42,
    test_size=0.2
)

# Load data
data_dir = "/path/to/your/signatures"
X, y, class_labels = classifier.load_signatures(data_dir)
classifier.class_labels = class_labels

# Prepare data
classifier.prepare_data(X, y, normalize=True)

# Train
classifier.train(feature_importance_top_n=10)

# Evaluate
metrics = classifier.evaluate()

# Visualize
classifier.visualize_results(save_path="results.png")

# Save model
classifier.save_model("my_model.pkl")
```

#### Option 2: Run the Main Script

Edit `random_forest.py` to set your data directory path:

```python
data_dir = "/path/to/your/signatures"  # Change this line
```

Then run:

```bash
python random_forest.py
```

#### Option 3: Run Examples

The `rf_examples.py` file provides multiple example usage patterns:

```bash
python rf_examples.py
```

Examples included:
1. **Basic Usage** - Simple end-to-end pipeline
2. **Custom Parameters** - Using custom hyperparameters
3. **Single Prediction** - Predict on a new sample
4. **Batch Predictions** - Predict multiple samples

## Configuration

### Hyperparameters

You can customize the classifier by passing parameters to `SignatureClassifier()`:

```python
classifier = SignatureClassifier(
    n_estimators=200,      # Number of trees (default: 100)
    random_state=42,       # Seed for reproducibility
    test_size=0.2          # Train/test split ratio (default: 0.2)
)
```

### Random Forest Parameters

To modify the Random Forest itself (tree depth, min samples, etc.), edit the parameters in the `train()` method:

```python
self.classifier = RandomForestClassifier(
    n_estimators=self.n_estimators,
    max_depth=10,              # Add this
    min_samples_split=5,       # Add this
    min_samples_leaf=2,        # Add this
    random_state=self.random_state,
    n_jobs=-1
)
```

## Output Files

When running the main pipeline, the following files are created:

- **rf_results.png** - Visualization of confusion matrix and feature importance
- **signature_classifier.pkl** - Trained model that can be loaded later
- **metrics.json** - All evaluation metrics in JSON format

## Evaluation Metrics Explained

- **Accuracy** - Overall correctness: (TP + TN) / (TP + TN + FP + FN)
- **Precision** - Of predicted positives, how many were correct: TP / (TP + FP)
- **Recall** - Of actual positives, how many were found: TP / (TP + FN)
- **F1 Score** - Harmonic mean of precision and recall
- **Confusion Matrix** - Shows true positives, false positives, true negatives, false negatives for each class
- **Feature Importance** - Which input features are most useful for predictions

## Making Predictions on New Data

```python
# Load trained model
classifier = SignatureClassifier()
classifier.load_model("signature_classifier.pkl")

# Load and normalize new signature
new_signature = np.load("new_signature.npy")
new_signature_norm = classifier.scaler.transform([new_signature.flatten()])

# Predict
prediction = classifier.classifier.predict(new_signature_norm)[0]
probabilities = classifier.classifier.predict_proba(new_signature_norm)[0]

print(f"Predicted class: {classifier.class_labels[prediction]}")
print(f"Probabilities: {probabilities}")
```

## Troubleshooting

### No class files found
- Ensure your directory structure matches the expected format
- Files must be in `.npy` format (numpy binary)
- Remove any hidden files or non-data directories

### Memory issues with large files
- For very large signature files, you may need to increase normalization
- Consider using feature selection or PCA to reduce dimensionality

### Class imbalance issues
- The train/test split uses stratification to maintain class proportions
- For highly imbalanced data, consider using class weights in RandomForestClassifier:
  ```python
  class_weight='balanced'  # Add to RandomForestClassifier params
  ```

## References

- [scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Model Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
