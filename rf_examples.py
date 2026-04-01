"""
Example usage of RandomForest classifier for signatures.

This script demonstrates how to use the SignatureClassifier class
with sample data and different configurations.
"""

import numpy as np
from os import listdir, mkdir
from os.path import exists, join
from random_forest import SignatureClassifier


def create_sample_data(output_dir):
    """Create sample signature data for testing."""
    if exists(output_dir):
        print(f"Sample data directory already exists: {output_dir}")
        return
    
    mkdir(output_dir)
    
    # Create two classes of random "signatures"
    # In real usage, these would be your actual signature data
    n_samples_per_class = 20
    n_features = 64  # Flattened size of 8x8 signature
    
    for class_idx, class_name in enumerate(['genuine', 'forged']):
        class_dir = join(output_dir, class_name)
        mkdir(class_dir)
        
        # Generate random signatures for this class
        for i in range(n_samples_per_class):
            # Create signature with class-specific pattern
            signature = np.random.randn(n_features) * (class_idx + 1)
            signature_file = join(class_dir, f'signature_{i:03d}.npy')
            np.save(signature_file, signature)
            print(f"Created {signature_file}")
    
    print(f"\nSample data created in {output_dir}")


def example_1_basic_usage():
    """Example 1: Basic usage with default parameters."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Modify this path to point to your signature data
    data_dir = "./sample_signatures"  
    
    # Create sample data for demonstration
    create_sample_data(data_dir)
    
    # Initialize classifier
    classifier = SignatureClassifier()
    
    # Load and train
    X, y, class_labels = classifier.load_signatures(data_dir)
    classifier.class_labels = class_labels
    classifier.prepare_data(X, y)
    classifier.train()
    
    # Evaluate
    metrics = classifier.evaluate()
    
    # Visualize
    classifier.visualize_results()


def example_2_custom_parameters():
    """Example 2: Using custom hyperparameters."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Parameters")
    print("="*60)
    
    data_dir = "./sample_signatures"
    
    # Custom parameters
    classifier = SignatureClassifier(
        n_estimators=200,      # More trees
        random_state=123,
        test_size=0.3          # Larger test set
    )
    
    X, y, class_labels = classifier.load_signatures(data_dir)
    classifier.class_labels = class_labels
    classifier.prepare_data(X, y, normalize=True)
    classifier.train(feature_importance_top_n=15)
    
    metrics = classifier.evaluate()
    classifier.visualize_results(save_path="results_custom.png")


def example_3_predict_new_samples():
    """Example 3: Make predictions on new samples."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Predict New Samples")
    print("="*60)
    
    data_dir = "./sample_signatures"
    
    # Assume model is trained from example 1
    classifier = SignatureClassifier()
    X, y, class_labels = classifier.load_signatures(data_dir)
    classifier.class_labels = class_labels
    classifier.prepare_data(X, y)
    classifier.train()
    
    # Create a new sample signature
    new_signature = np.random.randn(X.shape[1])
    
    # Normalize it
    new_signature_normalized = classifier.scaler.transform([new_signature])
    
    # Predict
    prediction = classifier.classifier.predict(new_signature_normalized)[0]
    probabilities = classifier.classifier.predict_proba(new_signature_normalized)[0]
    
    print(f"\nNew sample prediction:")
    print(f"  Predicted class: {class_labels[prediction]}")
    print(f"  Class label: {prediction}")
    print(f"\n  Class probabilities:")
    for class_label in sorted(class_labels.keys()):
        class_name = class_labels[class_label]
        prob = probabilities[class_label]
        print(f"    {class_name}: {prob:.4f}")


def example_4_batch_predictions():
    """Example 4: Make batch predictions."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Predictions")
    print("="*60)
    
    data_dir = "./sample_signatures"
    
    classifier = SignatureClassifier()
    X, y, class_labels = classifier.load_signatures(data_dir)
    classifier.class_labels = class_labels
    classifier.prepare_data(X, y)
    classifier.train()
    
    # Create batch of new samples
    batch_size = 5
    n_features = X.shape[1]
    new_batch = np.random.randn(batch_size, n_features)
    
    # Normalize
    new_batch_normalized = classifier.scaler.transform(new_batch)
    
    # Predict
    predictions = classifier.classifier.predict(new_batch_normalized)
    probabilities = classifier.classifier.predict_proba(new_batch_normalized)
    
    print(f"\nBatch predictions for {batch_size} samples:")
    for i, pred in enumerate(predictions):
        print(f"\n  Sample {i+1}:")
        print(f"    Predicted class: {class_labels[pred]}")
        for class_label in sorted(class_labels.keys()):
            prob = probabilities[i, class_label]
            if prob > 0.1:  # Only show significant probabilities
                print(f"    {class_labels[class_label]}: {prob:.4f}")


if __name__ == '__main__':
    # Run examples
    print("Random Forest Signature Classifier - Examples\n")
    
    # Uncomment the example you want to run:
    
    example_1_basic_usage()
    # example_2_custom_parameters()
    # example_3_predict_new_samples()
    # example_4_batch_predictions()
