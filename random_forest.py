"""
Random Forest Classifier for Signature Classification

This module loads signature files (numpy arrays), trains a random forest classifier,
and evaluates its performance with multiple metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import exists, isdir, join
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


class SignatureClassifier:
    """Random Forest classifier for signature files."""
    
    def __init__(self, n_estimators=100, random_state=42, test_size=0.2):
        """
        Initialize the classifier.
        
        Args:
            n_estimators: Number of trees in the random forest
            random_state: Random seed for reproducibility
            test_size: Proportion of data to use for testing
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.test_size = test_size
        self.classifier = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.class_labels = None
        
    def load_signatures(self, data_dir, label_mapping=None):
        """
        Load signature files from a directory structure.
        
        Expects directory structure: data_dir/class_name/signature_files.npy
        
        Args:
            data_dir: Path to directory containing subdirectories for each class
            label_mapping: Optional dict mapping class names to numeric labels
            
        Returns:
            X: Feature array (n_samples, n_features)
            y: Label array (n_samples,)
            class_labels: Dict mapping numeric labels to class names
        """
        X = []
        y = []
        class_labels = {}
        label_counter = 0
        
        if not exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Get all subdirectories (classes)
        classes = [d for d in listdir(data_dir) 
                  if isdir(join(data_dir, d)) and not d.startswith('.')]
        
        if not classes:
            raise ValueError(f"No class subdirectories found in {data_dir}")
        
        print(f"Found {len(classes)} classes: {classes}")
        
        for class_name in sorted(classes):
            if label_mapping and class_name in label_mapping:
                class_label = label_mapping[class_name]
            else:
                class_label = label_counter
                label_counter += 1
            
            class_labels[class_label] = class_name
            class_dir = join(data_dir, class_name)
            
            # Load all .npy files in this class directory
            files = [f for f in listdir(class_dir) if f.endswith('.npy')]
            print(f"  Class '{class_name}' ({class_label}): Found {len(files)} files")
            
            for file in files:
                file_path = join(class_dir, file)
                try:
                    data = np.load(file_path)
                    
                    # Flatten if multidimensional
                    if data.ndim > 1:
                        data = data.flatten()
                    
                    X.append(data)
                    y.append(class_label)
                    
                except Exception as e:
                    print(f"    Warning: Could not load {file}: {e}")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nLoaded {len(X)} samples with {X.shape[1]} features")
        return X, y, class_labels
    
    def prepare_data(self, X, y, normalize=True):
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature array
            y: Label array
            normalize: Whether to normalize features using StandardScaler
        """
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Normalize if requested
        if normalize:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        else:
            self.X_train = self.X_train
            self.X_test = self.X_test
        
        print(f"\nData split:")
        print(f"  Training samples: {self.X_train.shape[0]}")
        print(f"  Testing samples: {self.X_test.shape[0]}")
        
    def train(self, feature_importance_top_n=10):
        """
        Train the random forest classifier.
        
        Args:
            feature_importance_top_n: Number of top features to display
        """
        if self.X_train is None:
            raise ValueError("Call prepare_data() first")
        
        print("\nTraining Random Forest Classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=1
        )
        
        self.classifier.fit(self.X_train, self.y_train)
        
        # Training accuracy
        train_accuracy = self.classifier.score(self.X_train, self.y_train)
        print(f"\nTraining accuracy: {train_accuracy:.4f}")
        
        # Show feature importance
        if feature_importance_top_n > 0:
            self._show_feature_importance(feature_importance_top_n)
    
    def _show_feature_importance(self, top_n):
        """Display top N most important features."""
        importances = self.classifier.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]
        
        print(f"\nTop {top_n} most important features:")
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. Feature {idx}: {importances[idx]:.6f}")
    
    def evaluate(self):
        """
        Evaluate the model on test set with multiple metrics.
        
        Returns:
            metrics: Dictionary containing all evaluation metrics
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() first")
        
        # Get predictions
        y_pred = self.classifier.predict(self.X_test)
        y_pred_proba = self.classifier.predict_proba(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Classification report
        class_report = classification_report(
            self.y_test, y_pred,
            target_names=[self.class_labels[i] for i in sorted(self.class_labels.keys())],
            zero_division=0
        )
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nDetailed Classification Report:")
        print(class_report)
        
        # ROC-AUC for binary classification
        n_classes = len(self.class_labels)
        if n_classes == 2:
            try:
                roc_auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                print(f"ROC-AUC Score: {roc_auc:.4f}")
            except Exception as e:
                print(f"Could not compute ROC-AUC: {e}")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': y_pred.tolist(),
            'true_labels': self.y_test.tolist()
        }
        
        return metrics
    
    def visualize_results(self, save_path=None):
        """
        Create visualizations of the results.
        
        Args:
            save_path: Optional path to save the figure
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() first")
        
        y_pred = self.classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confusion matrix
        im = axes[0].imshow(cm, cmap='Blues', aspect='auto')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        axes[0].set_title('Confusion Matrix')
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                           color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        plt.colorbar(im, ax=axes[0])
        
        # Feature importance (top 20)
        importances = self.classifier.feature_importances_
        top_indices = np.argsort(importances)[-20:][::-1]
        top_importances = importances[top_indices]
        
        axes[1].barh(range(len(top_importances)), top_importances)
        axes[1].set_xlabel('Importance')
        axes[1].set_ylabel('Feature Index')
        axes[1].set_title('Top 20 Feature Importances')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'scaler': self.scaler,
                'class_labels': self.class_labels
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a previously trained model from a file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.classifier = data['classifier']
            self.scaler = data['scaler']
            self.class_labels = data['class_labels']
        print(f"Model loaded from {filepath}")


def main():
    """
    Example usage of the SignatureClassifier.
    Modify the data_dir path to match your signature file structure.
    """
    
    # ===== CONFIGURATION =====
    # Directory structure should be:
    # data_dir/
    #   class_1/
    #     signature_1.npy
    #     signature_2.npy
    #   class_2/
    #     signature_1.npy
    #     signature_2.npy
    
    data_dir = "/Users/cindygrimm/VSCode/data/cherry/signatures/"  # Modify this path
    output_dir = "/Users/cindygrimm/VSCode/cherry_hyper/"
    
    # Check if data directory exists
    if not exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("\nPlease organize your signature files in this structure:")
        print("data_dir/")
        print("  class_name_1/")
        print("    signature_1.npy")
        print("    signature_2.npy")
        print("  class_name_2/")
        print("    signature_1.npy")
        return
    
    # ===== INITIALIZE CLASSIFIER =====
    classifier = SignatureClassifier(
        n_estimators=100,
        random_state=42,
        test_size=0.2
    )
    
    # ===== LOAD DATA =====
    print("Loading signatures...")
    X, y, class_labels = classifier.load_signatures(data_dir)
    classifier.class_labels = class_labels
    
    # ===== PREPARE DATA =====
    classifier.prepare_data(X, y, normalize=True)
    
    # ===== TRAIN MODEL =====
    classifier.train(feature_importance_top_n=10)
    
    # ===== EVALUATE MODEL =====
    metrics = classifier.evaluate()
    
    # ===== VISUALIZE RESULTS =====
    classifier.visualize_results(
        save_path=join(output_dir, "rf_results.png")
    )
    
    # ===== SAVE MODEL =====
    classifier.save_model(join(output_dir, "signature_classifier.pkl"))
    
    # ===== SAVE METRICS =====
    metrics_file = join(output_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")


if __name__ == '__main__':
    main()
