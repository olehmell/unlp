import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from typing import List
import torch

def calculate_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, technique_mapping: dict) -> dict:
    """Calculate Macro-F1 score for manipulation techniques classification."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )
    
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    class_metrics = {}
    for technique, idx in technique_mapping.items():
        class_metrics[technique] = {
            'precision': precision[idx],
            'recall': recall[idx],
            'f1': f1[idx],
            'support': support[idx]
        }
    
    return {
        'macro_metrics': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'class_metrics': class_metrics
    }

def evaluate_predictions(texts: List[str], true_labels: List[List[str]], classifier, threshold: float = 0.5, batch_size: int = 16) -> dict:
    """Evaluate model predictions against true labels using batch processing."""
    num_samples = len(texts)
    num_classes = len(classifier.technique_mapping)
    
    # Initialize arrays for true labels and predictions
    y_true = np.zeros((num_samples, num_classes))
    y_pred = np.zeros((num_samples, num_classes))
    
    # Process true labels
    for i, techniques in enumerate(true_labels):
        if techniques is not None:
            for technique in techniques:
                if technique in classifier.technique_mapping:
                    y_true[i, classifier.technique_mapping[technique]] = 1
    
    # Process predictions in batches
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch_texts = texts[i:batch_end]
        
        # Get predictions for the current batch
        batch_predictions = classifier.predict(batch_texts, threshold)
        
        # Convert predicted techniques to multi-hot encoding
        for j, techniques in enumerate(batch_predictions):
            for technique in techniques:
                if technique in classifier.technique_mapping:
                    y_pred[i + j, classifier.technique_mapping[technique]] = 1
    
    return calculate_macro_f1(y_true, y_pred, classifier.technique_mapping)

def print_evaluation_results(metrics: dict):
    """Print formatted evaluation results."""
    print("\nModel Performance Metrics:")
    print(f"Macro Precision: {metrics['macro_metrics']['precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_metrics']['recall']:.4f}")
    print(f"Macro F1: {metrics['macro_metrics']['f1']:.4f}")
    
    print("\nPer-technique Performance:")
    for technique, metrics in metrics['class_metrics'].items():
        print(f"\n{technique}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Support: {metrics['support']}")