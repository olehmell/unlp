import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from typing import List
import torch
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

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

def evaluate_predictions(texts: List[str], true_labels: List[List[str]], classifier, threshold: float = 0.5) -> dict:
    """Evaluate model predictions against true labels."""
    logging.info(f"\nStarting evaluation of {len(texts)} samples with threshold {threshold}")
    
    # Initialize arrays for true labels
    num_samples = len(texts)
    num_classes = len(classifier.technique_mapping)
    y_true = np.zeros((num_samples, num_classes))
    y_pred = np.zeros((num_samples, num_classes))
    
    # Process true labels
    for i, techniques in enumerate(true_labels):
        logging.info(f"Processing true labels for sample {i+1}/{num_samples}")
        logging.info(f"Techniques: {techniques}")
        if techniques is not None:
            for technique in techniques:
                if technique in classifier.technique_mapping:
                    y_true[i, classifier.technique_mapping[technique]] = 1
    
    # Process predictions one by one
    for i in tqdm(range(num_samples), desc="Evaluating"):
        text = texts[i]
        true_techniques = true_labels[i] if true_labels[i] is not None else []
        
        # Get prediction and raw scores for single text
        prediction, raw_scores = classifier.predict_with_scores([text], threshold)
        prediction = prediction[0]  # Get first element since predict returns a list
        raw_scores = raw_scores[0]  # Get first element of scores
        
        # Log prediction details
        logging.info(f"\nSample {i+1}/{num_samples}:")
        logging.info(f"Text: {text[:100]}...")  # Log first 100 chars of text
        logging.info(f"True labels: {true_techniques}")
        logging.info("Predictions and scores:")
        
        # Log scores for each technique
        for technique, idx in classifier.technique_mapping.items():
            score = raw_scores[idx]
            predicted = technique in prediction
            logging.info(f"  {technique}: {score:.4f} {'(predicted)' if predicted else ''}")
        
        # Convert predicted techniques to multi-hot encoding
        for technique in prediction:
            if technique in classifier.technique_mapping:
                y_pred[i, classifier.technique_mapping[technique]] = 1
    
    return calculate_macro_f1(y_true, y_pred, classifier.technique_mapping)

def print_evaluation_results(metrics: dict):
    """Print formatted evaluation results."""
    logging.info("\nOverall Performance Metrics:")
    logging.info("=" * 50)
    logging.info(f"Macro Precision: {metrics['macro_metrics']['precision']:.4f}")
    logging.info(f"Macro Recall: {metrics['macro_metrics']['recall']:.4f}")
    logging.info(f"Macro F1: {metrics['macro_metrics']['f1']:.4f}")
    
    logging.info("\nPer-technique Performance:")
    logging.info("=" * 50)
    for technique, metrics in metrics['class_metrics'].items():
        logging.info(f"\n{technique.replace('_', ' ').title()}:")
        logging.info(f"  Precision: {metrics['precision']:.4f}")
        logging.info(f"  Recall: {metrics['recall']:.4f}")
        logging.info(f"  F1: {metrics['f1']:.4f}")
        logging.info(f"  Support: {metrics['support']}")