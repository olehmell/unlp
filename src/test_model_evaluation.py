import pandas as pd
from llm_classifier import ManipulationClassifier
from sklearn.model_selection import train_test_split
from evaluation import evaluate_predictions, print_evaluation_results

def test_model_predictions():
    # Load validation data
    df = pd.read_parquet('data/bin/train.parquet')
    _, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['manipulative'])
    
    # Initialize the classifier
    classifier = ManipulationClassifier("models/ModernBERT-base-manipulation/checkpoint-192")
    print("\nModel Evaluation Test Started...")
    # Evaluate model performance
    metrics = evaluate_predictions(val_df['content'].tolist(), val_df['techniques'].tolist(), classifier)
    print("\nModel Evaluation Results:")
    print_evaluation_results(metrics)
    print("\nModel Evaluation Test Completed.")
    return metrics

if __name__ == "__main__":
    test_model_predictions()