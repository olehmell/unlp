import pandas as pd
from sklearn.model_selection import train_test_split
from llm_classifier import ManipulationClassifier
from evaluation import evaluate_predictions, print_evaluation_results

def train_model(model_name):
    # Load and prepare the dataset
    df = pd.read_parquet('data/bin/train.parquet')

    # Split the dataset into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['manipulative'])

    # Initialize the classifier with specified model
    classifier = ManipulationClassifier(model_name=model_name)

    print(f"Training {model_name}")
    print(f"Training data shape: {train_df.shape}")
    print(f"Validation data shape: {val_df.shape}")

    # Train the model
    classifier.train(
        train_df=train_df,
        val_df=val_df,
        output_dir=f'models/{model_name.split("/")[-1]}-manipulation',
        num_epochs=10,
        batch_size=24
    )

    # Example prediction and evaluation
    sample_texts = val_df['content'].head(5).tolist()
    predictions = classifier.predict(sample_texts)

    # Print results with more details
    print("\nPipeline Verification Results:")
    print(f"Number of test samples: {len(sample_texts)}")
    for i, (text, pred) in enumerate(zip(sample_texts, predictions), 1):
        print(f"\nSample {i}:")
        print(f"Text preview: {text[:100]}...")
        print(f"Prediction type: {type(pred)}")
        print(f"Predicted techniques: {pred}")

if __name__ == "__main__":
    train_model("answerdotai/ModernBERT-base")