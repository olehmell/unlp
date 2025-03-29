import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
import pymongo
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# MongoDB connection
def connect_to_mongodb():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI environment variable not set")
    
    client = pymongo.MongoClient(
        mongo_uri,
        tlsAllowInvalidCertificates=True  # This bypasses SSL certificate verification
    )
    db = client.manipulation_db
    collection = db.analyses
    
    return collection

def load_embeddings_from_mongodb():
    collection = connect_to_mongodb()
    existing_docs = collection.find({}, {'embedding': 1, '_id': 1, 'techniques': 1})

    # Initialize lists to store data
    embeddings = []
    labels = []
    
    for doc in existing_docs:
        print(f"Processing document {doc['_id']}")
        # Extract embedding
        embedding_data = doc.get("embedding", [])
        
        # Extract techniques (labels)
        techniques = doc.get("techniques", [])
        print(f"Techniques: {techniques}")
        
        # Skip documents with missing embeddings
        if not embedding_data:
            continue
            
        # For documents with no techniques, assign a special "none" class
        if not techniques:
            techniques = ["none"]
            
        embeddings.append(embedding_data)
        labels.append(techniques)
    
    return np.array(embeddings), labels

# Load data from MongoDB
print("Loading embeddings from MongoDB...")
X, y_raw = load_embeddings_from_mongodb()

# Convert multilabel format
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y_raw)
print(f"Labels: {mlb.classes_}")
print(f"Number of samples: {len(X)}")
print(f"Number of labels: {len(mlb.classes_)}")

# Calculate class weights to deal with imbalance
class_counts = np.sum(y, axis=0)
total_samples = len(y)
class_weights = {i: total_samples / (len(mlb.classes_) * count) for i, count in enumerate(class_counts)}

print("Class distribution:")
for i, label in enumerate(mlb.classes_):
    count = class_counts[i]
    print(f"{label}: {count} samples, weight: {class_weights[i]:.2f}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

# Create a custom objective function for multilabel with weights
def weighted_binary_cross_entropy(y_pred, dtrain):
    y_true = dtrain.get_label()
    # Reshape y_true to match shape of y_pred
    y_true = y_true.reshape(len(y_pred) // len(mlb.classes_), len(mlb.classes_)).ravel()
    
    # Clip predictions for numerical stability
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    
    # Get class indices for applying weights
    num_samples = len(y_pred) // len(mlb.classes_)
    class_indices = np.tile(np.arange(len(mlb.classes_)), num_samples)
    
    # Apply class weights
    weights = np.array([class_weights[i] for i in class_indices])
    
    # Compute gradient and hessian
    grad = weights * (y_pred - y_true)
    hess = weights * y_pred * (1 - y_pred)
    
    return grad, hess

# Prepare data for LightGBM
# For multilabel, we need to flatten the labels
y_train_flat = y_train.flatten()
y_test_flat = y_test.flatten()

# Repeat features for each label
X_train_rep = np.repeat(X_train, len(mlb.classes_), axis=0)
X_test_rep = np.repeat(X_test, len(mlb.classes_), axis=0)

# Create datasets
lgb_train = lgb.Dataset(X_train_rep, label=y_train_flat)
lgb_eval = lgb.Dataset(X_test_rep, label=y_test_flat, reference=lgb_train)

# Parameters for LightGBM
params = {
    'objective': 'binary',  # We treat each label as binary classification
    'boosting_type': 'gbdt',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# Alternative approach with built-in handling
# We can also use LightGBM's built-in handling for binary objective with class weights
print("Training model...")
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=10)]
)

# Make predictions
print("Making predictions...")
y_pred_proba = gbm.predict(X_test_rep)
y_pred_proba = y_pred_proba.reshape(len(X_test), len(mlb.classes_))

# Find optimal thresholds for each class
thresholds = []
for i in range(len(mlb.classes_)):
    # You can use validation data to optimize thresholds
    # A simple approach is to use a lower threshold for rare classes
    if class_counts[i] < 200:  # For rare classes
        thresholds.append(0.3)
    else:
        thresholds.append(0.5)

# Apply thresholds
y_pred = np.zeros_like(y_pred_proba)
for i in range(len(mlb.classes_)):
    y_pred[:, i] = (y_pred_proba[:, i] >= thresholds[i]).astype(int)

# Evaluate model
print("Evaluating model...")
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

# Print metrics for each class
print("\nClass-wise evaluation:")
for i, label in enumerate(mlb.classes_):
    print(f"{label}:")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall: {recall[i]:.4f}")
    print(f"  F1-score: {f1[i]:.4f}")
    print(f"  Support: {class_counts[i]}")

# Print average metrics
print("\nAggregate metrics:")
print("Macro avg:")
print(f"  Precision: {np.mean(precision):.4f}")
print(f"  Recall: {np.mean(recall):.4f}")
print(f"  F1-score: {np.mean(f1):.4f}")

# Save model
print("Saving model...")
model_path = os.path.join("models", "lightgbm_multilabel.txt")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
gbm.save_model(model_path)
print(f"Model saved to {model_path}")

# Save label encoder
label_encoder_path = os.path.join("models", "multilabel_binarizer.json")
with open(label_encoder_path, 'w') as f:
    json.dump({
        'classes': mlb.classes_.tolist()
    }, f)
print(f"Label encoder saved to {label_encoder_path}") 