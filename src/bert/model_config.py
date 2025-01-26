import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Tuple
import os

def get_model_config(model_name: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Initialize model and tokenizer based on model name.

    Args:
        model_name (str): Name of the model to initialize. If the model name is a local path,
                         it will load a fine-tuned model. Otherwise, it will load from HuggingFace Hub.

    Returns:
        Tuple[AutoModelForSequenceClassification, AutoTokenizer]: Initialized model and tokenizer
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Check if model_name is a local path
    is_local_model = os.path.exists(model_name)
    if is_local_model and not os.path.exists(os.path.join(model_name, "config.json")):
        raise ValueError(f"Model directory '{model_name}' exists but doesn't contain a valid model (no config.json found)")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        
        # Load model with appropriate configuration
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=10,  # Number of manipulation techniques
            problem_type="multi_label_classification",
            device_map=device,
            torch_dtype=torch.bfloat16,
            local_files_only=is_local_model
        )
        
        return model, tokenizer
    
    except Exception as e:
        raise Exception(f"Failed to load model from '{model_name}'. Error: {str(e)}")