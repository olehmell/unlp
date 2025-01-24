import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Tuple

def get_model_config(model_name: str = "answerdotai/ModernBERT-base") -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Initialize model and tokenizer based on model name.

    Args:
        model_name (str): Name of the model to initialize

    Returns:
        Tuple[AutoModelForSequenceClassification, AutoTokenizer]: Initialized model and tokenizer
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=10,  # Number of manipulation techniques
        problem_type="multi_label_classification",
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    
    return model, tokenizer