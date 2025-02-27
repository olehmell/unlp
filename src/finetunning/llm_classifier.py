import torch
from transformers import TrainingArguments, Trainer
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from model_config import get_model_config

class ManipulationClassifier:
    def __init__(self, model_name):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model, self.tokenizer = get_model_config(model_name)

        # Mapping of technique names to indices
        self.technique_mapping = {
            'loaded_language': 0,
            'glittering_generalities': 1,
            'euphoria': 2,
            'appeal_to_fear': 3,
            'fud': 4,
            'bandwagon': 5,
            'cliche': 6,
            'whataboutism': 7,
            'cherry_picking': 8,
            'straw_man': 9
        }

    def prepare_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Convert techniques to multi-hot encoding
        labels = np.zeros((len(df), len(self.technique_mapping)))
        for idx, techniques in enumerate(df['techniques']):
            if isinstance(techniques, list):
                for technique in techniques:
                    if technique in self.technique_mapping:
                        labels[idx, self.technique_mapping[technique]] = 1

        # Tokenize texts
        encodings = self.tokenizer(
            df['content'].tolist(),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.FloatTensor(labels)
        }

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        output_dir: str = 'models/mdeberta-manipulation',
        num_epochs: int = 3,
        batch_size: int = 8
    ):
        train_dataset = ManipulationDataset(self.prepare_data(train_df))
        val_dataset = ManipulationDataset(self.prepare_data(val_df))

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()

    def predict(self, texts: List[str], threshold: float = 0.5) -> List[List[str]]:
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.sigmoid(outputs.logits)

        predictions = (predictions > threshold).cpu().numpy()
        
        result = []
        for pred in predictions:
            techniques = []
            for technique, idx in self.technique_mapping.items():
                if pred[idx]:
                    techniques.append(technique)
            result.append(techniques)
            
        return result

    def predict_with_scores(self, texts: List[str], threshold: float = 0.5) -> tuple[List[List[str]], List[np.ndarray]]:
        """Predict manipulation techniques and return raw scores."""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encodings)
            scores = torch.sigmoid(outputs.logits)
            predictions = (scores > threshold)
            
            # Convert to float32 before moving to CPU and numpy
            scores = scores.float()
            predictions = predictions.float()

        predictions = predictions.cpu().numpy()
        scores = scores.cpu().numpy()
        
        result = []
        for pred in predictions:
            techniques = []
            for technique, idx in self.technique_mapping.items():
                if pred[idx]:
                    techniques.append(technique)
            result.append(techniques)
            
        return result, scores

class ManipulationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])