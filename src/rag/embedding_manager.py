import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import logging
from typing import List
import re
import emoji

class EmbeddingManager:
    def __init__(self):
        # Initialize E5 model
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to(self.device)
        self.model.eval()  # Set to evaluation mode
        self.dimension = 1024  # E5 large embedding dimension
        
    def _preprocess_text(self, text: str) -> str:
        """Clean text before embedding.
        
        Removes:
        - Emojis
        - URLs
        - Converts to lowercase
        """
        # Remove emojis
        text = emoji.replace_emoji(text, '')
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    def get_embedding(self, text: str, max_length: int = 512) -> np.ndarray:
        """Get embeddings using E5 model with support for longer texts."""
        try:
            # Preprocess text
            text = self._preprocess_text(text)
            
            # Split text into chunks if it's too long
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                word_tokens = len(self.tokenizer.encode(word))
                if current_length + word_tokens > max_length:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = word_tokens
                else:
                    current_chunk.append(word)
                    current_length += word_tokens
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            if not chunks:
                chunks = [text]
            
            # Get embeddings for each chunk
            embeddings = []
            for chunk in chunks:
                chunk_text = f"passage: {chunk}"
                inputs = self.tokenizer(
                    chunk_text,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    chunk_embedding = self._average_pool(outputs.last_hidden_state, inputs['attention_mask'])
                    chunk_embedding = torch.nn.functional.normalize(chunk_embedding, p=2, dim=1)
                    embeddings.append(chunk_embedding.cpu().numpy()[0])
            
            # Average all chunk embeddings
            final_embedding = np.mean(embeddings, axis=0)
            return final_embedding
            
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            raise

    def _average_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Average pooling of the last hidden states using attention mask.
        This is the recommended pooling strategy for E5 models.
        """
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def enrich_embeddings(self, text: str, trigger_positions: List[List[int]]) -> np.ndarray:
        """
        Enrich text embeddings with trigger word information using an advanced weighting scheme.
        
        This method creates a combined embedding that emphasizes both the full context and trigger words.
        The weighting scheme considers:
        1. Full text embedding (highest weight) to maintain overall context
        2. Individual trigger word embeddings weighted by their relative position and length
        
        Args:
            text: The full text to analyze
            trigger_positions: List of [start, end] positions for trigger words
        
        Returns:
            Combined embedding vector that emphasizes both context and trigger words
        """
        try:
            # Get base text embedding
            text_embedding = self.get_embedding(text)
            
            logging.info(f"Trigger positions: {trigger_positions}")
            
            # Get embeddings for trigger words with position-based weights
            trigger_embeddings = []
            trigger_weights = []
            text_length = len(text)
            
            for start, end in trigger_positions:
                trigger_text = text[start:end]
                trigger_emb = self.get_embedding(trigger_text)
                
                # Calculate weight based on position and length
                position_weight = 1.0 - (start / text_length) * 0.3  # Words earlier in text get slightly more weight
                length_weight = len(trigger_text) / len(text)  # Longer triggers get proportionally more weight
                weight = 0.5 * (position_weight + length_weight)  # Combine factors, max weight 0.5
                
                trigger_embeddings.append(trigger_emb)
                trigger_weights.append(weight)
            
            # Combine embeddings with weighted average
            # Base text gets weight 1.0 to maintain overall context
            all_embeddings = [text_embedding] + trigger_embeddings
            weights = [1.0] + trigger_weights
            
            # Normalize weights and compute weighted sum
            weights = np.array(weights) / sum(weights)
            weighted_sum = np.sum(
                [w * e for w, e in zip(weights, all_embeddings)], 
                axis=0
            )
            
            return weighted_sum
            
        except Exception as e:
            logging.error(f"Error in embedding enrichment: {e}")
            return text_embedding  # Fallback to base embedding