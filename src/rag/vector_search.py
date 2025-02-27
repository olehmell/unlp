import faiss
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pymongo import MongoClient
from datetime import datetime
import uuid
import logging
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_search.log'),
        logging.StreamHandler()
    ]
)

class ManipulationVectorDB:
    def __init__(self, mongo_uri: str):
        # Initialize E5 model
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') # TODO: change to cuda
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Initialize MongoDB
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client.manipulation_db
        self.analyses = self.db.analyses
        
        # Initialize FAISS index
        self.dimension = 1024  # E5 large embedding dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Load existing vectors if any
        self._load_existing_vectors()
        logging.info("VectorDB initialized successfully")
    
    def _load_existing_vectors(self):
        """Load existing vectors from MongoDB to FAISS."""
        existing_docs = self.analyses.find({}, {'embedding': 1, '_id': 1})
        vectors = []
        self.id_mapping = []
        
        for doc in existing_docs:
            if 'embedding' in doc:
                vectors.append(doc['embedding'])
                self.id_mapping.append(str(doc['_id']))
        
        if vectors:
            vectors_np = np.array(vectors).astype('float32')
            self.index.add(vectors_np)
            logging.info(f"Loaded {len(vectors)} existing vectors")
    
    def get_embedding(self, text: str, max_length: int = 512) -> np.ndarray:
        """Get embeddings using E5 model with support for longer texts."""
        try:
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
                    chunk_embedding = self.average_pool(outputs.last_hidden_state, inputs['attention_mask'])
                    chunk_embedding = torch.nn.functional.normalize(chunk_embedding, p=2, dim=1)
                    embeddings.append(chunk_embedding.cpu().numpy()[0])
            
            # Average all chunk embeddings
            final_embedding = np.mean(embeddings, axis=0)
            return final_embedding
            
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            raise

    def average_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Average pooling of the last hidden states using attention mask.
        This is the recommended pooling strategy for E5 models.
        """
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def enrich_embeddings(self, text: str, trigger_positions: List[List[int]]) -> np.ndarray:
        """
        Enrich text embeddings with trigger word information.
        
        Args:
            text: The full text
            trigger_positions: List of [start, end] positions for trigger words
        
        Returns:
            Combined embedding vector
        """
        try:
            # Get base text embedding
            text_embedding = self.get_embedding(text)
            
            # Get embeddings for trigger words
            trigger_texts = [text[start:end] for start, end in trigger_positions]
            trigger_embeddings = []
            
            for trigger_text in trigger_texts:
                trigger_emb = self.get_embedding(trigger_text)
                trigger_embeddings.append(trigger_emb)
            
            # Combine embeddings with weighted average
            all_embeddings = [text_embedding] + trigger_embeddings
            weights = [1.0] + [0.5] * len(trigger_embeddings)  # Base text gets more weight
            weighted_sum = np.sum(
                [w * e for w, e in zip(weights, all_embeddings)], 
                axis=0
            )
            return weighted_sum / sum(weights)
            
        except Exception as e:
            logging.error(f"Error in embedding enrichment: {e}")
            return text_embedding  # Fallback to base embedding
    
    def create_attention_mask(self, text_length: int, trigger_positions: List[List[int]]) -> np.ndarray:
        """
        Create attention mask highlighting trigger words.
        
        Args:
            text_length: Length of the full text
            trigger_positions: List of [start, end] positions for trigger words
        
        Returns:
            Numpy array with attention weights
        """
        mask = np.ones(text_length) * 0.5  # Base attention weight
        for start, end in trigger_positions:
            mask[start:end] = 1.0  # Higher attention to trigger words
        return mask

    def add_text(self, text: str, techniques: List[str], trigger_positions: Optional[List[List[int]]] = None, 
                 store_full_text: bool = False, manipulative: bool = False, doc_id: Optional[str] = None) -> str:
        """Add new text to the database with trigger word information and manipulation flag."""
        try:
            # Check if document with this ID already exists
            if doc_id and self.analyses.find_one({"_id": doc_id}):
                logging.info(f"Document with ID {doc_id} already exists, skipping")
                return doc_id
            
            is_trigger_words = trigger_positions is not None

            # Get enriched embedding
            if is_trigger_words:
                logging.info(f"Enriching embedding with trigger words")
                embedding = self.enrich_embeddings(text, trigger_positions)
            else:
                logging.info(f"Getting embedding without trigger words")
                embedding = self.get_embedding(text)
            
            # Convert numpy array to list for MongoDB storage
            embedding_list = embedding.tolist()
            
            # Ensure manipulative is a Python bool and techniques is a list
            manipulative = bool(manipulative)
            techniques = list(techniques) if techniques else []
            
            logging.info(f"Embedding created with techniques: {techniques}")
            
            # Prepare document
            doc = {
                "_id": doc_id or str(uuid.uuid4()),
                "embedding": embedding_list,
                "techniques": techniques,
                "manipulative": manipulative,
                "created_at": datetime.now()
            }
            
            if is_trigger_words:
                logging.info(f"Adding trigger positions")
                doc["trigger_positions"] = [pos.tolist() if isinstance(pos, np.ndarray) else pos 
                                  for pos in trigger_positions]
                doc["attention_mask"] = self.create_attention_mask(len(text), trigger_positions).tolist()
            
            if store_full_text:
                logging.info(f"Adding full text")
                doc["original_text"] = text
            
            logging.info(f"Adding to MongoDB with techniques: {doc['techniques']}")
            # Add to MongoDB
            self.analyses.insert_one(doc)

            logging.info(f"Adding to FAISS")
            # Add to FAISS
            self.index.add(np.array([embedding]).astype('float32'))
            self.id_mapping.append(doc["_id"])
            
            logging.info(f"Successfully added text with ID: {doc['_id']}, manipulative: {manipulative}, techniques: {techniques}")
            return doc["_id"]
            
        except Exception as e:
            logging.error(f"Error adding text: {e}")
            raise
    
    def find_similar(self, text: str, trigger_positions: Optional[List[List[int]]] = None, k: int = 5) -> List[Dict]:
        """Find similar texts using enriched embeddings when trigger words are present."""
        try:
            # Get enriched embedding for query
            if trigger_positions:
                query_embedding = self.enrich_embeddings(text, trigger_positions)
            else:
                query_embedding = self.get_embedding(text)
            
            # Search in FAISS
            D, I = self.index.search(np.array([query_embedding]).astype('float32'), k)
            
            # Get results from MongoDB
            results = []
            for idx in I[0]:
                if idx < len(self.id_mapping):
                    doc_id = self.id_mapping[idx]
                    doc = self.analyses.find_one({"_id": doc_id})
                    if doc:
                        result = {
                            "id": doc["_id"],
                            "techniques": doc["techniques"],
                            "manipulative": doc["manipulative"],
                            "similarity_score": float(D[0][len(results)])
                        }
                        
                        if "trigger_positions" in doc:
                            result["trigger_positions"] = doc["trigger_positions"]
                        
                        if "original_text" in doc:
                            result["original_text"] = doc["original_text"]
                            
                        results.append(result)
            
            logging.info(f"Found {len(results)} similar texts")
            return results
            
        except Exception as e:
            logging.error(f"Error finding similar texts: {e}")
            raise
    
    def bulk_add_from(self, path: str, text_col: str, techniques_col: str, 
                     trigger_positions_col: Optional[str] = None,
                     manipulative_col: str = 'manipulative',
                     id_col: str = 'id',
                     store_full_text: bool = False, batch_size: int = 10):
        """Add multiple texts from parquet file with trigger word support and manipulation flag."""
        try:
            df = pd.read_parquet(path)
            total = len(df)
            
            for i in range(0, total, batch_size):
                batch = df.iloc[i:i+batch_size]
                for _, row in batch.iterrows():
                    text = row[text_col]
                    logging.info(f"Processing techniques: {row[techniques_col]}")
                    # Convert techniques to list, handling different input formats
                    if isinstance(row[techniques_col], str):
                        # Handle string format like "['loaded_language' 'bandwagon']"
                        techniques = row[techniques_col].strip('[]').replace("'", "").split()
                    elif isinstance(row[techniques_col], (list, np.ndarray)):
                        # Handle list or numpy array
                        techniques = list(row[techniques_col])
                    else:
                        techniques = []
                    logging.info(f"Processing techniques res: {techniques}")
                    trigger_positions = row.get(trigger_positions_col, []) if trigger_positions_col else None
                    manipulative = bool(row.get(manipulative_col, False))
                    doc_id = str(row.get(id_col)) if id_col in row else None  # Get document ID
                    
                    self.add_text(
                        text, 
                        techniques, 
                        trigger_positions, 
                        store_full_text, 
                        manipulative,
                        doc_id
                    )
                logging.info(f"Processed {i+len(batch)}/{total} texts")
                
        except Exception as e:
            logging.error(f"Error in bulk add: {e}")
            raise
    
    def get_technique_statistics(self) -> Dict:
        """Get statistics about stored manipulation techniques and manipulation ratio."""
        try:
            # Get technique statistics
            technique_pipeline = [
                {"$unwind": "$techniques"},
                {"$group": {
                    "_id": "$techniques",
                    "count": {"$sum": 1},
                    "manipulative_count": {
                        "$sum": {"$cond": ["$manipulative", 1, 0]}
                    }
                }},
                {"$sort": {"count": -1}}
            ]
            
            # Get overall manipulation statistics
            manipulation_pipeline = [
                {"$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "manipulative_count": {
                        "$sum": {"$cond": ["$manipulative", 1, 0]}
                    }
                }}
            ]
            
            technique_stats = list(self.analyses.aggregate(technique_pipeline))
            manipulation_stats = list(self.analyses.aggregate(manipulation_pipeline))
            
            # Calculate percentages
            if manipulation_stats:
                total = manipulation_stats[0]["total"]
                manipulative_count = manipulation_stats[0]["manipulative_count"]
                manipulation_ratio = manipulative_count / total if total > 0 else 0
            else:
                manipulation_ratio = 0
            
            stats = {
                "techniques": technique_stats,
                "manipulation_ratio": manipulation_ratio,
                "total_documents": total if manipulation_stats else 0,
                "manipulative_documents": manipulative_count if manipulation_stats else 0
            }
            
            logging.info(f"Generated statistics for {len(technique_stats)} techniques")
            return stats
            
        except Exception as e:
            logging.error(f"Error getting statistics: {e}")
            raise

def main():
    try:
        # Initialize database
        db = ManipulationVectorDB(os.getenv('MONGO_URI'))
        
        # Add sample data
        # db.bulk_add_from(
        #     'data/bin/train.parquet',
        #     'content',
        #     'techniques',
        #     trigger_positions_col='trigger_words',
        #     manipulative_col='manipulative',
        #     id_col='id',
        #     store_full_text=True
        # )
        
        # Test similarity search
        test_text = "Новий огляд мапи DeepState від російського військового експерта, кухара путіна 2 розряду, спеціаліста по снарядному голоду та ректора музичної академії міноборони рф Євгєнія Пригожина. Пригожин прогнозує, що невдовзі настане день звільнення Криму і день розпаду росії. Каже, що передумови цього вже створені. *Відео взяли з каналу ФД. @informnapalm"
        similar_texts = db.find_similar(test_text, k=5)
        
        for result in similar_texts:
            logging.info(f"\nSimilar text found:")
            logging.info(f"Techniques: {result['techniques']}")
            logging.info(f"Manipulative: {result['manipulative']}")
            logging.info(f"Similarity score: {result['similarity_score']}")
            if "original_text" in result:
                logging.info(f"Text: {result['original_text'][:200]}...")
            
    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()