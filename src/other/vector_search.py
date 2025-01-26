import faiss
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from openai import OpenAI
import voyageai
from pymongo import MongoClient
import json
from datetime import datetime
import uuid
import logging
import os
from dotenv import load_dotenv

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
        # Initialize clients
        self.vo = voyageai.Client(api_key=os.getenv('VOYAGE_API_KEY'))
        self.llm = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )
        
        # Initialize MongoDB
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client.manipulation_db
        self.analyses = self.db.analyses
        
        # Initialize FAISS index
        self.dimension = 1024  # Voyage AI embedding dimension
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
    
    def _analyze_with_llm(self, text: str) -> Dict:
        """Get structured analysis from DeepSeek."""
        prompt = """Analyze this text for manipulation techniques. Return JSON with:
        {
            "key_phrases": [list of key phrases],
            "manipulation_markers": [list of manipulation indicators],
            "emotional_triggers": [list of emotional triggers],
            "rhetoric_devices": [list of rhetorical devices used],
            "summary": "brief analysis of manipulation techniques"
        }

        Text: """ + text
        
        try:
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes text for manipulation techniques. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content
            logging.info(f"\nLLM Response for text: {text[:100]}...\n{content}")
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON if response contains additional text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                raise
                
        except Exception as e:
            logging.error(f"Error in DeepSeek analysis: {e}")
            return {
                "key_phrases": [],
                "manipulation_markers": [],
                "emotional_triggers": [],
                "rhetoric_devices": [],
                "summary": "Analysis failed"
            }
    
    def add_text(self, text: str, techniques: List[str], store_full_text: bool = False) -> str:
        """Add new text to the database."""
        try:
            # Get embedding
            embedding = self.vo.embed([text], model="voyage-3-lite", input_type="document").embeddings[0]
            
            # Get analysis
            analysis = self._analyze_with_llm(text)
            
            # Prepare document
            doc = {
                "_id": str(uuid.uuid4()),
                "embedding": embedding.tolist(),
                "analysis": analysis,
                "techniques": techniques,
                "created_at": datetime.utcnow()
            }
            
            if store_full_text:
                doc["original_text"] = text
            
            # Add to MongoDB
            self.analyses.insert_one(doc)
            
            # Add to FAISS
            self.index.add(np.array([embedding]).astype('float32'))
            self.id_mapping.append(doc["_id"])
            
            logging.info(f"Successfully added text with ID: {doc['_id']}")
            return doc["_id"]
            
        except Exception as e:
            logging.error(f"Error adding text: {e}")
            raise
    
    def find_similar(self, text: str, k: int = 5) -> List[Dict]:
        """Find similar texts and their manipulation techniques."""
        try:
            # Get embedding for query
            query_embedding = self.vo.embed([text], model="voyage-3-lite", input_type="document").embeddings[0]
            
            # Search in FAISS
            D, I = self.index.search(np.array([query_embedding]).astype('float32'), k)
            
            # Get results from MongoDB
            results = []
            for idx in I[0]:
                if idx < len(self.id_mapping):
                    doc_id = self.id_mapping[idx]
                    doc = self.analyses.find_one({"_id": doc_id})
                    if doc:
                        results.append({
                            "id": doc["_id"],
                            "techniques": doc["techniques"],
                            "analysis": doc["analysis"],
                            "original_text": doc.get("original_text"),
                            "similarity_score": float(D[0][len(results)])
                        })
            
            logging.info(f"Found {len(results)} similar texts")
            return results
            
        except Exception as e:
            logging.error(f"Error finding similar texts: {e}")
            raise
    
    def bulk_add_from_csv(self, csv_path: str, text_col: str, techniques_col: str, 
                         store_full_text: bool = False, batch_size: int = 10):
        """Add multiple texts from CSV file."""
        try:
            df = pd.read_csv(csv_path)
            total = len(df)
            
            for i in range(0, total, batch_size):
                batch = df.iloc[i:i+batch_size]
                for _, row in batch.iterrows():
                    text = row[text_col]
                    techniques = row[techniques_col].split(',') if pd.notna(row[techniques_col]) else []
                    self.add_text(text, techniques, store_full_text)
                logging.info(f"Processed {i+len(batch)}/{total} texts")
                
        except Exception as e:
            logging.error(f"Error in bulk add: {e}")
            raise
    
    def get_technique_statistics(self) -> Dict:
        """Get statistics about stored manipulation techniques."""
        try:
            pipeline = [
                {"$unwind": "$techniques"},
                {"$group": {
                    "_id": "$techniques",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}}
            ]
            
            stats = list(self.analyses.aggregate(pipeline))
            logging.info(f"Generated statistics for {len(stats)} techniques")
            return stats
            
        except Exception as e:
            logging.error(f"Error getting technique statistics: {e}")
            raise

def main():
    try:
        # Initialize database
        db = ManipulationVectorDB("mongodb://localhost:27017/")
        
        # Add sample data
        db.bulk_add_from_csv('data/train.csv', 'content', 'techniques', store_full_text=True)
        
        # Test similarity search
        test_text = "Your test text here"
        similar_texts = db.find_similar(test_text, k=5)
        
        for result in similar_texts:
            logging.info(f"\nSimilar text found:")
            logging.info(f"Techniques: {result['techniques']}")
            logging.info(f"Similarity score: {result['similarity_score']}")
            logging.info(f"Analysis: {result['analysis']['summary']}")
            
    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()