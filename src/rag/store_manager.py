import faiss
import numpy as np
from typing import List, Dict, Optional
from pymongo import MongoClient
from datetime import datetime
import uuid
import logging

class StoreManager:
    def __init__(self, mongo_uri: str, dimension: int = 1024):
        # Initialize MongoDB
        self.mongo_client = MongoClient(
            mongo_uri,
            tlsAllowInvalidCertificates=True  # This bypasses SSL certificate verification
        )
        self.db = self.mongo_client.manipulation_db
        self.analyses = self.db.analyses
        
        # Initialize FAISS index
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Load existing vectors if any
        self._load_existing_vectors()
        logging.info("StoreManager initialized successfully")
    
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

    def add_text(self, text: str, embedding: np.ndarray, techniques: List[str], 
                trigger_positions: Optional[List[List[int]]] = None, 
                manipulative: bool = False, 
                doc_id: Optional[str] = None) -> str:
        """Add new text to the database with trigger word information and manipulation flag."""
        try:
            # Check if document with this ID already exists
            if doc_id and self.analyses.find_one({"_id": doc_id}):
                logging.info(f"Document with ID {doc_id} already exists, skipping")
                return doc_id
            
            # Convert numpy array to list for MongoDB storage
            embedding_list = embedding.tolist()
            
            # Ensure manipulative is a Python bool and techniques is a list
            manipulative = bool(manipulative)

            techniques = list(techniques) if techniques else []
            # Prepare document
            doc = {
                "_id": doc_id or str(uuid.uuid4()),
                "embedding": embedding_list,
                "techniques": techniques,
                "manipulative": manipulative,
                "created_at": datetime.now(),
                "original_text": text
            }

            logging.info(f"Trigger techniques: {techniques}")
            
            if trigger_positions is not None:
                doc["trigger_positions"] = [pos.tolist() if isinstance(pos, np.ndarray) else pos 
                                         for pos in trigger_positions]
            
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

    def find_similar(self, embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Find similar texts using the provided embedding."""
        try:
            # Search in FAISS
            D, I = self.index.search(np.array([embedding]).astype('float32'), k)
            
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
                total = 0
                manipulative_count = 0
                manipulation_ratio = 0
            
            stats = {
                "techniques": technique_stats,
                "manipulation_ratio": manipulation_ratio,
                "total_documents": total,
                "manipulative_documents": manipulative_count
            }
            
            logging.info(f"Generated statistics for {len(technique_stats)} techniques")
            return stats
            
        except Exception as e:
            logging.error(f"Error getting statistics: {e}")
            raise 