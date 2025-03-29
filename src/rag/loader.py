import pandas as pd
import logging
import os
from dotenv import load_dotenv
from embedding_manager import EmbeddingManager
from store_manager import StoreManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('loader.log'),
        logging.StreamHandler()
    ]
)

def load_data(file_path: str, batch_size: int = 10) -> None:
    """
    Load data from a parquet file into the vector store.
    
    Args:
        file_path: Path to the parquet file
        batch_size: Number of records to process at once
    """
    try:
        # Initialize managers
        embedding_manager = EmbeddingManager()
        store_manager = StoreManager(os.getenv('MONGO_URI'))
        
        # Load data
        df = pd.read_parquet(file_path)
        total = len(df)
        logging.info(f"Loading {total} records from {file_path}")
        
        # Process in batches
        for i in range(0, total, batch_size):
            batch = df.iloc[i:i+batch_size]
            for _, row in batch.iterrows():
                try:
                    # Get text and metadata
                    text = row['content']
                    techniques = row['techniques']
                    
                    if techniques is None:
                        techniques = []
                    else:
                        techniques = list(row['techniques'])

                    manipulative = bool(row.get('manipulative', False))
                    doc_id = str(row['id']) if 'id' in row else None

                    # Get trigger positions if available
                    trigger_positions = row.get('trigger_words', [])
                    
                    # Get embedding
                    if trigger_positions is not None:
                        embedding = embedding_manager.enrich_embeddings(text, trigger_positions)
                    else:   
                        embedding = embedding_manager.get_embedding(text)
                    
                    # Store in database
                    store_manager.add_text(
                        text=text,
                        embedding=embedding,
                        techniques=techniques,
                        trigger_positions=trigger_positions,
                        manipulative=manipulative,
                        doc_id=doc_id
                    )
                    
                except Exception as e:
                    logging.error(f"Error processing record: {e}")
                    continue
            
            logging.info(f"Processed {i+len(batch)}/{total} records")
            
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    load_data(
        file_path='data/bin/train.parquet',
        batch_size=10
    ) 
    