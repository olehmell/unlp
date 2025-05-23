import asyncio
import logging
import os
import pandas as pd
import json
import time
from dotenv import load_dotenv
from typing import Dict, List, Set
from embedding_manager import EmbeddingManager
from store_manager import StoreManager
from llm_manager import LLMManager

# Load environment variables
load_dotenv()

test_ids = [
    "7159f802-6f99-4e9d-97bd-6f565a4a0fae",
    "d3d66069-5f9b-4e54-970b-e634ca345f3b",
    "00951c6b-3b44-4a24-b7da-f1a5c4093ed6",
    "ea68d4c2-8817-4182-b0db-8ef88726cef0",
    "6b3741e5-4952-4d7a-90b0-abdab5d8d497",
]

# Constants
SAVE_BATCH_SIZE = 5  # Save after processing this many texts

def load_test_data() -> pd.DataFrame:
    """Load test data from CSV file."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'csv', 'test.csv')
    df = pd.read_csv(csv_path)
    # return df[df['id'].isin(test_ids)]
    return df

def get_results_jsonl_path() -> str:
    """Get path to results JSONL file."""
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'json', 'results.jsonl')

def load_processed_ids() -> Set[str]:
    """
    Load IDs of texts that have already been processed from the JSONL file.
    
    Returns:
        Set of processed text IDs
    """
    processed_ids = set()
    jsonl_path = get_results_jsonl_path()
    
    if os.path.exists(jsonl_path):
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed_ids.add(data.get('id'))
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping invalid JSON line in results file")
        except Exception as e:
            logging.error(f"Error loading results JSONL: {e}")
    
    return processed_ids

def append_results_batch(results_batch: List[tuple]) -> None:
    """
    Append a batch of results to the JSONL file.
    
    Args:
        results_batch: List of tuples (text_id, identified_techniques)
    """
    if not results_batch:
        return
    
    jsonl_path = get_results_jsonl_path()
    
    try:
        with open(jsonl_path, 'a', encoding='utf-8') as f:
            for text_id, techniques in results_batch:
                result = {
                    'id': text_id,
                    'techniques': techniques,
                    'timestamp': time.time()
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logging.info(f"Appended {len(results_batch)} results to {jsonl_path}")
    except Exception as e:
        logging.error(f"Error appending results to JSONL: {e}")
        raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('manipulation_analysis.log'),
        logging.StreamHandler()
    ]
)

class ManipulationAnalyzer:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.store_manager = StoreManager(os.getenv('MONGO_URI'))
        self.llm_manager = LLMManager()
        logging.info("ManipulationAnalyzer initialized successfully")
    
    async def analyze_text(self, text: str, k: int = 10) -> Dict:
        """
        Analyze a text for manipulation techniques using the RAG pipeline.
        
        Args:
            text: The text to analyze
            k: Number of similar texts to retrieve
            
        Returns:
            LLM output directly without additional formatting
        """
        try:
            # Get embedding for the query text
            embedding = self.embedding_manager.get_embedding(text)
            
            # Find similar texts
            similar_texts = self.store_manager.find_similar(embedding, k=k)
            
            # Build prompt and get LLM analysis
            prompt = self.llm_manager.build_prompt(text, similar_texts)

            # Get raw LLM output
            llm_response = await self.llm_manager.get_analysis(prompt)
            
            # Return the raw LLM output
            return llm_response['analysis']
            
        except Exception as e:
            logging.error(f"Error analyzing text: {e}")
            raise

async def main():
    try:
        # Initialize analyzer
        analyzer = ManipulationAnalyzer()
        
        # Load test data from CSV
        test_data = load_test_data()
        logging.info(f"Loaded {len(test_data)} test cases from CSV")
        
        # Load or create results JSON and get processed IDs
        processed_ids = load_processed_ids()
        logging.info(f"Already processed {len(processed_ids)} texts")
        
        # Prepare for batch processing
        start_time = time.time()
        results_batch = []
        processed_count = 0
        total_count = len(test_data)
        
        # Process each test case
        for _, row in test_data.iterrows():
            text_id = row['id']
            
            # Skip if already processed
            if text_id in processed_ids:
                logging.info(f"Skipping already processed ID: {text_id}")
                continue
            
            print(f"\nProcessing text with ID: {text_id} ({processed_count + 1}/{total_count - len(processed_ids)})")
            
            try:
                # Add a sleep interval before making the next request
                await asyncio.sleep(2)
                
                # Run analysis and get raw LLM output
                llm_output = await analyzer.analyze_text(row['content'], k=10)
                
                # Extract techniques and add to batch
                identified_techniques = llm_output['manipulation_techniques']
                print("\nIdentified techniques:", identified_techniques)
                
                results_batch.append((text_id, identified_techniques))
                processed_ids.add(text_id)
                processed_count += 1
                
                # Save batch if threshold reached
                if len(results_batch) >= SAVE_BATCH_SIZE:
                    append_results_batch(results_batch)
                    results_batch = []
                    
                    # Show progress
                    elapsed = time.time() - start_time
                    avg_time = elapsed / processed_count
                    remaining = (total_count - len(processed_ids) - processed_count) * avg_time
                    print(f"\nProgress: {processed_count}/{total_count - len(processed_ids)} texts processed")
                    print(f"Avg. time per text: {avg_time:.2f}s, Est. remaining time: {remaining/60:.2f} minutes")
                
            except Exception as e:
                logging.error(f"Error processing text ID {text_id}: {e}")
                print(f"Error processing text ID {text_id}: {e}")
                # Continue with next text
                continue
            
            print("\n" + "="*50)
        
        # Save any remaining results
        if results_batch:
            append_results_batch(results_batch)
        
        total_time = time.time() - start_time
        logging.info(f"Completed processing {processed_count} texts in {total_time/60:.2f} minutes")
        print(f"\nCompleted processing {processed_count} texts in {total_time/60:.2f} minutes")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())