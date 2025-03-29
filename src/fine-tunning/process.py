import os
import time
import json
import logging
import asyncio
import pandas as pd
from mistralai import Mistral
from dotenv import load_dotenv
from typing import Dict, List, Set
from pydantic import BaseModel, Field
from pyparsing import Enum

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fine_tune_processing.log'),
        logging.StreamHandler()
    ]
)

# Constants
SAVE_BATCH_SIZE = 5  # Save after processing this many texts
MODEL_ID = "ft:mistral-small-latest:e3b79b5f:20250317:23c7051d"  # Your fine-tuned model ID
# MODEL_ID="ft:open-mistral-nemo:e3b79b5f:20250318:ad997c5f"

class ManipulationTechnique(str, Enum):
    STRAW_MAN = "straw_man"
    APPEAL_TO_FEAR = "appeal_to_fear"
    FUD = "fud"
    BANDWAGON = "bandwagon"
    WHATABOUTISM = "whataboutism"
    LOADED_LANGUAGE = "loaded_language"
    GLITTERING_GENERALITIES = "glittering_generalities"
    EUPHORIA = "euphoria"
    CHERRY_PICKING = "cherry_picking"
    CLICHE = "cliche"

class TextAnalysis(BaseModel):
    """Data model for text analysis output."""
    manipulation_techniques: List[ManipulationTechnique] = Field(default_factory=list)

def load_test_data(csv_path: str = None) -> pd.DataFrame:
    """Load test data from CSV file."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'csv', 'test.csv')
    
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded {len(df)} rows from {csv_path}")
        return df.sample(n=10, random_state=42).reset_index(drop=True)
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        raise

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

def append_results_batch(results_batch: List[Dict]) -> None:
    """
    Append a batch of results to the JSONL file.
    
    Args:
        results_batch: List of dictionaries with result data
    """
    if not results_batch:
        return
    
    jsonl_path = get_results_jsonl_path()
    
    try:
        with open(jsonl_path, 'a', encoding='utf-8') as f:
            for result in results_batch:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logging.info(f"Appended {len(results_batch)} results to {jsonl_path}")
    except Exception as e:
        logging.error(f"Error appending results to JSONL: {e}")
        raise

class FineTunedProcessor:
    def __init__(self):
        self.api_key = "qW37oQZvs6c93c2jEAmtwYZq6scsGo1i" #os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable must be set")
        print(f"API Key is: {self.api_key}")
        self.client = Mistral(api_key=self.api_key)

        self.model = MODEL_ID
        logging.info(f"FineTunedProcessor initialized with model: {self.model}")

    async def process_text(self, text: str) -> Dict:
        """Process text using the fine-tuned model."""
        try:
            response = await asyncio.to_thread(
                self.client.chat.parse,
                model=self.model,
                messages=[
                    {
                        "role": "user", 
                        "content": f"Analyze the following post and list any manipulation techniques used:\n\nPost: {text}"
                    }
                ],
                response_format=TextAnalysis,
                temperature=0.1
            )
            
            # Extract the parsed result
            analysis = response.choices[0].message.parsed

            print(f"Analysis: {analysis}")
            
            return analysis.dict()
            
        except Exception as e:
            logging.error(f"Error processing text: {e}")
            raise

async def main():
    try:
        # Initialize processor
        processor = FineTunedProcessor()
        
        # Load test data from CSV
        test_data = load_test_data()
        logging.info(f"Loaded {len(test_data)} test cases from CSV")
        
        # Load processed IDs from JSONL file
        processed_ids = load_processed_ids()
        logging.info(f"Already processed {len(processed_ids)} texts")
        
        # Prepare for batch processing
        start_time = time.time()
        results_batch = []
        processed_count = 0
        total_count = len(test_data)
        
        # Process each test case
        for idx, row in test_data.iterrows():
            text_id = row['id']
            
            # Skip if already processed
            if text_id in processed_ids:
                logging.info(f"Skipping already processed ID: {text_id}")
                continue
            
            print(f"\nProcessing text with ID: {text_id} ({processed_count + 1}/{total_count - len(processed_ids)})")
            
            try:
                # Add a sleep interval before making the next request
                await asyncio.sleep(0.5)
                
                # Process text with fine-tuned model
                analysis_result = await processor.process_text(row['content'])
                
                # Extract techniques and print result
                identified_techniques = analysis_result['manipulation_techniques']
                print("\nIdentified techniques:", identified_techniques)
                
                # Add result to batch
                results_batch.append({
                    'id': text_id,
                    'content': row['content'],
                    'techniques': identified_techniques,
                    'timestamp': time.time()
                })
                
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