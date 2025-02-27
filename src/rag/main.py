import asyncio
import logging
import os
from dotenv import load_dotenv
from typing import Dict, List
from embedding_manager import EmbeddingManager
from store_manager import StoreManager
from llm_manager import LLMManager

# Load environment variables
load_dotenv()

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
            Dictionary containing the analysis results
        """
        try:
            # Get embedding for the query text
            embedding = self.embedding_manager.get_embedding(text)
            
            # Find similar texts
            similar_texts = self.store_manager.find_similar(embedding, k=k)
            
            # Build prompt and get LLM analysis
            prompt = self.llm_manager.build_prompt(text, similar_texts)
            analysis = await self.llm_manager.get_analysis(prompt)
            
            # Format results
            formatted_analysis = self.llm_manager.format_analysis(analysis)
            
            return {
                "query_text": text,
                "similar_texts": similar_texts,
                "analysis": formatted_analysis
            }
            
        except Exception as e:
            logging.error(f"Error analyzing text: {e}")
            raise

async def main():
    try:
        # Initialize analyzer
        analyzer = ManipulationAnalyzer()
        
        # Example text for analysis
        test_text = """Новий огляд мапи DeepState від російського військового експерта, 
        кухара путіна 2 розряду, спеціаліста по снарядному голоду та ректора музичної 
        академії міноборони рф Євгєнія Пригожина. Пригожин прогнозує, що невдовзі настане 
        день звільнення Криму і день розпаду росії. Каже, що передумови цього вже створені. 
        *Відео взяли з каналу ФД. @informnapalm"""
        
        # Run analysis
        results = await analyzer.analyze_text(test_text, k=10)
        
        # Print results
        print("\nAnalysis Results:")
        print("----------------")
        print(f"Query Text: {results['query_text'][:200]}...")
        print("\nSimilar Texts Found:", len(results['similar_texts']))
        print("\nAnalysis:")
        print(results['analysis'])
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 