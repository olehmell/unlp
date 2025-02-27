from typing import List, Dict
import logging
import os
from dotenv import load_dotenv
from mistralai import Mistral

# Load environment variables
load_dotenv()

class LLMManager:
    def __init__(self):
        self.api_key = "Mw1Qh5yjUhi3Ko5T3YQoegIKCpZDATar" # os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")
        self.client = Mistral(api_key=self.api_key)
        self.model = "mistral-large-latest"
        
    def build_prompt(self, query_text: str, similar_texts: List[Dict]) -> str:
        """Build a prompt for the LLM using the query and similar texts."""
        prompt = f"""Analyze the following text for potential manipulation techniques:

Query text: {query_text}

Similar texts and their identified manipulation techniques:

"""
        for i, text in enumerate(similar_texts, 1):
            prompt += f"\n{i}. Text: {text.get('original_text', 'N/A')[:200]}..."
            prompt += f"\nTechniques: {', '.join(text['techniques'])}"
            prompt += f"\nManipulative: {text['manipulative']}"
            prompt += f"\nSimilarity score: {text['similarity_score']:.4f}\n"

        prompt += """\nBased on the similar texts above, please:
1. Identify potential manipulation techniques in the query text
2. Explain why each technique is present
3. Rate the likelihood of manipulation (0-100%)
4. Provide suggestions for critical evaluation

Your analysis:"""

        return prompt

    async def get_analysis(self, prompt: str) -> Dict:
        """Get analysis from the LLM using the provided prompt."""
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing manipulation techniques in text."},
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_object"
                }
            )

            analysis = response.choices[0].message.content
            
            return {
                "analysis": analysis,
                "model": self.model,
                "usage": response.usage.total_tokens
            }
            
        except Exception as e:
            logging.error(f"Error getting LLM analysis: {e}")
            raise

    def format_analysis(self, analysis_result: Dict) -> str:
        """Format the analysis result for presentation."""
        return f"""
Analysis Results:
----------------
{analysis_result['analysis']}

Model: {analysis_result['model']}
Tokens used: {analysis_result['usage']}
""" 