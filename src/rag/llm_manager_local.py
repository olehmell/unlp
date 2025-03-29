from typing import List, Dict
import logging
import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

class LLMManager:
    def __init__(self):
        self.base_url = "http://localhost:10000"
        self.model = "mistral-nemo"  # or any other model you have in Ollama
        self.prompt_template = self._load_prompt_template()
        
    def _load_prompt_template(self) -> str:
        """Load the prompt template from file."""
        prompt_path = os.path.join(os.path.dirname(__file__), 'v1.prompt')
        with open(prompt_path, 'r') as f:
            return f.read()
        
    def build_prompt(self, query_text: str, similar_texts: List[Dict]) -> str:
        """Build a prompt for the LLM using the query and similar texts."""
        similar_texts_section = ""
        for i, text in enumerate(similar_texts, 1):
            similar_texts_section += f"\n{i}. Text: {text.get('original_text', 'N/A')[:200]}..."
            similar_texts_section += f"\nTechniques: {', '.join(text['techniques'])}"
            similar_texts_section += f"\nManipulative: {text['manipulative']}"
            similar_texts_section += f"\nSimilarity score: {text['similarity_score']:.4f}\n"

        print(similar_texts_section)

        print(query_text)

        return self.prompt_template.format(
            query_text=query_text,
            similar_texts_section=similar_texts_section
        )

    async def get_analysis(self, prompt: str) -> Dict:
        """Get analysis from the LLM using the provided prompt."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert in analyzing manipulation techniques in text."},
                        {"role": "user", "content": prompt}
                    ],
                    "format": "json",
                    "stream": False
                }
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['message']['content']
            
            # Try to parse the content as JSON
            try:
                content_json = json.loads(content)
                analysis = content_json
            except json.JSONDecodeError:
                # If parsing fails, return the raw content
                analysis = content
            
            return {
                "analysis": analysis,
                "model": self.model
            }
            
        except Exception as e:
            logging.error(f"Error getting LLM analysis: {e}")
            raise

    def format_analysis(self, analysis_result: Dict) -> str:
        """Format the analysis result for presentation."""
        analysis = analysis_result['analysis']
        
        if isinstance(analysis, str):
            # Try to parse if it's a string
            try:
                analysis_obj = json.loads(analysis)
                formatted_analysis = json.dumps(analysis_obj, indent=2)
            except json.JSONDecodeError:
                formatted_analysis = analysis
        else:
            # It's already a dict or list
            formatted_analysis = json.dumps(analysis, indent=2)
        
        return f"""
Analysis Results:
----------------
{formatted_analysis}

Model: {analysis_result['model']}
""" 