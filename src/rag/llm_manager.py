from typing import List, Dict, Optional
import logging
import os
import json
import asyncio
from mistralai import Mistral
from pydantic import BaseModel, Field

class TextAnalysis(BaseModel):
    """Data model for text analysis output."""
    manipulation_techniques: List[str] = Field(description="List of identified manipulation techniques")

class LLMManager:
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable must be set")
        
        self.client = Mistral(api_key=self.api_key)
        self.model = "open-mistral-nemo"  # Using the MistralAI open-mistral-nemo model
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

        return self.prompt_template.format(
            query_text=query_text,
            similar_texts_section=similar_texts_section
        )

    async def get_analysis(self, prompt: str) -> Dict:
        """Get analysis from the LLM using the provided prompt."""
        try:
            # Use Mistral's parse method to get structured output
            response = await asyncio.to_thread(
                self.client.chat.parse,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in identifying manipulation techniques in text, specializing in multilabel classification for imbalanced datasets.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                response_format=TextAnalysis,
                temperature=0.1
            )
            
            # Extract the parsed result
            analysis = response.choices[0].message.parsed
            
            return {
                "analysis": analysis.dict(),
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