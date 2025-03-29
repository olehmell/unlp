import os
import pandas as pd
import uuid
import glob
import re
from typing import List, Dict, Optional

def identify_language(filename: str) -> str:
    """Identify language based on filename."""
    if 'uk' in filename:
        return 'uk'
    elif 'ru' in filename:
        return 'ru'
    else:
        # Default to Ukrainian if not specified
        return 'uk'

def get_language_from_content(text: str) -> str:
    """Attempt to detect language from content for cases where filename doesn't indicate it."""
    # Simple heuristic based on character frequency
    cyrillic_uk = set('їієґ')
    cyrillic_ru = set('ыэъё')
    
    uk_count = sum(1 for char in text.lower() if char in cyrillic_uk)
    ru_count = sum(1 for char in text.lower() if char in cyrillic_ru)
    
    if uk_count > ru_count:
        return 'uk'
    elif ru_count > uk_count:
        return 'ru'
    else:
        # Check character distribution for less obvious cases
        # Ukrainian uses 'і' more frequently than Russian
        if text.lower().count('і') > text.lower().count('ы'):
            return 'uk'
        else:
            return 'ru'

def extract_techniques(techniques_str: str) -> List[str]:
    """Extract techniques from string representation."""
    # Remove brackets, quotes and clean up the string
    if pd.isna(techniques_str):
        return []
    
    # Handle both string list format and actual list
    if isinstance(techniques_str, list):
        return techniques_str
    
    # Try to extract from string representation: ['tech1', 'tech2']
    techniques = re.findall(r"'([^']*)'", techniques_str)
    if techniques:
        return techniques
    
    # If simple string, return as single-item list
    return [techniques_str]

def create_synthetic_dataset(output_path: str):
    """Create a unified synthetic dataset from all available files."""
    # Find all synthetic data files
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    synthetic_files = glob.glob(os.path.join(data_dir, 'synthetic_data*.csv'))
    
    if not synthetic_files:
        raise FileNotFoundError(f"No synthetic data files found in {data_dir}")
    
    print(f"Found {len(synthetic_files)} synthetic data files")
    for file in synthetic_files:
        print(f"  - {os.path.basename(file)}")
    
    all_data = []
    
    for file_path in synthetic_files:
        file_name = os.path.basename(file_path)
        language = identify_language(file_name)
        
        try:
            df = pd.read_csv(file_path)
            print(f"Loading {file_name}: {len(df)} records")
            
            # Process each row
            for _, row in df.iterrows():
                # Clean content - remove extra quotes
                content = row['content']
                if isinstance(content, str):
                    content = content.strip('"')
                
                # Determine language if not clear from filename
                text_language = language
                if language == 'uk' and get_language_from_content(content) == 'ru':
                    text_language = 'ru'
                elif language == 'ru' and get_language_from_content(content) == 'uk':
                    text_language = 'uk'
                
                # Extract techniques
                techniques = extract_techniques(row['techniques'])
                
                # All synthetic examples are manipulative
                manipulative = True
                
                # Create a UUID for the entry
                entry_id = str(uuid.uuid4())
                
                # Add to dataset
                all_data.append({
                    'id': entry_id,
                    'content': content,
                    'lang': text_language,
                    'manipulative': manipulative,
                    'techniques': techniques,
                    'trigger_words': []  # Empty for synthetic data
                })
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    # Convert to DataFrame
    df_combined = pd.DataFrame(all_data)
    
    # Format techniques as string in the format Pandas will understand
    df_combined['techniques'] = df_combined['techniques'].apply(lambda x: str(x).replace("'", ""))
    
    # Save the combined dataset
    df_combined.to_csv(output_path, index=False)
    print(f"Created synthetic dataset with {len(df_combined)} records at {output_path}")
    
    # Print summary statistics
    print("\nLanguage Distribution:")
    print(df_combined['lang'].value_counts())
    
    print("\nTechniques Distribution:")
    all_techniques = []
    for techniques_list in df_combined['techniques']:
        techniques = extract_techniques(techniques_list)
        all_techniques.extend(techniques)
    
    technique_counts = {}
    for technique in all_techniques:
        technique_counts[technique] = technique_counts.get(technique, 0) + 1
    
    for technique, count in sorted(technique_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {technique}: {count}")

if __name__ == "__main__":
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'data',
        'synthetic_train_dataset.csv'
    )
    create_synthetic_dataset(output_path)
    print("Done!")
