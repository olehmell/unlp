import json
import csv
import sys
import os

def main():
    # Define the expected columns from submission.csv
    expected_columns = [
        "id", "straw_man", "appeal_to_fear", "fud", "bandwagon", 
        "whataboutism", "loaded_language", "glittering_generalities", 
        "euphoria", "cherry_picking", "cliche"
    ]
    
    # Input and output paths
    jsonl_file = "data/json/results-non-finetune.jsonl"
    output_csv = "data/unlp-2025-shared-task-classification-techniques/submission.csv"
    
    # Load the JSONL data
    data = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Create and write to CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(expected_columns)
        
        # Track any unknown techniques found
        unknown_techniques = set()
        
        # Process each entry from the JSONL file
        for entry in data:
            row = [entry["id"]]
            
            # Check for each expected technique
            for technique in expected_columns[1:]:  # Skip "id"
                if technique in entry.get("techniques", []):
                    row.append(1)
                else:
                    row.append(0)
            
            # Track any techniques that aren't in the expected columns
            for technique in entry.get("techniques", []):
                if technique not in expected_columns:
                    unknown_techniques.add(technique)
            
            writer.writerow(row)
    
    # Print warnings for any unknown techniques
    if unknown_techniques:
        print("WARNING: The following techniques were found in the JSONL but are not in the expected columns:")
        for technique in sorted(unknown_techniques):
            print(f"- {technique}")
        print(f"Total unknown techniques: {len(unknown_techniques)}")
    
    print(f"Successfully converted {len(data)} entries from {jsonl_file} to {output_csv}")

if __name__ == "__main__":
    main()