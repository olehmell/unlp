import json

# Load the existing dataset (reading JSONL properly)
data = []
with open('data/json/train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # Skip empty lines
            data.append(json.loads(line))

# Open the output file
with open('data/json/fine_tuning_dataset.jsonl', 'w', encoding='utf-8') as f:
    for entry in data:
        # Create the messages array format required by OpenAI fine-tuning API
        messages = [
            {"role": "user", "content": f"Analyze the following post and list any manipulation techniques used:\n\nPost: {entry['content']}"},
            {"role": "assistant", "content": str(entry['manipulative']).lower() if entry['manipulative'] else "[]"}
        ]
        
        # Create the JSON object in the correct format
        json_line = json.dumps(messages, ensure_ascii=False)
        
        # Write to file
        f.write(json_line + '\n')