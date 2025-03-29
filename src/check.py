import pandas as pd

# Load the CSV files
train_df = pd.read_csv('/Users/olehmell/projects/datasciense/unlp/data/csv/test.csv')
submission_df = pd.read_csv('/Users/olehmell/projects/datasciense/unlp/data/submission_gemma_15.csv')

print("train_df columns:", train_df.head(2))
print("submission_df columns:", submission_df.head(2))

# Extract the IDs from both files
train_ids = set(train_df['id'])
submission_ids = set(submission_df['id'])

# Find IDs in train.csv but not in submission_gemma_15.csv
missing_ids = train_ids - submission_ids

# Print the results
print(f"Number of IDs in train.csv: {len(train_ids)}")
print(f"Number of IDs in submission_gemma_15.csv: {len(submission_ids)}")
print(f"Number of missing IDs: {len(missing_ids)}")

# Print the actual missing IDs if there aren't too many
if len(missing_ids) < 100:
    print("Missing IDs:")
    for id in sorted(missing_ids):
        print(id)
else:
    print(f"There are {len(missing_ids)} missing IDs. First 10:")
    for id in sorted(list(missing_ids))[:10]:
        print(id)
    
    # Optionally save to a file if there are many missing IDs
    with open('/Users/olehmell/projects/datasciense/unlp/missing_ids.txt', 'w') as f:
        for id in sorted(missing_ids):
            f.write(f"{id}\n")
        print("All missing IDs have been saved to 'missing_ids.txt'")
