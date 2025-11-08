import pandas as pd

# 1. Read the Parquet file
df = pd.read_parquet('/home/harsh/AI/project/Sa2VA/data/Ref-coco/test-00000-of-00002.parquet')

# 2. Write it to a CSV file
# Assuming your DataFrame is named 'df'
# After: Access the columns directly
first_row = df.iloc[0]

print("--- Full Question ---")
print(first_row['question'])

print("\n--- Full Answer ---")
print(first_row['answer'])