import os
import pandas as pd

df = pd.read_csv('data.csv')

print("Original Data:")
print(df.head())

df.columns = df.columns.str.strip()
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
df = df.map(lambda x: x.lower() if isinstance(x, str) else x)

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

workspace = os.getenv('GITHUB_WORKSPACE')
output_path = os.path.join(workspace, 'ModelCleaning', 'cleaned_data.csv')

df.to_csv(output_path, index=False)
print("\nCleaned data saved to:", output_path)
