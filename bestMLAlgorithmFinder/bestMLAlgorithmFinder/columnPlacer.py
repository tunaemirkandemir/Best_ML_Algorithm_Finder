import os

import pandas as pd

# Replace your_dataset below with your csv files name.
file_path = 'your_dataset.csv'
df = pd.read_csv(file_path)

# Choose the column to move to the last position. Unless the src (Best MLAlgorithm finder) won't work.
column_to_move = 'your_column_name'


if column_to_move in df.columns:

    df = pd.concat([df.drop(column_to_move, axis=1), df[column_to_move]], axis=1)

    # This will be your next modified csv.
    base_file_name, file_extension = os.path.splitext(os.path.basename(file_path))
    modified_file_path = f"modified_{base_file_name}_dataset.csv"
    df.to_csv(modified_file_path, index=False)

    print(f"Column '{column_to_move}' moved to the last position. Modified dataset saved to '{modified_file_path}'.")
else:
    print(f"Column '{column_to_move}' not found in the dataset.")
