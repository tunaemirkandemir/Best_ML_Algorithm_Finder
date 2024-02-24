import os

import pandas as pd
# 'your_dataset.extension' with your local file
file_path = 'your_dataset.extension'
# extension should be json or xlsx or
base_file_name, file_extension = os.path.splitext(os.path.basename(file_path))
if file_extension == '.json':

    json_file_path = file_path
    df = pd.read_json(json_file_path)

    csv_file_path = f'{base_file_name}.csv'
    df.to_csv(csv_file_path, index=False)
else:
    xlsx_file_name = file_path
    df = pd.read_excel(xlsx_file_name)

    csv_file_path = f'{base_file_name}.csv'
    df.to_csv(csv_file_path, index=False)