import json
import pandas as pd
import os

df_path = os.path.dirname(os.path.realpath(__file__))
df_file = open(os.path.join(df_path, "tech_soft_none.json"), encoding='utf-8')
data = json.load(df_file)

# file = open('tech_soft_none.json', encoding='utf-8')
# data = json.load(file)

df = pd.json_normalize(data['data'])
df.label = df.label.astype('category')
test_df = df.iloc[50:80, :]
