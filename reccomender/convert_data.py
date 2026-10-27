# -*- coding: cp1251 -*-
import json
import os
import pandas as pd

json_file_path = 'C:/Users/diuli/source/repos/reccomender/reccomender/input_ratings.json'
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df['ratings'] = df['ratings'].replace(-1.0, 0.0)

df['user_id'] = df['user_id'].astype(str)
df['item_id'] = df['item_id'].astype(str)
df['rating'] = df['ratings'].astype(float) 
df['timestamp'] = 1.0 

df = df[['user_id', 'item_id', 'rating', 'timestamp']]
df.columns = ['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']

os.makedirs('C:/Users/diuli/source/repos/reccomender/reccomender/dataset/movies', exist_ok=True)

output_path = 'C:/Users/diuli/source/repos/reccomender/reccomender/dataset/movies/movies.inter'
df.to_csv(output_path, sep='\t', index=False)

print("Done")

