import pandas as pd
import os
from tqdm import tqdm

path = 'E:/DiffusionDB/train.csv'
df = pd.read_csv(path)
df.drop(columns=['Filter'], inplace=True)
df.set_index('Index', inplace=True)
df.to_csv('E:/DiffusionDB/backup.csv')
tbar = tqdm(desc='Verifying images', total=len(df))

e_files = os.listdir('E:/DiffusionDB/images')
f_files = os.listdir("F:/DiffusionDB/images")

e_files = [f"E:/DiffusionDB/images/{file}" for file in e_files]
f_files = [f"F:/DiffusionDB/images/{file}" for file in f_files]

files = e_files + f_files


def function(image_path):
    tbar.update(1)
    if image_path in files:
        return True
    else:
        return False


df['Validation'] = df['image_path'].apply(function)
invalid_rows = df[df['Validation'] == False].index
df.drop(invalid_rows, inplace=True)
tbar.close()
df.to_csv('E:/DiffusionDB/backup2.csv')
