import pandas as pd

df = pd.read_csv('F:/DiffusionDB/train.csv')
new_df = pd.DataFrame(columns=['Index', 'Prompt', 'image_path'])

new_df['Prompt'] = df.iloc[0, :].to_list()
new_df['image_path'] = df.columns

new_df['Index'] = list(range(len(df.columns)))
new_df.set_index('Index', inplace=True)

new_df = new_df.iloc[1:, :]

new_df.to_csv('E:/DiffusionDB/train.csv')
new_df.to_csv('F:/DiffusionDB/train.csv')
