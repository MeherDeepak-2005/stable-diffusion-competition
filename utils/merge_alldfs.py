import os
import pandas as pd
from tqdm import tqdm

# json_file_path = "F:/DiffusionDB/jsonFiles"
# list_dirs = os.listdir(json_file_path)
#
# list_dirs = [f"{json_file_path}/{file}" for file in list_dirs]
#
# dataframes = []
#
# for list_dir in list_dirs:
#     df = pd.read_json(list_dir)
#     dataframes.append(df)
#
#
# df = pd.concat(dataframes, axis=1)
# print(df.head())
#
# df.to_csv('E:/DiffusionDB/train.csv')
# df.to_csv('F:/DiffusionDB/train.csv')
#
path1 = '../dataset/train.csv'
path2 = '../dataset/eval.csv'

train_df = pd.read_csv(path1)
eval_df = pd.read_csv(path2)

tbar = tqdm(desc="Change Path for SD", total=len(train_df) + len(eval_df))


def change_imagePath(image_path):
    image_path = image_path.split('/')[-1]
    image_path = f"E:/DiffusionDB/images/{image_path}"
    tbar.update(1)
    return image_path


def eval_path(image_path):
    image_path = image_path.split('/')[-1]
    image_path = f"F:/DiffusionDB/{image_path}"
    tbar.update(1)
    return image_path


tbar.close()

train_df['image_path'] = train_df['image_path'].apply(change_imagePath)
eval_df['image_path'] = eval_df['image_path'].apply(eval_path)

train_df.to_csv(path1)
eval_df.to_csv(path2)

f_files = os.listdir('F:/DiffusionDB/images/')
e_files = os.listdir('E:/DiffusionDB/images/')

original_df = pd.read_csv("F:/DiffusionDB/train.csv")
print("Original DF before:", len(original_df))
tbar = tqdm("Original DF", total=len(original_df))
filters = []
paths = []
for image_path in original_df['image_path'].to_list():
    if image_path in f_files:
        image_path = f"F:/DiffusionDB/images/{image_path}"
        validate = 1

    if image_path in e_files:
        image_path = f"E:/DiffusionDB/images/{image_path}"
        validate = 1

    else:
        validate = 0

    filters.append(validate)
    paths.append(image_path)
    tbar.update(1)

original_df['image_path'] = paths
original_df['Validate'] = filters

indexes = original_df[original_df['Validate'] == 0].index
original_df.drop(indexes, inplace=True)
print("Original DF after", len(original_df))

final_df = pd.concat([original_df, eval_df, train_df], axis=0)
print(final_df.head())
print(len(final_df))

final_df.to_csv('E:/DiffusionDB/train.csv')
