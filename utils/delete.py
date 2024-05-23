import shutil
import os
from tqdm import tqdm

path = "D:/MachineLearning/KaggleCompetitions/Stable Diffusion Challenge/datasets/train_dataset/eval_images"
destination = "F:/DiffusionDB/images/"

files = os.listdir(path)
tbar = tqdm(desc='Moving files', total=len(files))

for file in files:
    file_path = f"{path}/{file}"
    shutil.move(file_path, destination)
    tbar.update(1)
tbar.close()

