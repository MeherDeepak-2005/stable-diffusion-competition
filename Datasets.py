import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import numpy as np
import pandas as pd
from PIL import Image
from torch import frombuffer
import clip
import os
import warnings


def deprecated_class(cls):
    """This is a decorator to mark a class as deprecated."""
    orig_init = cls.__init__

    def new_init(self, *args, **kwargs):
        warnings.warn(
            f"The '{cls.__name__}' class is deprecated.",
            category=DeprecationWarning,
            stacklevel=2
        )
        orig_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls


class ImagePromptDataset(Dataset):
    def __init__(self, preprocess_image, train=True):
        super(ImagePromptDataset, self).__init__()
        self.preprocess_image = preprocess_image
        if train:
            self.df = pd.read_csv('datasets/train_dataset/train.csv')
        else:
            self.df = pd.read_csv('datasets/train_dataset/eval.csv')

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_path = self.df['image_path'].iloc[idx]
        processed_image = self.preprocess_image(Image.open(image_path))

        prompt = self.df['Prompt'].iloc[idx]
        tokens = self.tokenizer.encode(prompt, padding="max_length", return_tensors='pt', max_length=60,
                                       truncation=True)

        return tokens.squeeze(0), processed_image


class Image2Prompt(Dataset):
    def __init__(self, preprocess_image):
        super(Image2Prompt, self).__init__()

        self.preprocess_image = preprocess_image
        self.df = pd.read_csv('E:/DiffusionDB/train.csv')
        self.embeddings = np.load('E:/DiffusionDB/train-embeddings.npz')['embeddings']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_path = self.df['image_path'].iloc[idx]
        if "F:/DiffusionDB/" in image_path:
            image_path = image_path.split('/')[-1]
            image_path = f"F:/DiffusionDB/images/{image_path}"
        processed_image = self.preprocess_image(Image.open(image_path))

        embeddings = self.embeddings[idx]

        return frombuffer(embeddings, dtype=torch.float), processed_image


@deprecated_class
class DataCleansing(Dataset):
    def __init__(self, preprocess_image, train=True):
        super(DataCleansing, self).__init__()

        self.preprocess_image = preprocess_image
        if train:
            self.df = pd.read_csv('datasets/train_dataset/train.csv')

        else:
            self.df = pd.read_csv('datasets/train_dataset/eval.csv')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_path = self.df['image_path'].iloc[idx]
        processed_image = self.preprocess_image(Image.open(image_path))

        prompt = self.df['Prompt'].iloc[idx]
        embeddings = clip.tokenize([prompt], truncate=True)

        return embeddings.squeeze(0), processed_image


@deprecated_class
class DiffusionDB(Dataset):
    def __init__(self, preprocess_image):
        super(DiffusionDB, self).__init__()

        self.path = 'E:/DiffusionDB/extracted'
        assert os.path.exists(self.path)

        self.images = []
        self.image_names = []
        self.json_dataframes = []

        subFolders = os.listdir(self.path)
        for folder in subFolders:
            new_path = os.path.join(self.path, folder)
            if new_path.endswith('.lock'):
                pass
            else:
                images = [os.path.join(new_path, file) for file in os.listdir(new_path)]
                image_names = [file for file in os.listdir(new_path) if not file.endswith('.json')]
                json_file = [image for image in images if image.endswith('.json')]
                if len(json_file) != 1:
                    pass
                else:
                    json_file = json_file[0]
                    images.pop(images.index(json_file))

                    self.images.extend(images)
                    self.image_names.extend(image_names)

                    df = pd.read_json(json_file)
                    self.json_dataframes.append(df)

        self.df = pd.concat(self.json_dataframes, axis=1)
        # self.df = self.df[self.image_names]
        # self.df.columns = list(range(len(self.images)))
        self.preprocess_image = preprocess_image

    def __len__(self):
        return len(self.df.columns)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)
        image = self.preprocess_image(image)

        prompt = self.df[idx].p
        prompt = clip.tokenize([prompt], truncate=True).squeeze(0)

        return prompt, image
