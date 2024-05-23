"""
Train Clip:
    Get Prompts, Images
    Encode: Prompts, Images
    Calculate Cosine Similarity loss
    Train the model until it reaches an accuracy of 90% or above
"""
import clip
import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

device = T.device("cuda" if T.cuda.is_available() else 'cpu')
print(f"Using {device} for training")
os.makedirs('../../models/trained/clipmodel/', exist_ok=True)

clip_model, preprocessor = clip.load('ViT-L/14', device=device, jit=False)
clip.model.convert_weights(clip_model)


class ClipTrainDataset(T.utils.data.Dataset):
    def __init__(self, train=True):
        super(ClipTrainDataset, self).__init__()
        if train:
            self.dataset = pd.read_csv("../../datasets/train_dataset/train.csv")
        else:
            self.dataset = pd.read_csv("../../datasets/train_dataset/eval.csv", nrows=1024)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset['image_path'].iloc[idx]
        prompt = self.dataset['Prompt'].iloc[idx]
        embeddings = clip.tokenize([prompt], truncate=True)

        embeddings = embeddings.squeeze(0).to(self.device)

        img = Image.open(image_path)
        processed_img = preprocessor(img).to(self.device)

        return embeddings.to(self.device), processed_img


train_dataset = ClipTrainDataset(train=True)
test_dataset = ClipTrainDataset(train=False)
batch_size = 16
TrainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
TestLoader = DataLoader(test_dataset, batch_size=batch_size)
optimizer = optim.AdamW(clip_model.parameters(), lr=1e-4)
criterion = nn.CosineEmbeddingLoss(reduction='mean')

epochs = 2
pbar = tqdm(desc='Clip model', total=len(TrainLoader) * epochs)

num_steps = len(TrainLoader) * epochs

warmup_ratio = 0.1
warmup_steps = int(num_steps * warmup_ratio)
scheduler = OneCycleLR(optimizer, max_lr=1e-4, total_steps=num_steps, pct_start=warmup_steps / num_steps)


def evaluate(model, dataloader):
    # Evaluate the model
    cosine_similarities = []

    with torch.no_grad():
        # model.eval()

        for tokenized_text, preprocessed_image in dataloader:

            # Compute image and text embeddings
            image_features = model.encode_image(preprocessed_image)
            text_features = model.encode_text(tokenized_text)

            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
            cosine_similarities.append(similarity.mean().item())

    # Calculate the average cosine similarity
    avg_cosine_similarity = np.mean(cosine_similarities)
    print(f"\n Average cosine similarity: {avg_cosine_similarity}")
    return avg_cosine_similarity


min_eval_score = 0.6
for epoch in range(epochs):
    running_loss = 0
    for i, (text_embeddings, image) in enumerate(TrainLoader):
        optimizer.zero_grad()
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text_embeddings)

        target = T.ones(text_features.size(0)).to('cuda')
        loss = criterion(image_features, text_features, target)

        if loss.item() <= 1e-3 or loss.item() == 1.0:
            break
        loss.backward()

        running_loss += loss.item()

        clip_model = clip_model.float()
        optimizer.step()
        scheduler.step()

        clip.model.convert_weights(clip_model)
        pbar.update(1)
        pbar.set_postfix_str(f"loss: {loss.item()} average loss: {running_loss / (i + 1)}")

        if (i + 1) % 50 == 0:
            score = evaluate(clip_model, TestLoader)
            if min_eval_score < score:
                min_eval_score = score
                print('\n saving the best model')
                torch.save(clip_model.state_dict(), '../../models/trained/clipmodel/v1.pt')

    if loss.item() <= 1e-2 or loss.item() == 1.0:
        break

if min_eval_score < score:
    torch.save(clip_model.state_dict(), '../../models/trained/clipmodel/v1.pt')
