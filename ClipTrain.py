from models.ClipModel import ClipModel
import torch
from Datasets import Image2Prompt
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.utils import single_model_test

clip_model = ClipModel(num_epochs=2, warmup_steps=10)
dataset = Image2Prompt(clip_model.processor)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
tbar = tqdm('Training', total=len(train_loader))
best = 0.0
test_similarity = 0.0


def train(model: ClipModel, dataloader: train_loader, save_path: str):
    losses = np.zeros(len(train_loader))
    i = 0
    global best, test_similarity
    for embeddings, images in dataloader:
        model.optimizer.zero_grad()
        embeddings, image = embeddings.to(model.device), images.to(model.device)

        outputs = model(image)
        gt = torch.ones(len(embeddings)).to(model.device)

        loss = model.criterion(outputs, embeddings, gt)
        accuracy = model.accuracy(outputs, embeddings).mean()

        if accuracy.item() > best:
            torch.save(model.state_dict(), save_path)
            best = accuracy.item()

        losses[i] = loss.item()

        loss.backward()
        model.clip_model = model.clip_model.float()

        model.optimizer.step()
        model.scheduler.step()

        model.convert_clip_weights()
        tbar.update(1)
        tbar.set_postfix_str(f"Loss:{loss.item():.4f}, Accuracy:{accuracy.item():.5f} SavedModelAt:{best} "
                             f"Test:{test_similarity:.6f}")

        if (i + 1) % 300 == 0:
            test_similarity = single_model_test(model, model.processor).item()

        if (i + 1) % 100 == 0:
            torch.save(model.state_dict(), save_path)

        i = i + 1

    return losses


if __name__ == "__main__":
    epochs = 2
    # if os.path.exists('./models/trained/clipmodel/better_data.pt'):
    #     print("Loading Model from path")
    #     clip_model.load_state_dict(torch.load('./models/trained/clipmodel/better_data.pt'))
    for x in range(epochs):
        loss_data = train(clip_model, train_loader, save_path='./models/trained/clipmodel/better_data.pt')
        plt.title("Loss/Cosine vs Epoch/int")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(x=list(range(len(loss_data))), y=loss_data)
        plt.savefig(f'./train_logging/{x}.png')
