from transformers import GPT2Model, get_cosine_schedule_with_warmup
import argparse
import clip
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm

learning_rate = 2e-5
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clip_model, preprocess_image = clip.load('./RN50.pt', device='cpu', jit=False)
clip_model.load_state_dict(torch.load('./prompt_trained_rn50.pt'))

parser = argparse.ArgumentParser(description='Train a GPT-2 model')

parser.add_argument('--gpt', type=str, default='gpt2',
                    help='The name of the GPT-2 model to use (default: "gpt2")')
parser.add_argument('--batch_size', type=int, default=1, help='GPT2 Model batch size.'
                                                              'Ideal range for normal PC ~ 8-10')
args = parser.parse_args()

model_name = args.gpt
batch_size = args.batch_size


# Freeze all the layers except the last one in transformer.h and lm_head layer
# for name, child in model_lm.named_children():
#     if 'lm_head' in name:
#         print("Found layer:", name)
#         for p in child.parameters():
#             p.requires_grad = True
#     elif name == 'transformer':
#         for sub_name, sub_child in child.named_children():
#             if 'h' in sub_name:
#                 for block_name, block_child in sub_child.named_children():
#                     if block_name == '4':  # Only unfreeze last layer
#                         print("Found layer:", sub_name, block_name)
#                         for p in block_child.parameters():
#                             p.requires_grad = True
#                     else:
#                         print("Freezing layer:", sub_name, block_name)
#                         for p in block_child.parameters():
#                             p.requires_grad = False
#             else:
#                 print("Freezing layer:", sub_name)
#                 for p in sub_child.parameters():
#                     p.requires_grad = False
#     else:
#         print("Freezing layer:", name)
#         for p in child.parameters():
#             p.requires_grad = False
class PromptGenerator(nn.Module):
    def __init__(self, embedding_size=384):
        super(PromptGenerator, self).__init__()
        self.embedding_size = embedding_size

        self.trim = nn.Linear(1024 * 768, 768)
        self.generator = nn.Linear(768, 384)
        self.gpt = GPT2Model.from_pretrained(model_name)

        self.att_mask = torch.ones

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = x.unsqueeze(1).repeat(1, 1, 1)
        x = x.to(torch.long)
        outs = self.gpt(input_ids=x, attention_mask=self.att_mask(x.size(0), x.size(1), dtype=torch.long)
                        .to(self.device))
        outs = outs.last_hidden_state.squeeze(0).squeeze(0)

        outs = outs.view(-1, outs.size(-2) * outs.size(-1))

        outs = self.trim(outs)
        outs = self.generator(outs)
        return outs


class ImagePromptDataset(Dataset):
    def __init__(self, train=True):
        if train:
            self.df = pd.read_csv('../../datasets/train_dataset/train.csv')
            self.embeddings = np.load('../../datasets/train_dataset/train-embeddings.npz')['embeddings']
        else:
            self.df = pd.read_csv('../../datasets/train_dataset/eval.csv')
            self.embeddings = np.load('../../datasets/train_dataset/eval-embeddings.npz')['embeddings']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_path = self.df['image_path'].iloc[idx]
        processed_image = preprocess_image(Image.open(image_path))

        embedding_array = self.embeddings[idx]

        return torch.frombuffer(embedding_array, dtype=torch.float), processed_image


train_data = ImagePromptDataset(train=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

print()

eval_data = ImagePromptDataset(train=False)
eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)

clip_model.to(device)
gpt2_model = PromptGenerator()

num_epochs = 1

cos_loss = torch.nn.CosineEmbeddingLoss(reduction='mean').to(device)
optimizer = optim.AdamW(gpt2_model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),
                        eps=epsilon)
scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.1 * len(train_loader),
                                            num_training_steps=1)

tbar = tqdm(desc='GPT Training:', total=num_epochs * len(train_loader))

for epoch in range(num_epochs):
    for batch_idx, (target_embeddings, image) in enumerate(train_loader):
        optimizer.zero_grad()
        target_embeddings = target_embeddings.to(device)
        image = image.to(device)
        gt = torch.ones((target_embeddings.size(0))).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image)

        predicted_embeddings = gpt2_model(image_features)

        loss = cos_loss(predicted_embeddings, target_embeddings, gt)

        loss.backward()
        optimizer.step()
        scheduler.step()
        tbar.update(batch_idx)
        tbar.set_postfix_str(f"Loss: {loss.item()}")
