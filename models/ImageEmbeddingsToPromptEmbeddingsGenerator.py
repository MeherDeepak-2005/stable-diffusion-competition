"""
Converting Image Embeddings into Prompt Embeddings
"""
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup


class Image2PromptEmbeddingGenerator(nn.Module):
    def __init__(self):
        super(Image2PromptEmbeddingGenerator, self).__init__()

        self.conv = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)

        self.lstm = nn.LSTM(input_size=30, hidden_size=64, bidirectional=True)

        self.linear = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.GELU(),
            nn.Linear(4096, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 384)
        )

        self.gap = nn.AdaptiveAvgPool1d(2048)
        self.conv_attention = nn.Conv1d(2048, 128, kernel_size=1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
        self.criterion = nn.CosineEmbeddingLoss(reduction='mean')
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=100, num_training_steps=10000)

    def forward(self, x):

        x = x.to(torch.float)
        x = self.conv(x)
        lstm_x, (_, _) = self.lstm(x)
        x = lstm_x.view(-1, lstm_x.size(1) * lstm_x.size(2))

        # attention layer
        att_x = self.gap(x)
        # print("Gap output (256, 2048) - ", att_x.size())
        att_x = att_x.unsqueeze(2)

        att_x = self.conv_attention(att_x)
        # print('Conv att output (256, 128, 1) - ', att_x.size())
        att_x = att_x.permute(0, 2, 1)

        mul_x = torch.mul(lstm_x, att_x)
        # print('Multiplied Layer - (256, 64, 128)', mul_x.size())

        mul_x = mul_x.view(-1, mul_x.size(1) * mul_x.size(2))

        x = self.linear(mul_x)

        return x

    def loss(self, preds, targets, gt):
        self.optimizer.zero_grad()

        loss = self.criterion(preds, targets, gt)
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()
