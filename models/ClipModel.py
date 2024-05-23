import torch
import torch.nn as nn
import torch.optim as optim
import clip


class ClipModel(nn.Module):
    def __init__(self, warmup_steps, num_epochs):
        super(ClipModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model, self.processor = clip.load('ViT-L/14@336px', jit=False, device=self.device)

        for name, child in self.clip_model.visual.named_children():
            for pn, p in child.named_parameters():
                if 'transformer.resblocks' in pn:
                    pn_list = pn.split('.')
                    number = pn_list[2]
                    number = int(number)

                    if number >= 0:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                if 'resblocks' in pn and not 'transformer.resblocks' in pn:
                    pn_list = pn.split('.')
                    number = pn_list[1]
                    number = int(number)

                    if number >= 0:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

        self.generator = nn.Linear(768, 384)
        self.to(self.device)

        self.criterion = nn.CosineEmbeddingLoss(reduction='mean')
        self.accuracy = nn.CosineSimilarity()

        self.optimizer = optim.AdamW(self.parameters(), lr=0.002, betas=(0.99, 0.995), eps=1e-8,
                                     weight_decay=0.02, amsgrad=False)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 1)

    def forward(self, x):
        # expects an image
        x = self.clip_model.encode_image(x)
        x = x.to(torch.float)

        embeddings = self.generator(x)

        return embeddings

    def convert_clip_weights(self):
        clip.model.convert_weights(self.clip_model)
        return
