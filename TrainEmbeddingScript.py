from models.V2_RandomNoiseError import ClipECA
from Datasets import Image2Prompt
from clip import clip
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from utils.utils import single_model_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ClipECA(warmup_steps=100, num_epochs=1000)
dataset = Image2Prompt(preprocess_image=model.processor)

train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
epochs = 2


class TrainEmbeddings:
    def __init__(self, model: ClipECA, dataloader, version='v2.pt'):
        self.save_path = 'models/trained/image2prompt/'
        self.version = version

        self.device = model.device

        self.model = model
        self.dataloader = dataloader

        self.tbar = None

        self.underscore = False
        self.dec_loss = 0.2
        self.n_steps = 0
        self.best = 0.3

    def run_per_epoch(self, embeddings, image):
        self.model.optimizer.zero_grad()

        embeddings = embeddings.to(self.model.device)
        image = image.to(self.model.device)

        outputs = self.model(image)
        gt = torch.ones(len(embeddings)).to(self.model.device)

        loss = self.model.criterion(outputs, embeddings, gt)
        accuracy = self.model.accurate_metric(outputs, embeddings).mean()

        # rand_lossItems = torch.Tensor([loss.item(), accuracy.item()]).to(self.model.device)

        # rand_loss = self.model.random_noise(rand_lossItems)

        # rand_lossItem = rand_loss.clone().detach().requires_grad_(True)

        loss.backward()

        self.model.vision = self.model.vision.float()
        self.model.optimizer.step()
        self.model.scheduler.step()
        self.model.convert_clip_weights()

        # return loss, accuracy, rand_loss
        return loss, accuracy

    def run_test(self):
        test_sim = single_model_test(self.model, self.model.processor)

        if test_sim > self.best:
            torch.save(self.model.state_dict(), self.save_path + self.version)
            torch.save(self.model.state_dict(), self.save_path + self.version)

            self.best = test_sim

        return test_sim

    def train(self):
        if self.tbar is None:
            self.tbar = tqdm(desc='Model train details:', total=len(self.dataloader))

        for i, (embeddings, image) in enumerate(self.dataloader):
            loss, accuracy = self.run_per_epoch(embeddings, image)

            test_sim = self.run_test()

            self.tbar.update(1)
            self.tbar.set_postfix_str(f"Loss:{loss.item()} Accuracy:{accuracy.item()} TestSimilarity:{test_sim.item()}")


model = TrainEmbeddings(
    model=model,
    dataloader=train_loader,
    version='better_data-embeddingalone.pt'
)

for epoch in range(epochs):
    model.train()
