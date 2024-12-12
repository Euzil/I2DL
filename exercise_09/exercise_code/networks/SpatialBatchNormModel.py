import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np


class AbstractNetwork(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def general_step(self, batch, batch_idx, mode):
        images, targets = batch

        # load X, y to device!
        images, targets = images.to(self.device), targets.to(self.device)

        # forward pass
        out = self.forward(images)

        # loss
        loss = F.cross_entropy(out, targets)

        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        return loss, n_correct

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, _ = self.general_step(batch, batch_idx, "test")
        return loss
    
    def prepare_data(self):

        # create dataset
        fashion_mnist_train = torchvision.datasets.FashionMNIST(
            root='../datasets', train=True, transform=transforms.ToTensor(), download=True)

        fashion_mnist_test = torchvision.datasets.FashionMNIST(
            root='../datasets', train=False, transform=transforms.ToTensor())

        torch.manual_seed(0)
        N = len(fashion_mnist_train)
        fashion_mnist_train, fashion_mnist_val = torch.utils.data.random_split(
            fashion_mnist_train, [int(N * 0.8), int(N * 0.2)])
        torch.manual_seed(torch.initial_seed())

        # set dataloaders
        torch.manual_seed(torch.initial_seed())
        train_dl = DataLoader(fashion_mnist_train, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(fashion_mnist_val, batch_size=self.batch_size)
        test_dl = DataLoader(fashion_mnist_test, batch_size=self.batch_size)

        return train_dl, val_dl, test_dl

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        return optim

    def getTestAcc(self, loader=None):
        if not loader:
            loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X, y = X.to(self.device), y.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc


class SimpleNetwork(AbstractNetwork):
    def __init__(
            self,
            batch_size,
            learning_rate,
            num_classes=10):
        super().__init__()

        # set hyperparams
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Linear(32*7*7, num_classes)

    def forward(self, x):

        # x.shape = [batch_size, 1, 28, 28]
        # load to device!
        x = x.to(self.device)

        # feed x into model!
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


class SpatialBatchNormNetwork(AbstractNetwork):

    def __init__(
            self,
            batch_size,
            learning_rate,
            num_classes=10):
        super().__init__()
        # set hyperparams
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        # x.shape = [batch_size, 1, 28, 28]
        # load to device!
        x = x.to(self.device)

        # feed x into model!
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x
