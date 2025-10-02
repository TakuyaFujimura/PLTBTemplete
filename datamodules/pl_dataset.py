from typing import Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, pin_memory:bool=False, num_workers:int=0, seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.seed = seed
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(self.seed)
        )
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers)
