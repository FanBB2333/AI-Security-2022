from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch
import numpy as np
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_MNIST_set():
    train_data = dataset.MNIST(root="mnist",
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
    test_data = dataset.MNIST(root="mnist",
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)
    return train_data, test_data


def get_MNIST_loader():
    train_data, test_data = get_MNIST_set()
    train_loader = DataLoader(train_data, batch_size=512)
    test_loader = DataLoader(test_data, batch_size=128)
    return train_loader, test_loader


from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/",
                                    #   save_top_k=5,
                                    #   monitor="valid/accuracy",
                                    #   filename='checkpoint_{epoch:02d}',
                                      )