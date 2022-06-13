
import json
import os
import shutil
from timeit import default_timer as timer

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_bare import Model
from utils import get_MNIST_loader
from pgd_attack import device

with open('config.json') as config_file:
    config = json.load(config_file)



train_loader, test_loader = get_MNIST_loader()
model = Model().to(device)



if __name__ == '__main__':
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-5)
    epoch = 10
    for e in range(epoch):
        for ibatch, batch_data in enumerate((train_loader)):
            x_input, y_input = batch_data
            x_input, y_input = x_input.to(device), y_input.to(device)
            loss, num_correct, accuracy = model(x_input, y_input)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('loss:', loss, 'accuracy:', accuracy)
