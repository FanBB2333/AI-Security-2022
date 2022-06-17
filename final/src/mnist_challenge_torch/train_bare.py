
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
from pgd_attack_pl import device, LinfPGDAttack
from utils import setup_seed

with open('config.json') as config_file:
    config = json.load(config_file)


setup_seed(25)

train_loader, test_loader = get_MNIST_loader()
model = Model().to(device)
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

if __name__ == '__main__':

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epoch = config['epoch']
    print(f'training for {epoch} epoch')
    for e in range(epoch):
        total_iter = len(train_loader)
        with tqdm(total=total_iter, desc=f'epoch {e}') as pbar:
            for ibatch, batch_data in enumerate(train_loader):
                x_input, y_input = batch_data
                x_input, y_input = x_input.to(device), y_input.to(device)
                loss_nat, num_correct_nat, accuracy_nat = model(x_input, y_input)
                if config['adv_train']:
                    x_batch_adv, y_batch_adv = attack.perturb(x_input, y_input)
                    x_batch_adv, y_batch_adv = x_batch_adv.to(device), y_batch_adv.to(device)
                    loss_adv, num_correct_adv, accuracy_adv = model(x_batch_adv, y_batch_adv)
                    loss_adv.backward()

                loss_nat.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix(loss=loss_nat, nat_acc=accuracy_nat, adv_acc=accuracy_adv if config['adv_train'] else None)
                pbar.update(1)
                # print('loss:', loss, 'accuracy:', accuracy)

    print('saving model...')
    torch.save(model.state_dict(), './checkpoints/model_bare.pt')
