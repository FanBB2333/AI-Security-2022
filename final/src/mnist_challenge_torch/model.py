from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # first convolutional layer
        self.conv_l1 = nn.Conv2d(in_channels=1,
                                 out_channels=32,
                                 kernel_size=(5, 5),
                                 stride=(1, 1),
                                 padding=2,
                                 bias=True,
                                 padding_mode='replicate')
        self.pool_l1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # second convolutional layer

        self.conv_l2 = nn.Conv2d(in_channels=32,
                                 out_channels=64,
                                 kernel_size=(5, 5),
                                 stride=(1, 1),
                                 padding=2,
                                 bias=True,
                                 padding_mode='replicate')
        self.pool_l2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # first fully connected layer
        self.fc1 = nn.Linear(in_features=7 * 7 * 64, out_features=1024, bias=True)

        # output layer
        self.fc2 = nn.Linear(in_features=1024, out_features=10, bias=True)


    def forward(self, x_input, y_input):
        # x_image = torch.reshape(x_input, [-1, 28, 28, 1])
        # x_image = x_input.view(-1, 28, 28, 1)
        x_image = x_input.view(-1, 1, 28, 28)
        h_conv1 = F.relu(self.conv_l1(x_image))
        h_pool1 = self.pool_l1(h_conv1)
        h_conv2 = F.relu(self.conv_l2(h_pool1))
        h_pool2 = self.pool_l2(h_conv2)
        h_pool2_flat = h_pool2.view(-1, 7 * 7 * 64)
        h_fc1 = F.relu(self.fc1(h_pool2_flat))
        h_fc2 = F.relu(self.fc2(h_fc1))
        # softmax_output = nn.functional.softmax(h_fc2)
        # loss = self.loss_fn(h_fc2, y_input)
        loss = F.cross_entropy(h_fc2, y_input)

        xent = loss.sum()

        y_pred = torch.argmax(h_fc2, 1)
        correct_prediction = torch.equal(y_pred, y_input)
        # num_correct = torch.sum(correct_prediction.to(torch.int64))
        # accuracy = torch.sum(correct_prediction.to(torch.float32))
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.forward(x, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
