"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import numpy as np
import optuna
import pytorch_lightning as pl
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, Dataset

from utils import get_MNIST_loader, checkpoint_callback, setup_seed
from model_pl import Model_PL
# from pgd_attack import LinfPGDAttack

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
# tf.set_random_seed(config['random_seed'])
#
# max_num_training_steps = config['max_num_training_steps']
# num_output_steps = config['num_output_steps']
# num_summary_steps = config['num_summary_steps']
# num_checkpoint_steps = config['num_checkpoint_steps']
#
# batch_size = config['training_batch_size']

# Setting up the data and the model


train_loader, test_loader = get_MNIST_loader()
model = Model_PL()


# # Set up adversary
# attack = LinfPGDAttack(model,
#                        config['epsilon'],
#                        config['k'],
#                        config['a'],
#                        config['random_start'],
#                        config['loss_func'])
#
# # Setting up the Tensorboard and checkpoint outputs
# model_dir = config['model_dir']
# if not os.path.exists(model_dir):
#   os.makedirs(model_dir)
#
# # We add accuracy and xent twice so we can easily make three types of
# # comparisons in Tensorboard:
# # - train vs eval (for a single run)
# # - train of different runs
# # - eval of different runs
#
# saver = tf.train.Saver(max_to_keep=3)
# tf.summary.scalar('accuracy adv train', model.accuracy)
# tf.summary.scalar('accuracy adv', model.accuracy)
# tf.summary.scalar('xent adv train', model.xent / batch_size)
# tf.summary.scalar('xent adv', model.xent / batch_size)
# tf.summary.image('images adv train', model.x_image)
# merged_summaries = tf.summary.merge_all()
#
# shutil.copy('config.json', model_dir)

if __name__ == "__main__":
    setup_seed(25)
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    
    trainer = pl.Trainer(max_epochs=200,
                         accelerator="gpu" if torch.cuda.is_available() else None,
                         # strategy="ddp" if torch.cuda.is_available() else None,
                         fast_dev_run=False,
                         gpus=-1 if torch.cuda.is_available() else 0,
                         val_check_interval=0.25,
                         callbacks=[checkpoint_callback],
                         )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)


