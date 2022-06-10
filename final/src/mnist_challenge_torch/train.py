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
import pytorch_lightning as pl
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, Dataset

from final.src.mnist_challenge_torch.dataset import get_MNIST_loader
from model import Model
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
model = Model()


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
    trainer = pl.Trainer(max_epochs=100,
                         accelerator="gpu",
                         strategy="ddp",
                         fast_dev_run=False,
                         gpus=-1,
                         val_check_interval=0.25,
                         )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

# with tf.Session() as sess:
#   # Initialize the summary writer, global variables, and our time counter.
#   summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
#   sess.run(tf.global_variables_initializer())
#   training_time = 0.0
#
#   # Main training loop
#   for ii in range(max_num_training_steps):
#     x_batch, y_batch = mnist.train.next_batch(batch_size)
#
#     # Compute Adversarial Perturbations
#     start = timer()
#     x_batch_adv = attack.perturb(x_batch, y_batch, sess)
#     end = timer()
#     training_time += end - start
#
#     nat_dict = {model.x_input: x_batch,
#                 model.y_input: y_batch}
#
#     adv_dict = {model.x_input: x_batch_adv,
#                 model.y_input: y_batch}
#
#     # Output to stdout
#     if ii % num_output_steps == 0:
#       nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
#       adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
#       print('Step {}:    ({})'.format(ii, datetime.now()))
#       print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
#       print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
#       if ii != 0:
#         print('    {} examples per second'.format(
#             num_output_steps * batch_size / training_time))
#         training_time = 0.0
#     # Tensorboard summaries
#     if ii % num_summary_steps == 0:
#       summary = sess.run(merged_summaries, feed_dict=adv_dict)
#       summary_writer.add_summary(summary, global_step.eval(sess))
#
#     # Write a checkpoint
#     if ii % num_checkpoint_steps == 0:
#       saver.save(sess,
#                  os.path.join(model_dir, 'checkpoint'),
#                  global_step=global_step)
#
#     # Actual training step
#     start = timer()
#     sess.run(train_step, feed_dict=adv_dict)
#     end = timer()
#     training_time += end - start
