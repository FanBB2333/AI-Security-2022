"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter

from model import Model
from pgd_attack import LinfPGDAttack
from pgd_attack import device
from dataset import get_MNIST_loader

summary_writer = SummaryWriter('eval_logs')

# Global constants
with open('config.json') as config_file:
    config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']

model_dir = config['model_dir']

train_loader, test_loader = get_MNIST_loader()

model = Model().to(device)
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])



# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

last_checkpoint_filename = ''
already_seen_state = False


# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename = None):
    # Restore the checkpoint

    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_nat = 0.
    total_xent_adv = 0.
    total_corr_nat = 0
    total_corr_adv = 0
    if filename != None:
        model.load_from_checkpoint(filename)
    # model.eval()
    # with torch.no_grad():
    for ibatch, batch_data in enumerate(tqdm(test_loader)):
        x_batch, y_batch = batch_data
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch_adv, y_batch_adv = attack.perturb(x_batch, y_batch)

        loss_nat, num_correct_nat, accuracy_nat = model(x_batch, y_batch)
        loss_adv, num_correct_adv, accuracy_adv = model(x_batch_adv, y_batch_adv)

        total_xent_nat += loss_nat.sum()
        total_xent_adv += loss_adv.sum()
        total_corr_nat += num_correct_nat
        total_corr_adv += num_correct_adv


    avg_xent_nat = total_xent_nat / num_eval_examples
    avg_xent_adv = total_xent_adv / num_eval_examples
    acc_nat = total_corr_nat / num_eval_examples
    acc_adv = total_corr_adv / num_eval_examples

    # summary = tf.Summary(value=[
    #     tf.Summary.Value(tag='xent adv eval', simple_value=avg_xent_adv),
    #     tf.Summary.Value(tag='xent adv', simple_value=avg_xent_adv),
    #     tf.Summary.Value(tag='xent nat', simple_value=avg_xent_nat),
    #     tf.Summary.Value(tag='accuracy adv eval', simple_value=acc_adv),
    #     tf.Summary.Value(tag='accuracy adv', simple_value=acc_adv),
    #     tf.Summary.Value(tag='accuracy nat', simple_value=acc_nat)])
    # summary_writer.add_summary(summary, global_step.eval(sess))

    print('natural: {:.2f}%'.format(100 * acc_nat))
    print('adversarial: {:.2f}%'.format(100 * acc_adv))
    print('avg nat loss: {:.4f}'.format(avg_xent_nat))
    print('avg adv loss: {:.4f}'.format(avg_xent_adv))



if __name__ == '__main__':
    evaluate_checkpoint()



# Infinite eval loop
# while True:
#     cur_checkpoint = tf.train.latest_checkpoint(model_dir)
#
#     # Case 1: No checkpoint yet
#     if cur_checkpoint is None:
#         if not already_seen_state:
#             print('No checkpoint yet, waiting ...', end='')
#             already_seen_state = True
#         else:
#             print('.', end='')
#         sys.stdout.flush()
#         time.sleep(10)
#     # Case 2: Previously unseen checkpoint
#     elif cur_checkpoint != last_checkpoint_filename:
#         print('\nCheckpoint {}, evaluating ...   ({})'.format(cur_checkpoint,
#                                                               datetime.now()))
#         sys.stdout.flush()
#         last_checkpoint_filename = cur_checkpoint
#         already_seen_state = False
#         evaluate_checkpoint(cur_checkpoint)
#     # Case 3: Previously evaluated checkpoint
#     else:
#         if not already_seen_state:
#             print('Waiting for the next checkpoint ...   ({})   '.format(
#                 datetime.now()),
#                 end='')
#             already_seen_state = True
#         else:
#             print('.', end='')
#         sys.stdout.flush()
#         time.sleep(10)
