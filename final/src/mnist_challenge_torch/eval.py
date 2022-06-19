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

from model_bare import Model
from pgd_attack import LinfPGDAttack
from pgd_attack import device
from utils import get_MNIST_loader
from utils import checkpoint_callback

# summary_writer = SummaryWriter('eval_logs')

# Global constants
with open('config.json') as config_file:
    config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']

model_dir = config['model_dir']




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
    # num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_nat = 0.
    total_xent_adv = 0.
    total_corr_nat = 0
    total_corr_adv = 0

    train_loader, test_loader = get_MNIST_loader()

    model = Model().to(device)
    attack = LinfPGDAttack(model,
                           config['epsilon'],
                           config['k'],
                           config['a'],
                           config['random_start'],
                           config['loss_func'])
    if filename != None:
        print(f'Loading model from checkpoint {filename}')
        model.load_state_dict(torch.load(filename))
        model = model.to(device)

    num_eval_examples = 0
    # model.eval()
    # with torch.no_grad():
    for ibatch, batch_data in enumerate(tqdm(test_loader)):
        torch.cuda.empty_cache()
        x_batch, y_batch = batch_data
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch_adv, y_batch_adv = attack.perturb(x_batch, y_batch)

        loss_nat, num_correct_nat, accuracy_nat = model(x_batch, y_batch)
        loss_adv, num_correct_adv, accuracy_adv = model(x_batch_adv, y_batch_adv)

        total_xent_nat += loss_nat.sum()
        total_xent_adv += loss_adv.sum()
        total_corr_nat += num_correct_nat
        total_corr_adv += num_correct_adv
        num_eval_examples += int(y_batch.shape[0])

        # print(f'corr: {num_correct_nat}, all:{int(y_batch.shape[0])}, acc: {accuracy_nat}')


    avg_xent_nat = total_xent_nat / num_eval_examples
    avg_xent_adv = total_xent_adv / num_eval_examples
    acc_nat = total_corr_nat / num_eval_examples
    acc_adv = total_corr_adv / num_eval_examples


    print('natural: {:.2f}%'.format(100 * acc_nat))
    print('adversarial: {:.2f}%'.format(100 * acc_adv))
    print('avg nat loss: {:.4f}'.format(avg_xent_nat))
    print('avg adv loss: {:.4f}'.format(avg_xent_adv))



if __name__ == '__main__':
    evaluate_checkpoint('./checkpoints/model_bare.pt')

