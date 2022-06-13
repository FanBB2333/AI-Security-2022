"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.autograd
from torch.autograd import Variable

from utils import get_MNIST_loader

import pytorch_lightning as pl
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinfPGDAttack:
    def __init__(self, model: pl.LightningModule, epsilon, k, a, random_start, loss_func):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start

        if loss_func == 'xent':
            pass
            # loss = model.xent
        elif loss_func == 'cw':
            pass
            # label_mask = tf.one_hot(model.y_input,
            #                         10,
            #                         on_value=1.0,
            #                         off_value=0.0,
            #                         dtype=tf.float32)
            # correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
            # wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax
            #                             - 1e4*label_mask, axis=1)
            # loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            # loss = model.xent

        # self.grad = tf.gradients(loss, model.x_input)[0]
        # self.grad = torch.autograd.grad(loss, model.x_input)[0]

    def perturb(self, x_nat, y):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            # x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = x_nat + torch.Tensor(x_nat.shape).uniform_(-self.epsilon, self.epsilon).to(device)
            x = torch.clamp(x, 0, 1)  # ensure valid pixel range
        else:
            x = x_nat.clone()

        for i in range(self.k):
            x = Variable(x.float(), requires_grad=True)
            # y = torch.tensor(y, dtype=torch.double)
            x = x.to(device)
            loss, num_correct, accuracy = self.model(x, y)
            # loss.requires_grad_(True)
            grad = torch.autograd.grad(loss, x)[0]
            # grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
            #                                       self.model.y_input: y})

            x1 = self.a * torch.sign(grad) + x
            x2 = torch.where(x1 > x_nat + self.epsilon, x_nat + self.epsilon, x1)
            x2 = torch.where(x2 < x_nat - self.epsilon, x_nat - self.epsilon, x2)
            # x2 = torch.clamp(x1, x_nat - self.epsilon, x_nat + self.epsilon)
            x3 = torch.clamp(x2, 0, 1)  # ensure valid pixel range

        return x3, y


if __name__ == '__main__':
    import json
    import sys
    import math
    from model_pl import Model_PL

    with open('config.json') as config_file:
        config = json.load(config_file)

    model = Model_PL()
    model = model.to(device)
    # TODO: Load model from checkpoint
    attack = LinfPGDAttack(model,
                           config['epsilon'],
                           config['k'],
                           config['a'],
                           config['random_start'],
                           config['loss_func'])

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = []  # adv accumulator

    train_loader, test_loader = get_MNIST_loader()
    print('Iterating over {} batches'.format(len(test_loader)))
    for ibatch, batch_data in enumerate(tqdm(test_loader)):
        x_batch, y_batch = batch_data
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch_adv, _ = attack.perturb(x_batch, y_batch)
        x_adv.append(x_batch_adv.cpu().detach().numpy())

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    # np.save(path, x_adv)
    print('Examples stored in {}'.format(path))
