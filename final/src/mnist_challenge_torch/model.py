from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self):
        super.__init__()
        # first convolutional layer
        self.conv_l1 = nn.Conv2d(in_channels=1,
                                 out_channels=32,
                                 kernel_size=(5, 5),
                                 stride=(1, 1),
                                 padding=2,
                                 bias=True,
                                 padding_mode='replicate')
        self.pool_l1 = nn.MaxPool2d(kernel_size=(2, 2),
                                    stride=(2, 2))

        # second convolutional layer

        self.conv_l2 = nn.Conv2d(in_channels=32,
                                 out_channels=64,
                                 kernel_size=(5, 5),
                                 stride=(1, 1),
                                 padding=2,
                                 bias=True,
                                 padding_mode='replicate')
        self.pool_l2 = nn.MaxPool2d(kernel_size=(2, 2),
                                    stride=(2, 2))


        W_conv1 = self._weight_variable([5, 5, 1, 32])
        b_conv1 = self._bias_variable([32])

        h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
        h_pool1 = self._max_pool_2x2(h_conv1)

        # second convolutional layer
        W_conv2 = self._weight_variable([5, 5, 32, 64])
        b_conv2 = self._bias_variable([64])

        h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self._max_pool_2x2(h_conv2)

        # first fully connected layer
        W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self._bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # output layer
        W_fc2 = self._weight_variable([1024, 10])
        b_fc2 = self._bias_variable([10])

        self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2

        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_input, logits=self.pre_softmax)

        self.xent = tf.reduce_sum(y_xent)

        self.y_pred = tf.argmax(self.pre_softmax, 1)

        correct_prediction = tf.equal(self.y_pred, self.y_input)

        self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def forward(self, x_input, y_input):
        x_image = torch.reshape(x_input, [-1, 28, 28, 1])
        h_conv1 = F.relu(self.conv_l1(x_image))
        h_pool1 = self.pool_l1(h_conv1)

    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool_2x2(x):
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
