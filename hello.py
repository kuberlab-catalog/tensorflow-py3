# ==============================================================================

# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import tensorflow as tf

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'Number of training iterations')
tf.app.flags.DEFINE_integer('model_version', 1, 'Version number of the model')
tf.app.flags.DEFINE_string('data_dir', '/tmp/mnist/data', 'Data directory')
tf.app.flags.DEFINE_string('log_dir', 'models/mnist', 'Log directory')
FLAGS = tf.app.flags.FLAGS


def main(_):
    print('Hello, tensorflow! ')
    print('Python version: %s', sys.version)


if __name__ == '__main__':
    tf.app.run()
