# ==============================================================================

# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order
# ==============================================================================
from __future__ import print_function

import tensorflow as tf

tf.app.flags.DEFINE_string('data_dir', '', 'Data directory')
tf.app.flags.DEFINE_string('log_dir', '', 'Log directory')
tf.app.flags.DEFINE_string('job_name', '', 'Job name')
tf.app.flags.DEFINE_string('worker_hosts', '', 'workers')
tf.app.flags.DEFINE_string('ps_hosts', '', 'workers')
tf.app.flags.DEFINE_integer('task_index',0, 'workers')
FLAGS = tf.app.flags.FLAGS


def main(_):
    print('******************')
    print('Data directory path: %s' % FLAGS.data_dir)
    print('Log directory path: %s' % FLAGS.log_dir)
    print('Job type: %s' % FLAGS.job_name)
    print('Workers: %s' % FLAGS.worker_hosts)
    print('Parameter servers: %s' % FLAGS.ps_hosts)
    print('Task index: %d' % FLAGS.task_index)
    print('******************')
    # Creates a graph.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(c))


if __name__ == '__main__':
    tf.app.run()
