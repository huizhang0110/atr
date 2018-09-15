import tensorflow as tf


class Res():
    def __init__(self, leakyReLu=False, istrain=True):
        self.istraining = istrain
        self.leakyReLu = leakyReLu

    def conv3x3(self, inputs, filters, strides=1):
        return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(3, 3), padding="same", strides=strides)

    def BasicBlock(self, inputs, filters, strides=1, downsample=None):
        residual = inputs
        conv1 = self.conv3x3(inputs=inputs, filters=filters, strides=strides)
        bn1 = tf.layers.batch_normalization(
            conv1, training=self.istraining)
        relu1 = tf.nn.relu(bn1)
        conv2 = self.conv3x3(inputs=relu1, filters=filters, strides=strides)
        bn2 = tf.layers.batch_normalization(conv2, training=self.istraining) #change
        # bn2 = tf.layers.batch_normalization(conv2)
        if downsample != None:
            residual = downsample(inputs, filters)
        bn2 += residual
        relu3 = tf.nn.relu(bn2)
        return relu3

    def __call__(self, data):

        def downsample(inputs, filters):
            conv1 = tf.layers.conv2d(inputs, filters=filters, kernel_size=(
                1, 1), padding='same')
            bn1 = tf.layers.batch_normalization(
                conv1, training=self.istraining)
            return bn1
        # 32 / 3 x 3 / 1 / 1
        with tf.variable_scope('conv1_1'):
            conv1 = tf.layers.conv2d(inputs=data, filters=32, kernel_size=(
                3, 3), strides=1, padding="same")
        # 32 / 3 x 3 / 1 / 1
        with tf.variable_scope('conv1_2'):
            conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=(
                3, 3), strides=1, padding="same")

        # 2 x 2 / 1
        pool1 = tf.layers.max_pooling2d(
            inputs=conv2, pool_size=[2, 2], strides=2)
        for i in range(2):
            with tf.variable_scope('block1_%d' % (i)):
                down = None
                if i == 0:
                    down = downsample
                pool1 = self.BasicBlock(pool1, 128, downsample=down)

        with tf.variable_scope('block1_conv'):
            pool1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(
                3, 3), strides=1, padding="same")
        # 2 x 2 / 1
        pool2 = tf.layers.max_pooling2d(
            inputs=pool1, pool_size=[2, 2], strides=2)

        for i in range(2):
            with tf.variable_scope('block2_%d' % (i)):
                down = None
                if i == 0:
                    down = downsample
                pool2 = self.BasicBlock(pool2, 256, downsample=down)

        with tf.variable_scope('block2_conv'):
            pool2 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(
                3, 3), strides=1, padding="same")
        # 2 x 2 / 1
        pool3 = tf.layers.max_pooling2d(inputs=pool2, pool_size=[
                                        2, 2], strides=[2, 1], padding="valid")
        pool3 = tf.pad(pool3, [[0, 0], [0, 0], [2, 2], [0, 0]])
        for i in range(5):
            with tf.variable_scope('block3_%d' % (i)):
                down = None
                if i == 0:
                    down = downsample
                pool3 = self.BasicBlock(pool3, 512, downsample=down)
        with tf.variable_scope('block3_conv'):
            pool3 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(
                3, 3), strides=1, padding="same")
        for i in range(3):
            with tf.variable_scope('block4_%d' % (i)):
                pool3 = self.BasicBlock(pool3, 512)
        with tf.variable_scope('block4_conv'):
            pool3 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(
                2, 2), strides=[2, 1], padding="valid")
        with tf.variable_scope('head'):
            conv5 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(
                2, 2), strides=1, padding="valid", activation=None)

            conv5 = tf.layers.batch_normalization(conv5, training=self.istraining) #change
            # conv5 = tf.layers.batch_normalization(conv5)
            conv5 = tf.nn.relu(conv5)
        # print(conv5.shape)
        return conv5

