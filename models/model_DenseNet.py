import tensorflow as tf
from tensorflow.contrib.layers.python.layers.layers import batch_norm
from tensorflow.contrib.layers.python.layers import xavier_initializer as xavier
from tensorflow.contrib.layers import variance_scaling_initializer

class convmodel():
    def __init__(self):
        growth = 16  # Growth Rate
        init = growth * 2  # Dense-BC
        growth4 = 4 * growth  # Dense-B
        blocksize = [6, 12, 24, 16]  # Dense-121

        with tf.variable_scope("model1"):
            self.w1 = tf.get_variable(name='w1', shape=[3, 3, 3, 1, init], initializer=xavier())

            self.w2 = []
            for i in range(0, blocksize[0]):
                self.w2.append(tf.get_variable(name='w2_1x1_' + str(i), shape=[1, 1, 1, init + i * growth, growth4],
                                               initializer=xavier()))
                self.w2.append(
                    tf.get_variable(name='w2_' + str(i), shape=[3, 3, 3, growth4, growth], initializer=xavier()))

            init1 = (init + blocksize[0] * growth) / 2
            self.w3_1x1 = tf.get_variable(name='w3_1x1', shape=[1, 1, 1, init + blocksize[0] * growth, init1],
                                          initializer=xavier())  # Dense-C

            self.w3 = []
            for i in range(0, blocksize[1]):
                self.w3.append(tf.get_variable(name='w3_1x1_' + str(i), shape=[1, 1, 1, init1 + i * growth, growth4],
                                               initializer=xavier()))
                self.w3.append(
                    tf.get_variable(name='w3_' + str(i), shape=[3, 3, 3, growth4, growth], initializer=xavier()))

            init2 = (init1 + blocksize[1] * growth) / 2
            self.w4_1x1 = tf.get_variable(name='w4_1x1', shape=[1, 1, 1, init1 + blocksize[1] * growth, init2],
                                          initializer=xavier())

            self.w4 = []
            for i in range(0, blocksize[2]):
                self.w4.append(tf.get_variable(name='w4_1x1_' + str(i), shape=[1, 1, 1, init2 + i * growth, growth4],
                                               initializer=xavier()))
                self.w4.append(
                    tf.get_variable(name='w4_' + str(i), shape=[3, 3, 3, growth4, growth], initializer=xavier()))

            init3 = (init2 + blocksize[2] * growth) / 2
            self.w5_1x1 = tf.get_variable(name='w5_1x1', shape=[1, 1, 1, init2 + blocksize[2] * growth, init3],
                                          initializer=xavier())

            self.w5 = []
            for i in range(0, blocksize[3]):
                self.w5.append(tf.get_variable(name='w5_1x1_' + str(i), shape=[1, 1, 1, init3 + i * growth, growth4],
                                               initializer=xavier()))
                self.w5.append(
                    tf.get_variable(name='w5_' + str(i), shape=[3, 3, 3, growth4, growth], initializer=xavier()))

            self.fc1 = tf.get_variable(name='fc1', shape=[1, 1, 1, init3 + blocksize[3] * growth, 2],
                                       initializer=xavier())
            self.fc1b = tf.get_variable(name='fc1b', shape=[2], initializer=xavier())

    def _denseblock(self, input, kernel1, kernel2, layer_name, is_training):

        with tf.name_scope(layer_name):
            c = batch_norm(input, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name + '1')
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel1, strides=(1, 1, 1, 1, 1), padding='SAME')
            c = batch_norm(c, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name + '2')
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel2, strides=(1, 1, 1, 1, 1), padding='SAME')

            input = tf.concat([input, c], axis=4)

        return input

    def _add_transition(self, c, kernel, layer_name, is_training):

        with tf.name_scope(layer_name):
            c = batch_norm(c, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name)
            c = tf.nn.relu(c)
        with tf.variable_scope("model1"):
            c = tf.nn.conv3d(c, kernel, strides=(1, 1, 1, 1, 1), padding='SAME')
            c = tf.nn.avg_pool3d(c, (1, 2, 2, 2, 1), strides=(1, 2, 2, 2, 1), padding='SAME')

        return c

    def convnet(self):

        image = tf.placeholder("float32", [None, 48, 32, 48, 1])
        id = tf.placeholder("int32", [None, 1])
        is_training = tf.placeholder(tf.bool, shape=())
        expand_dims = tf.placeholder(tf.bool, shape=())

        conv = tf.nn.conv3d(image, self.w1, strides=(1, 2, 2, 2, 1), padding='SAME')

        for i in range(0, len(self.w2), 2):
            conv = self._denseblock(conv, self.w2[i], self.w2[i + 1], 'conv2_' + str(i), is_training)

        conv = self._add_transition(conv, self.w3_1x1, 'conv3_1x1', is_training)

        for i in range(0, len(self.w3), 2):
            conv = self._denseblock(conv, self.w3[i], self.w3[i + 1], 'conv3_' + str(i), is_training)

        conv = self._add_transition(conv, self.w4_1x1, 'conv4_1x1', is_training)

        for i in range(0, len(self.w4), 2):
            conv = self._denseblock(conv, self.w4[i], self.w4[i + 1], 'conv4_' + str(i), is_training)

        conv = self._add_transition(conv, self.w5_1x1, 'conv5_1x1', is_training)

        for i in range(0, len(self.w5), 2):
            conv = self._denseblock(conv, self.w5[i], self.w5[i + 1], 'conv5_' + str(i), is_training)

        conv = batch_norm(conv, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                          is_training=is_training, scope='bn_last')
        conv = tf.nn.relu(conv)
        shape = conv.get_shape().as_list()
        conv = tf.nn.avg_pool3d(conv, (1, shape[1], shape[2], shape[3], 1), strides=(1, 1, 1, 1, 1), padding='VALID')

        conv = tf.nn.conv3d(conv, self.fc1, strides=(1, 1, 1, 1, 1), padding='VALID') + self.fc1b
        conv = tf.squeeze(conv)
        conv = tf.cond(expand_dims,lambda:tf.expand_dims(conv, axis=0),lambda:conv)

        onehot_labels = tf.one_hot(indices=id, depth=2, on_value=1.0, off_value=0.0)
        alpha = 0.1
        onehot_labels = tf.add((1 - alpha) * onehot_labels, alpha / 2)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=conv)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(cross_entropy)

        tf.add_to_collection('losses', loss)

        prediction = tf.equal(tf.expand_dims(tf.argmax(conv, 1), axis=1), tf.cast(id, tf.int64))
        with tf.name_scope('accuracy'):
            acc = tf.reduce_mean(tf.cast(prediction, tf.float32))

        return image, id, loss, acc, is_training, expand_dims, tf.nn.softmax(conv, dim=-1), cross_entropy


    def convcam(self):

        # image = tf.placeholder("float32", [None, 48, 32, 48, 1])
        image = tf.placeholder("float32", [None, 48, 48, 48, 1])
        id = tf.placeholder("int32", [None, 1])

        is_training = tf.placeholder(tf.bool, shape=())

        conv = tf.nn.conv3d(image, self.w1, strides=(1, 1, 1, 1, 1), padding='SAME')

        for i in range(0, len(self.w2), 2):
            conv = self._denseblock(conv, self.w2[i], self.w2[i + 1], 'conv2_' + str(i), is_training)

        conv = self._add_transition(conv, self.w3_1x1, 'conv3_1x1', is_training)
        # cam = conv

        for i in range(0, len(self.w3), 2):
            conv = self._denseblock(conv, self.w3[i], self.w3[i + 1], 'conv3_' + str(i), is_training)

        conv = self._add_transition(conv, self.w4_1x1, 'conv4_1x1', is_training)
        # cam = conv

        for i in range(0, len(self.w4), 2):
            conv = self._denseblock(conv, self.w4[i], self.w4[i + 1], 'conv4_' + str(i), is_training)

        conv = self._add_transition(conv, self.w5_1x1, 'conv5_1x1', is_training)

        cam = conv

        for i in range(0, len(self.w5), 2):
            conv = self._denseblock(conv, self.w5[i], self.w5[i + 1], 'conv5_' + str(i), is_training)
        conv = batch_norm(conv, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                          is_training=is_training, scope='bn_last')
        conv = tf.nn.relu(conv)
        shape = conv.get_shape().as_list()
        conv = tf.nn.avg_pool3d(conv, (1, shape[1], shape[2], shape[3], 1), strides=(1, 1, 1, 1, 1), padding='VALID')

        conv = tf.nn.conv3d(conv, self.fc1, strides=(1, 1, 1, 1, 1), padding='VALID') + self.fc1b
        conv = tf.squeeze(conv)
        conv = tf.expand_dims(conv, axis=0)

        return image,id, is_training, tf.nn.softmax(conv, dim=-1), cam


    def convval(self,th):

        image = tf.placeholder("float32", [None, 48, 32, 48, 1])
        id = tf.placeholder("int32", [None, 1])
        is_training = tf.placeholder(tf.bool, shape=())
        expand_dims = tf.placeholder(tf.bool, shape=())

        conv = tf.nn.conv3d(image, self.w1, strides=(1, 2, 2, 2, 1), padding='SAME')

        for i in range(0, len(self.w2), 2):
            conv = self._denseblock(conv, self.w2[i], self.w2[i + 1], 'conv2_' + str(i), is_training)

        conv = self._add_transition(conv, self.w3_1x1, 'conv3_1x1', is_training)

        for i in range(0, len(self.w3), 2):
            conv = self._denseblock(conv, self.w3[i], self.w3[i + 1], 'conv3_' + str(i), is_training)

        conv = self._add_transition(conv, self.w4_1x1, 'conv4_1x1', is_training)

        for i in range(0, len(self.w4), 2):
            conv = self._denseblock(conv, self.w4[i], self.w4[i + 1], 'conv4_' + str(i), is_training)

        conv = self._add_transition(conv, self.w5_1x1, 'conv5_1x1', is_training)

        for i in range(0, len(self.w5), 2):
            conv = self._denseblock(conv, self.w5[i], self.w5[i + 1], 'conv5_' + str(i), is_training)

        output=conv

        conv = batch_norm(conv, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                          is_training=is_training, scope='bn_last')
        conv = tf.nn.relu(conv)

        shape = conv.get_shape().as_list()
        conv = tf.nn.avg_pool3d(conv, (1, shape[1], shape[2], shape[3], 1), strides=(1, 1, 1, 1, 1), padding='VALID')

        conv = tf.nn.conv3d(conv, self.fc1, strides=(1, 1, 1, 1, 1), padding='VALID') + self.fc1b
        conv = tf.squeeze(conv)
        conv = tf.cond(expand_dims,lambda:tf.expand_dims(conv, axis=0),lambda:conv)
        onehot_labels = tf.one_hot(indices=id, depth=2, on_value=1.0, off_value=0.0)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=conv)
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(cross_entropy)

        tf.add_to_collection('losses', loss)
        prediction = tf.equal(tf.expand_dims(tf.argmax(conv, 1), axis=1), tf.cast(id, tf.int64))
        with tf.name_scope('accuracy'):
            acc = tf.reduce_mean(tf.cast(prediction, tf.float32))

        return image, id, loss, acc, is_training, expand_dims,tf.nn.softmax(conv, dim=-1),output
