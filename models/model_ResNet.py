import tensorflow as tf
from tensorflow.contrib.layers.python.layers.layers import batch_norm
from tensorflow.contrib.layers.python.layers import xavier_initializer as xavier

class convmodel():
    def __init__(self):
        init = 32
        # blocksize = [3, 4, 6, 3]  # Resnet-50
        blocksize = [3, 4, 23, 3]  # Resnet-101
        # blocksize = [3, 8, 36, 3]  # Resnet-152

        with tf.variable_scope("model1"):
            self.w1 = tf.get_variable(name='w1', shape=[3, 3, 3, 1, init], initializer=xavier())

            init1=init * 4 # 256

            self.w2_init = tf.get_variable(name='w2_init', shape=[1, 1, 1, init, init1], initializer=xavier())
            self.w2 = []
            for i in range(0, blocksize[0]):
                if i ==0:
                    self.w2.append(tf.get_variable(name='w2_a_' + str(i), shape=[1, 1, 1, init, init1/4],
                                               initializer=xavier()))
                else:
                    self.w2.append(tf.get_variable(name='w2_a_' + str(i), shape=[1, 1, 1, init1, init1/4],
                                               initializer=xavier()))
                self.w2.append(
                    tf.get_variable(name='w2_b_' + str(i), shape=[3, 3, 3, init1/4, init1/4], initializer=xavier()))
                self.w2.append(
                    tf.get_variable(name='w2_c_' + str(i), shape=[1, 1, 1, init1/4, init1], initializer=xavier()))

            init2=init1 * 2 #512
            self.w3_init = tf.get_variable(name='w3_init', shape=[1, 1, 1, init1, init2], initializer=xavier())
            self.w3 = []
            for i in range(0, blocksize[1]):
                if i == 0:
                    self.w3.append(tf.get_variable(name='w3_a_' + str(i), shape=[1, 1, 1, init1, init2/4],
                                               initializer=xavier()))
                else:
                    self.w3.append(tf.get_variable(name='w3_a_' + str(i), shape=[1, 1, 1, init2, init2 / 4],
                                                   initializer=xavier()))

                self.w3.append(
                    tf.get_variable(name='w3_b_' + str(i), shape=[3, 3, 3, init2/4, init2/4], initializer=xavier()))
                self.w3.append(tf.get_variable(name='w3_c_' + str(i), shape=[1, 1, 1, init2/4, init2],
                                               initializer=xavier()))

            init3 = init2 * 2 #1024
            self.w4_init = tf.get_variable(name='w4_init', shape=[1, 1, 1, init2, init3], initializer=xavier())
            self.w4 = []
            for i in range(0, blocksize[2]):
                if i == 0:
                    self.w4.append(tf.get_variable(name='w4_a_' + str(i), shape=[1, 1, 1, init2, init3/4],
                                                   initializer=xavier()))
                else:
                    self.w4.append(tf.get_variable(name='w4_a_' + str(i), shape=[1, 1, 1, init3, init3 / 4],
                                                   initializer=xavier()))
                self.w4.append(
                    tf.get_variable(name='w4_b_' + str(i), shape=[3, 3, 3, init3/4, init3/4], initializer=xavier()))
                self.w4.append(tf.get_variable(name='w4_c_' + str(i), shape=[1, 1, 1, init3/4, init3],
                                               initializer=xavier()))

            init4 = init3 * 2 #2048
            self.w5_init = tf.get_variable(name='w5_init', shape=[1, 1, 1, init3, init4], initializer=xavier())
            self.w5 = []
            for i in range(0, blocksize[3]):
                if i == 0:
                    self.w5.append(tf.get_variable(name='w5_a_' + str(i), shape=[1, 1, 1, init3, init4/4],
                                                   initializer=xavier()))
                else :
                    self.w5.append(tf.get_variable(name='w5_a_' + str(i), shape=[1, 1, 1, init4, init4 / 4],
                                                   initializer=xavier()))
                self.w5.append(
                    tf.get_variable(name='w5_b_' + str(i), shape=[3, 3, 3, init4/4, init4/4], initializer=xavier()))
                self.w5.append(tf.get_variable(name='w5_c_' + str(i), shape=[1, 1, 1, init4/4, init4],
                                               initializer=xavier()))

            self.fc1 = tf.get_variable(name='fc1', shape=[1, 1, 1, init4, 2],
                                       initializer=xavier())
            self.fc1b = tf.get_variable(name='fc1b', shape=[2], initializer=xavier())

    def residual_block_2layers(self, input, kernel1, kernel2, layer_name, is_training):

        with tf.name_scope(layer_name):
            c = batch_norm(input, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name + '1')
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel1, strides=(1, 1, 1, 1, 1), padding='SAME')
            c = batch_norm(c, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name + '2')
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel2, strides=(1, 1, 1, 1, 1), padding='SAME')

            output = input + c

        return output

    def residual_block_3layers(self, input, kernel1, kernel2, kernel3, layer_name, is_training):

        with tf.name_scope(layer_name):
            c = batch_norm(input, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name + '1')
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel1, strides=(1, 1, 1, 1, 1), padding='SAME')
            c = batch_norm(c, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name + '2')
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel2, strides=(1, 1, 1, 1, 1), padding='SAME')
            c = batch_norm(c, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name + '3')
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel3, strides=(1, 1, 1, 1, 1), padding='SAME')

            output = input + c
            output = tf.nn.relu(output)
        return output

    def residual_block_3layers_downsampling(self, input, kernel_0, kernel1, kernel2, kernel3, layer_name, is_training):

        with tf.name_scope(layer_name):
            input = batch_norm(input, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name + '1')
            input = tf.nn.relu(input)

            c = tf.nn.conv3d(input, kernel1, strides=(1, 2, 2, 2, 1), padding='SAME')
            input = tf.nn.conv3d(input, kernel_0, strides=(1, 2, 2, 2, 1), padding='SAME')

            c = batch_norm(c, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name + '2')
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel2, strides=(1, 1, 1, 1, 1), padding='SAME')
            c = batch_norm(c, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name + '3')
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel3, strides=(1, 1, 1, 1, 1), padding='SAME')

            output = input + c
            output = tf.nn.relu(output)
        return output

    def convnet(self):
        image = tf.placeholder("float32", [None, 48, 32, 48, 1])

        id = tf.placeholder("int32", [None, 1])
        is_training = tf.placeholder(tf.bool, shape=())
        expand_dims = tf.placeholder(tf.bool, shape=())

        conv = tf.nn.conv3d(image, self.w1, strides=(1, 1, 1, 1, 1), padding='SAME')

        for i in range(0, len(self.w2), 3):
            if i == 0:
                conv = self.residual_block_3layers_downsampling(conv, self.w2_init,self.w2[i], self.w2[i + 1], self.w2[i + 2], 'conv2_' + str(i), is_training)
            else:
                conv = self.residual_block_3layers(conv, self.w2[i], self.w2[i + 1], self.w2[i + 2], 'conv2_' + str(i),
                                                   is_training)
        for i in range(0, len(self.w3), 3):
            if i == 0:
                conv = self.residual_block_3layers_downsampling(conv, self.w3_init, self.w3[i], self.w3[i + 1], self.w3[i + 2], 'conv3_' + str(i),
                                                   is_training)
            else:
                conv = self.residual_block_3layers(conv, self.w3[i], self.w3[i + 1], self.w3[i + 2],'conv3_' + str(i), is_training)

        for i in range(0, len(self.w4), 3):
            if i == 0:
                conv = self.residual_block_3layers_downsampling(conv, self.w4_init, self.w4[i], self.w4[i + 1], self.w4[i + 2],'conv4_' + str(i), is_training)
            else:
                conv = self.residual_block_3layers(conv, self.w4[i], self.w4[i + 1], self.w4[i + 2], 'conv4_' + str(i),
                                                   is_training)

        for i in range(0, len(self.w5), 3):
            if i == 0:
                conv = self.residual_block_3layers_downsampling(conv, self.w5_init, self.w5[i], self.w5[i + 1], self.w5[i + 2], 'conv5_' + str(i), is_training)
            else:
                conv = self.residual_block_3layers(conv, self.w5[i], self.w5[i + 1], self.w5[i + 2], 'conv5_' + str(i),
                                                   is_training)

        conv = batch_norm(conv, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                          is_training=is_training, scope='bn_last')
        conv = tf.nn.relu(conv)
        shape = conv.get_shape().as_list()
        print shape
        conv = tf.nn.avg_pool3d(conv, (1, shape[1], shape[2], shape[3], 1), strides=(1, 1, 1, 1, 1), padding='VALID')

        conv = tf.nn.conv3d(conv, self.fc1, strides=(1, 1, 1, 1, 1), padding='VALID') + self.fc1b
        conv = tf.squeeze(conv)
        conv = tf.cond(expand_dims,lambda:tf.expand_dims(conv, axis=0),lambda:conv)


        onehot_labels = tf.one_hot(indices=id, depth=2, on_value=1.0, off_value=0.0)
        alpha=0.1
        onehot_labels = tf.add((1 - alpha) * onehot_labels, alpha / 2)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=conv)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(cross_entropy)

        tf.add_to_collection('losses', loss)

        prediction = tf.equal(tf.expand_dims(tf.argmax(conv, 1), axis=1), tf.cast(id, tf.int64))
        with tf.name_scope('accuracy'):
            acc = tf.reduce_mean(tf.cast(prediction, tf.float32))

        return image, id, loss, acc, is_training, expand_dims, tf.nn.softmax(conv, dim=-1), cross_entropy#, expand_dims

    def convval(self,th):
        image = tf.placeholder("float32", [None, 48, 32, 48, 1])

        id = tf.placeholder("int32", [None, 1])
        is_training = tf.placeholder(tf.bool, shape=())
        expand_dims = tf.placeholder(tf.bool, shape=())

        conv = tf.nn.conv3d(image, self.w1, strides=(1, 1, 1, 1, 1), padding='SAME')

        for i in range(0, len(self.w2), 3):
            if i == 0:
                conv = self.residual_block_3layers_downsampling(conv, self.w2_init,self.w2[i], self.w2[i + 1], self.w2[i + 2], 'conv2_' + str(i), is_training)
            else:
                conv = self.residual_block_3layers(conv, self.w2[i], self.w2[i + 1], self.w2[i + 2], 'conv2_' + str(i),
                                                   is_training)
        for i in range(0, len(self.w3), 3):
            if i == 0:
                conv = self.residual_block_3layers_downsampling(conv, self.w3_init, self.w3[i], self.w3[i + 1], self.w3[i + 2], 'conv3_' + str(i),
                                                   is_training)
            else:
                conv = self.residual_block_3layers(conv, self.w3[i], self.w3[i + 1], self.w3[i + 2],'conv3_' + str(i), is_training)

        for i in range(0, len(self.w4), 3):
            if i == 0:
                conv = self.residual_block_3layers_downsampling(conv, self.w4_init, self.w4[i], self.w4[i + 1], self.w4[i + 2],'conv4_' + str(i), is_training)
            else:
                conv = self.residual_block_3layers(conv, self.w4[i], self.w4[i + 1], self.w4[i + 2], 'conv4_' + str(i),
                                                   is_training)

        for i in range(0, len(self.w5), 3):
            if i == 0:
                conv = self.residual_block_3layers_downsampling(conv, self.w5_init, self.w5[i], self.w5[i + 1], self.w5[i + 2], 'conv5_' + str(i), is_training)
            else:
                conv = self.residual_block_3layers(conv, self.w5[i], self.w5[i + 1], self.w5[i + 2], 'conv5_' + str(i),
                                                   is_training)

        conv = batch_norm(conv, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                          is_training=is_training, scope='bn_last')
        output = conv
        conv = tf.nn.relu(conv)
        shape = conv.get_shape().as_list()
        print shape
        conv = tf.nn.avg_pool3d(conv, (1, shape[1], shape[2], shape[3], 1), strides=(1, 1, 1, 1, 1), padding='VALID')

        conv = tf.nn.conv3d(conv, self.fc1, strides=(1, 1, 1, 1, 1), padding='VALID') + self.fc1b
        conv = tf.squeeze(conv)
        conv = tf.cond(expand_dims,lambda:tf.expand_dims(conv, axis=0),lambda:conv)

        onehot_labels = tf.one_hot(indices=id, depth=2, on_value=1.0, off_value=0.0)
        alpha=0.1
        onehot_labels = tf.add((1 - alpha) * onehot_labels, alpha / 2)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=conv)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(cross_entropy)

        tf.add_to_collection('losses', loss)

        prediction = tf.equal(tf.expand_dims(tf.argmax(conv, 1), axis=1), tf.cast(id, tf.int64))
        with tf.name_scope('accuracy'):
            acc = tf.reduce_mean(tf.cast(prediction, tf.float32))

        return image, id, loss, acc, is_training, expand_dims,tf.nn.softmax(conv, dim=-1),output#, expand_dims
