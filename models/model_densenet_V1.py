import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers.layers import batch_norm
from tensorflow.contrib.layers.python.layers import xavier_initializer as xavier

class convmodel():
    def __init__(self,loss_weight):
        '''
        :param loss_weight:'equal','uncertainty','revised_uncertainty','DWA'
        '''
        self.loss_weight=loss_weight
        growth = 16 # Growth Rate
        init = growth * 2 # Dense-BC
        growth4 = 4 * growth # Dense-B
        blocksize = [6, 12, 24, 16] # Dense-121

        with tf.variable_scope("model1"):
            self.w1 = tf.get_variable(name='w1', shape=[3,3,3,1,init], initializer=xavier())
            # self.w1 = tf.get_variable(name='w1', shape=[5,5,5,1,init], initializer=xavier())

            self.w2 = []
            for i in range(0, blocksize[0]):
                self.w2.append(tf.get_variable(name='w2_1x1_' + str(i), shape=[1,1,1,init + i * growth, growth4], initializer=xavier()))
                self.w2.append(tf.get_variable(name='w2_' + str(i), shape=[3,3,3,growth4,growth], initializer=xavier()))

            init1 = (init + blocksize[0] * growth) / 2 #64
            self.w3_1x1 = tf.get_variable(name='w3_1x1', shape=[1,1,1,init + blocksize[0] * growth, init1], initializer=xavier()) # Dense-C

            self.w3 = []
            for i in range(0, blocksize[1]):
                self.w3.append(tf.get_variable(name='w3_1x1_' + str(i), shape=[1, 1, 1, init1 + i * growth, growth4],
                                               initializer=xavier()))
                self.w3.append(
                    tf.get_variable(name='w3_' + str(i), shape=[3, 3, 3, growth4, growth], initializer=xavier()))

            init2 = (init1 + blocksize[1] * growth) / 2 #128
            self.w4_1x1 = tf.get_variable(name='w4_1x1', shape=[1, 1, 1, init1 + blocksize[1] * growth, init2],
                                          initializer=xavier())
            self.w4 = []
            for i in range(0, blocksize[2]):
                self.w4.append(tf.get_variable(name='w4_1x1_' + str(i), shape=[1, 1, 1, init2 + i * growth, growth4],
                                               initializer=xavier()))
                self.w4.append(
                    tf.get_variable(name='w4_' + str(i), shape=[3, 3, 3, growth4, growth], initializer=xavier()))

            init3 = (init2 + blocksize[2] * growth) / 2 #256
            self.w5_1x1 = tf.get_variable(name='w5_1x1', shape=[1, 1, 1, init2 + blocksize[2] * growth, init3],
                                          initializer=xavier())

            self.w5 = []
            for i in range(0, blocksize[3]):
                self.w5.append(tf.get_variable(name='w5_1x1_' + str(i), shape=[1, 1, 1, init3 + i * growth, growth4],
                                               initializer=xavier()))
                self.w5.append(
                    tf.get_variable(name='w5_' + str(i), shape=[3, 3, 3, growth4, growth], initializer=xavier()))

            self.fc1_det = tf.get_variable(name='fc1_det', shape=[1, 1, 1, init1, 2],
                                       initializer=xavier())
            # self.fc1_det = tf.get_variable(name='fc1_det', shape=[1, 1, 1, init2, 2],
            #                            initializer=xavier())
            self.fc1b_det = tf.get_variable(name='fc1b_det', shape=[2], initializer=xavier())

            self.fc1_cls = tf.get_variable(name='fc1_cls', shape=[1, 1, 1, init3 + blocksize[3] * growth, 2],
                                       initializer=xavier())
            self.fc1b_cls = tf.get_variable(name='fc1b_cls', shape=[2], initializer=xavier())
            if loss_weight == 'equal':
                self.log_vars = tf.get_variable(name='log_vars', shape=[2], initializer=tf.constant_initializer([0.5,0.5]), trainable=False)
            else:
                self.log_vars = tf.get_variable(name='log_vars', shape=[2], initializer=tf.constant_initializer([0.5,0.5]))

    def denseblock(self, input, kernel1, kernel2, layer_name, is_training):

        with tf.name_scope(layer_name):
            c = batch_norm(input, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name + '1')
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel1, strides=(1, 1, 1, 1, 1), padding='SAME')
            c = batch_norm(c, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name + '2')
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel2, strides=(1, 1, 1, 1, 1), padding='SAME')

            # c = self.SE_block(c,c._shape_as_list()[-1],16,layer_name+'_SE')
            input = tf.concat([input, c], axis=4)

        return input

    def add_transition(self, c, kernel, layer_name, is_training):

        with tf.name_scope(layer_name):
            c = batch_norm(c, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name)
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel, strides=(1, 1, 1, 1, 1), padding='SAME')
            c = tf.nn.avg_pool3d(c, (1, 2, 2, 2, 1), strides=(1, 2, 2, 2, 1), padding='SAME')

        return c

    def add_transition_wo_pool(self, c, kernel, layer_name, is_training):

        with tf.name_scope(layer_name):
            c = batch_norm(c, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name)
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel, strides=(1, 1, 1, 1, 1), padding='SAME')

        return c

    def convnet_weighted(self, zSize, ySize, xSize, alpha=0.0):

        image = tf.placeholder("float32", [None, zSize, ySize, xSize, 1])
        det_label = tf.placeholder("int32", [None, zSize/2,ySize/2,xSize/2])
        cls_label = tf.placeholder("int32", [None, 1])
        expand_dims = tf.placeholder(tf.bool, shape=())

        is_training = tf.placeholder(tf.bool, shape=())
        conv = tf.nn.conv3d(image, self.w1, strides=(1, 2, 2, 2, 1), padding='SAME')
        # conv = tf.nn.conv3d(image, self.w1, strides=(1, 1, 1, 1, 1), padding='SAME')
        for i in range(0, len(self.w2), 2):
            conv = self.denseblock(conv, self.w2[i], self.w2[i + 1], 'conv2_' + str(i), is_training)
        ###########    V1   ##########
        conv = self.add_transition_wo_pool(conv, self.w3_1x1, 'conv3_1x1', is_training)
        det_conv = tf.nn.relu(conv)
        det_conv = tf.nn.conv3d(det_conv, self.fc1_det, strides=(1, 1, 1, 1, 1), padding='SAME') + self.fc1b_det
        conv = tf.nn.avg_pool3d(conv, (1, 2, 2, 2, 1), strides=(1, 2, 2, 2, 1), padding='SAME')
        ##############################
        # conv = self.add_transition(conv, self.w3_1x1, 'conv3_1x1', is_training)
        for i in range(0, len(self.w3), 2):
            conv = self.denseblock(conv, self.w3[i], self.w3[i + 1], 'conv3_' + str(i), is_training)
        conv = self.add_transition(conv, self.w4_1x1, 'conv4_1x1', is_training)
        # conv = self.add_transition_wo_pool(conv, self.w4_1x1, 'conv4_1x1', is_training)
        # det_conv = tf.nn.relu(conv)
        # det_conv = tf.nn.conv3d(det_conv, self.fc1_det, strides=(1, 1, 1, 1, 1), padding='SAME') + self.fc1b_det
        # conv = tf.nn.avg_pool3d(conv, (1, 2, 2, 2, 1), strides=(1, 2, 2, 2, 1), padding='SAME')

        ### maskattention module add
        for i in range(0, len(self.w4), 2):
            conv = self.denseblock(conv, self.w4[i], self.w4[i + 1], 'conv4_' + str(i), is_training)
        conv = self.add_transition(conv, self.w5_1x1, 'conv5_1x1', is_training)

        for i in range(0, len(self.w5), 2):
            conv = self.denseblock(conv, self.w5[i], self.w5[i + 1], 'conv5_' + str(i), is_training)
        conv = batch_norm(conv, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                          is_training=is_training, scope='bn_last')
        conv = tf.nn.relu(conv)
        shape = conv.get_shape().as_list()
        conv = tf.nn.avg_pool3d(conv, (1, shape[1], shape[2], shape[3], 1), strides=(1, 1, 1, 1, 1), padding='VALID')

        conv = tf.nn.conv3d(conv, self.fc1_cls, strides=(1, 1, 1, 1, 1), padding='VALID') + self.fc1b_cls
        conv = tf.squeeze(conv)
        conv = tf.cond(expand_dims,lambda:tf.expand_dims(conv, axis=0),lambda:conv)

        ########## detection loss
        onehot_labels = tf.one_hot(det_label, 2, 1., 0.)
        flat_logits = tf.reshape(det_conv, [-1,2])
        flat_labels = tf.reshape(onehot_labels, [-1,2])
        loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)

        class_weight = tf.constant([1.,20.])

        weight_map = tf.reduce_max(tf.multiply(flat_labels,class_weight), axis = 1)
        weighted_loss = tf.multiply(loss_map,weight_map)
        det_loss = tf.reduce_mean(weighted_loss) # weighted on label = 1 as much as class weight

        class_weight_reverse = tf.constant([1.,50.])

        argmax_onehat_conv = tf.one_hot(tf.cast(tf.argmax(det_conv,4),tf.int32), depth=2, on_value=1.,off_value=0.)
        flat_argmax_onehot_conv = tf.reshape(argmax_onehat_conv, [-1,2])
        weight_map_reverse = tf.reduce_max(tf.multiply(flat_argmax_onehot_conv,class_weight_reverse),axis=1)
        weighted_loss_reverse = tf.multiply(loss_map,weight_map_reverse)
        det_loss += tf.reduce_mean(weighted_loss_reverse) # weighted on logit = 1 as much as class weight reverse

        ########## classificiation loss
        onehot_labels = tf.one_hot(indices=cls_label, depth=2, on_value=1.0, off_value=0.0)
        alpha = 0.1
        onehot_labels = tf.add((1 - alpha) * onehot_labels, alpha / 2)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=conv)
        cls_loss = tf.reduce_mean(cross_entropy)

        ##### fixed loss
        if self.loss_weight == 'equal':
            loss = self.log_vars[0]*det_loss + self.log_vars[1]*cls_loss

        #####uncertainty weights
        elif self.loss_weight == 'uncertainty':
            factor1 = tf.div(1.0, tf.pow(self.log_vars[0],2))
            factor2 = tf.div(1.0, tf.pow(self.log_vars[1],2))
            loss = factor1*det_loss + factor2*cls_loss + tf.log(self.log_vars[0]) + tf.log(self.log_vars[1]) + 0.001*tf.nn.l2_loss(self.log_vars)

        #####revised uncertainty weights
        elif self.loss_weight == 'revised uncertainty':
            factor1 = tf.div(1.0, tf.pow(self.log_vars[0],2))
            factor2 = tf.div(1.0, tf.pow(self.log_vars[1],2))
            loss = factor1*det_loss + factor2*cls_loss + tf.log(1 + tf.pow(self.log_vars[0],2)) + tf.log(1 + tf.pow(self.log_vars[1],2)) + 0.001*tf.nn.l2_loss(self.log_vars)

        tf.add_to_collection('losses',loss)

        det_prediction = tf.equal(tf.argmax(det_conv,4), tf.cast(det_label, tf.int64))
        with tf.name_scope('accuracy'):
            det_acc = tf.reduce_mean(tf.cast(det_prediction, tf.float32))

        cls_prediction = tf.equal(tf.expand_dims(tf.argmax(conv, 1), axis=1), tf.cast(cls_label, tf.int64))
        with tf.name_scope('accuracy'):
            cls_acc = tf.reduce_mean(tf.cast(cls_prediction, tf.float32))

        return image,det_label,cls_label,det_loss,cls_loss,loss,det_acc,cls_acc,\
               is_training,expand_dims,tf.nn.softmax(det_conv,dim=-1),tf.nn.softmax(conv, dim=-1), self.log_vars

    def convval(self,th):

        image = tf.placeholder("float32", [None, 48, 32, 48, 1])
        cls_label = tf.placeholder("int32", [None, 1])
        expand_dims = tf.placeholder(tf.bool, shape=())


        is_training = tf.placeholder(tf.bool, shape=())
        conv = tf.nn.conv3d(image, self.w1, strides=(1, 2, 2, 2, 1), padding='SAME')
        for i in range(0, len(self.w2), 2):
            conv = self.denseblock(conv, self.w2[i], self.w2[i + 1], 'conv2_' + str(i), is_training)
        conv = self.add_transition_wo_pool(conv, self.w3_1x1, 'conv3_1x1', is_training)
        conv = tf.nn.avg_pool3d(conv, (1, 2, 2, 2, 1), strides=(1, 2, 2, 2, 1), padding='SAME')
        for i in range(0, len(self.w3), 2):
            conv = self.denseblock(conv, self.w3[i], self.w3[i + 1], 'conv3_' + str(i), is_training)
        conv = self.add_transition(conv, self.w4_1x1, 'conv4_1x1', is_training)

        for i in range(0, len(self.w4), 2):
            conv = self.denseblock(conv, self.w4[i], self.w4[i + 1], 'conv4_' + str(i), is_training)
        conv = self.add_transition(conv, self.w5_1x1, 'conv5_1x1', is_training)

        for i in range(0, len(self.w5), 2):
            conv = self.denseblock(conv, self.w5[i], self.w5[i + 1], 'conv5_' + str(i), is_training)
        output = conv
        conv = batch_norm(conv, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                          is_training=is_training, scope='bn_last')
        conv = tf.nn.relu(conv)
        shape = conv.get_shape().as_list()
        conv = tf.nn.avg_pool3d(conv, (1, shape[1], shape[2], shape[3], 1), strides=(1, 1, 1, 1, 1), padding='VALID')

        conv = tf.nn.conv3d(conv, self.fc1_cls, strides=(1, 1, 1, 1, 1), padding='VALID') + self.fc1b_cls
        conv = tf.squeeze(conv)
        conv = tf.cond(expand_dims,lambda:tf.expand_dims(conv, axis=0),lambda:conv)

        ########## classificiation loss
        onehot_labels = tf.one_hot(indices=cls_label, depth=2, on_value=1.0, off_value=0.0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=conv)
        cls_loss = tf.reduce_mean(cross_entropy)


        tf.add_to_collection('losses',cls_loss)

        conv_th = tf.cast(tf.greater_equal(conv[:,1],th), tf.int64)
        cls_prediction = tf.equal(tf.expand_dims(conv_th, axis=1), tf.cast(cls_label, tf.int64))
        with tf.name_scope('accuracy'):
            cls_acc = tf.reduce_mean(tf.cast(cls_prediction, tf.float32))

        return image, cls_label,cls_loss,cls_acc, is_training, expand_dims, tf.nn.softmax(conv, dim=-1), output

    def convval_branch(self,th):

        image = tf.placeholder("float32", [None, 48, 32, 48, 1])
        det_label = tf.placeholder("int32", [None, 24, 16, 24])
        expand_dims = tf.placeholder(tf.bool, shape=())
        is_training = tf.placeholder(tf.bool, shape=())

        conv = tf.nn.conv3d(image, self.w1, strides=(1, 2, 2, 2, 1), padding='SAME')
        for i in range(0, len(self.w2), 2):
            conv = self.denseblock(conv, self.w2[i], self.w2[i + 1], 'conv2_' + str(i), is_training)
        ###########    V1   ##########
        conv = self.add_transition_wo_pool(conv, self.w3_1x1, 'conv3_1x1', is_training)
        det_conv = tf.nn.relu(conv)
        det_conv = tf.nn.conv3d(det_conv, self.fc1_det, strides=(1, 1, 1, 1, 1), padding='SAME') + self.fc1b_det

        ########## detection loss
        onehot_labels = tf.one_hot(det_label, 2, 1., 0.)
        flat_logits = tf.reshape(det_conv, [-1, 2])
        flat_labels = tf.reshape(onehot_labels, [-1, 2])
        loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
        det_loss = tf.reduce_mean(loss_map)  # weighted on label = 1 as much as class weight
        tf.add_to_collection('losses', det_loss)
        det_prediction = tf.equal(tf.argmax(det_conv, 4), tf.cast(det_label, tf.int64))
        with tf.name_scope('accuracy'):
            det_acc = tf.reduce_mean(tf.cast(det_prediction, tf.float32))

        return image, det_label, det_loss, det_acc, is_training, expand_dims, tf.nn.softmax(det_conv, dim=-1)

    def SE_block(self, input_x, out_dim, ratio, layer_name):
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):

            in_shape = input_x._shape_as_list()
            # print 'input_shape', input_x._shape_as_list()

            squeeze = tf.nn.avg_pool3d(input_x,[1, in_shape[1], in_shape[2], in_shape[3], 1], strides=[1 ,1 ,1 ,1 ,1], padding='VALID')
            # print 'squeeze shape', squeeze._shape_as_list()
            # excitation = tf.layers.dense(input=squeeze,use_bias=False, units=out_dim//ratio)
            excitation = self.Fully_connected(squeeze, out_dim // ratio, layer_name + '_fully_connected1')
            excitation = tf.nn.relu(excitation)
            # print 'excitation1 shape', excitation._shape_as_list()
            # excitation = tf.layers.dense(input=excitation,use_bias=False, units=out_dim//ratio)
            excitation = self.Fully_connected(excitation, out_dim, layer_name + '_fully_connected2')
            excitation = tf.nn.sigmoid(excitation)
            # print 'excitation2 shape', excitation._shape_as_list()

            excitation = tf.reshape(excitation, [-1, 1, 1, 1, out_dim])
            # print 'reshape shape', excitation._shape_as_list()

            scale = input_x * excitation

        return scale


