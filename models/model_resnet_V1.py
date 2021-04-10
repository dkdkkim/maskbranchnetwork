import tensorflow as tf
from tensorflow.contrib.layers.python.layers.layers import batch_norm
from tensorflow.contrib.layers.python.layers import xavier_initializer as xavier
from tensorflow.contrib.layers import variance_scaling_initializer

class convmodel():
    def __init__(self, loss_weight):
        self.loss_weight = loss_weight
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

            self.fc1_det = tf.get_variable(name='fc1_det', shape=[1, 1, 1, init1, 2],
                                       initializer=xavier())
            # self.fc1_det = tf.get_variable(name='fc1_det', shape=[1, 1, 1, init2, 2],
            #                            initializer=xavier())
            self.fc1b_det = tf.get_variable(name='fc1b_det', shape=[2], initializer=xavier())

            self.fc1_cls = tf.get_variable(name='fc1_cls', shape=[1, 1, 1, init4, 2],
                                       initializer=xavier())
            self.fc1b_cls = tf.get_variable(name='fc1b_cls', shape=[2], initializer=xavier())

            if loss_weight == 'equal':
                self.log_vars = tf.get_variable(name='log_vars', shape=[2], initializer=tf.constant_initializer([0.5,0.5]), trainable=False)
            else:
                self.log_vars = tf.get_variable(name='log_vars', shape=[2], initializer=tf.constant_initializer([0.5,0.5]))

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

            # c = self.SE_block(c,c._shape_as_list()[-1],16,layer_name+'_SE')
            # c = self.CBAM_block(c,layer_name+'_CBAM',kernel3)

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

            # c = self.SE_block(c,c._shape_as_list()[-1],16,layer_name+'_SE')
            # c = self.CBAM_block(c,layer_name+'_CBAM',kernel3)

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

            # c = self.SE_block(c,c._shape_as_list()[-1],16,layer_name+'_SE')
            # c = self.CBAM_block(c,layer_name+'_CBAM',kernel3)

            output = input + c
            output = tf.nn.relu(output)
        return output

    def convnet_weighted(self, zSize, ySize, xSize, alpha=0.0):

        image = tf.placeholder("float32", [None, zSize, ySize, xSize, 1])
        det_label = tf.placeholder("int32", [None, zSize/2,ySize/2,xSize/2])
        cls_label = tf.placeholder("int32", [None, 1])
        is_training = tf.placeholder(tf.bool, shape=())
        expand_dims = tf.placeholder(tf.bool, shape=())

        conv = tf.nn.conv3d(image, self.w1, strides=(1, 1, 1, 1, 1), padding='SAME')
        # conv = tf.nn.conv3d(image, self.w1, strides=(1, 2, 2, 2, 1), padding='SAME')

        for i in range(0, len(self.w2), 3):
            if i == 0:
                conv = self.residual_block_3layers_downsampling(conv, self.w2_init,self.w2[i], self.w2[i + 1], self.w2[i + 2], 'conv2_' + str(i), is_training)
            else:
                conv = self.residual_block_3layers(conv, self.w2[i], self.w2[i + 1], self.w2[i + 2], 'conv2_' + str(i),
                                                   is_training)
        ###########    V1   ##########
        det_conv = tf.nn.relu(conv)
        det_conv = tf.nn.conv3d(det_conv, self.fc1_det, strides=(1, 1, 1, 1, 1), padding='SAME') + self.fc1b_det
        # conv = tf.nn.avg_pool3d(conv, (1, 2, 2, 2, 1), strides=(1, 2, 2, 2, 1), padding='SAME')
        ##############################
        for i in range(0, len(self.w3), 3):
            if i == 0:
                conv = self.residual_block_3layers_downsampling(conv, self.w3_init, self.w3[i], self.w3[i + 1], self.w3[i + 2], 'conv3_' + str(i),
                                                   is_training)
            else:
                conv = self.residual_block_3layers(conv, self.w3[i], self.w3[i + 1], self.w3[i + 2],'conv3_' + str(i), is_training)

        # conv = self._add_transition(conv, self.w4_1x1, 'conv4_1x1', is_training)
        for i in range(0, len(self.w4), 3):
            if i == 0:
                conv = self.residual_block_3layers_downsampling(conv, self.w4_init, self.w4[i], self.w4[i + 1], self.w4[i + 2],'conv4_' + str(i), is_training)
            else:
                conv = self.residual_block_3layers(conv, self.w4[i], self.w4[i + 1], self.w4[i + 2], 'conv4_' + str(i),
                                                   is_training)

        # conv = self._add_transition(conv, self.w5_1x1, 'conv5_1x1', is_training)
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

        conv = tf.nn.conv3d(conv, self.fc1_cls, strides=(1, 1, 1, 1, 1), padding='VALID') + self.fc1b_cls
        conv = tf.squeeze(conv)
        conv = tf.cond(expand_dims,lambda:tf.expand_dims(conv, axis=0),lambda:conv)

        ########## detection loss
        onehot_labels = tf.one_hot(det_label, 2, 1., 0.)
        flat_logits = tf.reshape(det_conv, [-1, 2])
        flat_labels = tf.reshape(onehot_labels, [-1, 2])
        loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)

        class_weight = tf.constant([1., 20.])

        weight_map = tf.reduce_max(tf.multiply(flat_labels, class_weight), axis=1)
        weighted_loss = tf.multiply(loss_map, weight_map)
        det_loss = tf.reduce_mean(weighted_loss)  # weighted on label = 1 as much as class weight

        class_weight_reverse = tf.constant([1., 50.])

        argmax_onehat_conv = tf.one_hot(tf.cast(tf.argmax(det_conv, 4), tf.int32), depth=2, on_value=1., off_value=0.)
        flat_argmax_onehot_conv = tf.reshape(argmax_onehat_conv, [-1, 2])
        weight_map_reverse = tf.reduce_max(tf.multiply(flat_argmax_onehot_conv, class_weight_reverse), axis=1)
        weighted_loss_reverse = tf.multiply(loss_map, weight_map_reverse)
        det_loss += tf.reduce_mean(weighted_loss_reverse)  # weighted on logit = 1 as much as class weight reverse

        ########## classificiation loss
        onehot_labels = tf.one_hot(indices=cls_label, depth=2, on_value=1.0, off_value=0.0)
        alpha = 0.1
        onehot_labels = tf.add((1 - alpha) * onehot_labels, alpha / 2)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=conv)
        cls_loss = tf.reduce_mean(cross_entropy)

        ##### fixed loss
        if self.loss_weight == 'equal':
            loss = det_loss + cls_loss

        #####uncertainty weights
        elif self.loss_weight == 'uncertainty':
            factor1 = tf.div(1.0, tf.pow(self.log_vars[0],2))
            factor2 = tf.div(1.0, tf.pow(self.log_vars[1],2))
            loss = factor1*det_loss + factor2*cls_loss + tf.log(self.log_vars[0]) + tf.log(self.log_vars[1]) + 0.001*tf.nn.l2_loss(self.log_vars)

        #####revised uncertainty weights
        elif self.loss_weight == 'revised_uncertainty':
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

        return image, det_label, cls_label, det_loss, cls_loss, loss, det_acc, cls_acc, \
               is_training, expand_dims, tf.nn.softmax(det_conv, dim=-1), tf.nn.softmax(conv, dim=-1), self.log_vars

    def convval(self, th):

        image = tf.placeholder("float32", [None, 48, 32, 48, 1])
        cls_label = tf.placeholder("int32", [None, 1])
        is_training = tf.placeholder(tf.bool, shape=())
        expand_dims = tf.placeholder(tf.bool, shape=())

        conv = tf.nn.conv3d(image, self.w1, strides=(1, 1, 1, 1, 1), padding='SAME')
        # conv = tf.nn.conv3d(image, self.w1, strides=(1, 2, 2, 2, 1), padding='SAME')

        for i in range(0, len(self.w2), 3):
            if i == 0:
                conv = self.residual_block_3layers_downsampling(conv, self.w2_init,self.w2[i], self.w2[i + 1], self.w2[i + 2], 'conv2_' + str(i), is_training)
            else:
                conv = self.residual_block_3layers(conv, self.w2[i], self.w2[i + 1], self.w2[i + 2], 'conv2_' + str(i),
                                                   is_training)
        ###########    V1   ##########
        det_conv = tf.nn.relu(conv)
        det_conv = tf.nn.conv3d(det_conv, self.fc1_det, strides=(1, 1, 1, 1, 1), padding='SAME') + self.fc1b_det
        # conv = tf.nn.avg_pool3d(conv, (1, 2, 2, 2, 1), strides=(1, 2, 2, 2, 1), padding='SAME')
        ##############################
        for i in range(0, len(self.w3), 3):
            if i == 0:
                conv = self.residual_block_3layers_downsampling(conv, self.w3_init, self.w3[i], self.w3[i + 1], self.w3[i + 2], 'conv3_' + str(i),
                                                   is_training)
            else:
                conv = self.residual_block_3layers(conv, self.w3[i], self.w3[i + 1], self.w3[i + 2],'conv3_' + str(i), is_training)

        # conv = self._add_transition(conv, self.w4_1x1, 'conv4_1x1', is_training)
        for i in range(0, len(self.w4), 3):
            if i == 0:
                conv = self.residual_block_3layers_downsampling(conv, self.w4_init, self.w4[i], self.w4[i + 1], self.w4[i + 2],'conv4_' + str(i), is_training)
            else:
                conv = self.residual_block_3layers(conv, self.w4[i], self.w4[i + 1], self.w4[i + 2], 'conv4_' + str(i),
                                                   is_training)

        # conv = self._add_transition(conv, self.w5_1x1, 'conv5_1x1', is_training)
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
        # print shape
        conv = tf.nn.avg_pool3d(conv, (1, shape[1], shape[2], shape[3], 1), strides=(1, 1, 1, 1, 1), padding='VALID')

        conv = tf.nn.conv3d(conv, self.fc1_cls, strides=(1, 1, 1, 1, 1), padding='VALID') + self.fc1b_cls
        conv = tf.squeeze(conv)
        conv = tf.cond(expand_dims,lambda:tf.expand_dims(conv, axis=0),lambda:conv)

        ########## classificiation loss
        onehot_labels = tf.one_hot(indices=cls_label, depth=2, on_value=1.0, off_value=0.0)
        alpha = 0.1
        onehot_labels = tf.add((1 - alpha) * onehot_labels, alpha / 2)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=conv)
        cls_loss = tf.reduce_mean(cross_entropy)

        tf.add_to_collection('losses',cls_loss)


        cls_prediction = tf.equal(tf.expand_dims(tf.argmax(conv, 1), axis=1), tf.cast(cls_label, tf.int64))
        with tf.name_scope('accuracy'):
            cls_acc = tf.reduce_mean(tf.cast(cls_prediction, tf.float32))

        return image, cls_label,cls_loss,cls_acc, is_training, expand_dims, tf.nn.softmax(conv, dim=-1), output


    def Fully_connected(self, x, units, layer_name):
        with tf.name_scope(layer_name):
            return tf.layers.dense(inputs=x, use_bias=False, units=units)

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

    def channel_attention(self, input,ratio):

        input_shape = input._shape_as_list()
        out_dim= input_shape[-1]

        avg_pool = tf.nn.avg_pool3d(input,[1,input_shape[1],input_shape[2],input_shape[2],1], strides=[1,1,1,1,1], padding='VALID')
        avg_pool = tf.layers.dense(inputs=avg_pool, use_bias=False, units=out_dim//ratio,name = 'CA_avg_fc1')
        avg_pool = tf.layers.dense(inputs=avg_pool, use_bias=False, units=out_dim,name = 'CA_avg_fc2')

        max_pool = tf.nn.max_pool3d(input,[1,input_shape[1],input_shape[2],input_shape[3],1], strides=[1,1,1,1,1], padding='VALID')
        max_pool = tf.layers.dense(max_pool, use_bias=False, units=out_dim//ratio,name = 'CA_max_fc1')
        max_pool = tf.layers.dense(max_pool, use_bias=False, units=out_dim,name = 'CA_max_fc2')

        CA_feature = tf.add(avg_pool,max_pool)
        CA_feature = tf.nn.sigmoid(CA_feature)

        return input * CA_feature

    def spacial_attention(self, input,kernel):

        # print 'SA input :', input._shape_as_list()
        avg_pool = tf.math.reduce_mean(input,axis=4,keepdims=True)
        max_pool = tf.math.reduce_max(input,axis=4,keepdims=True)
        # print 'pool shape :', avg_pool._shape_as_list(), max_pool._shape_as_list()
        concat = tf.concat([avg_pool,max_pool],axis=4)
        # print 'concat shape :', concat._shape_as_list()
        SA_feature = tf.nn.conv3d(concat,kernel, strides=(1,1,1,1,1), padding='SAME',name='SA')
        SA_feature = tf.sigmoid(SA_feature)
        return input * SA_feature

    def CBAM_block(self,input,layer_name,kernel,ratio=8):#ratio=8

        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):

            # print 'CBAM input :',input._shape_as_list()
            cbam_feature = self.channel_attention(input,ratio)
            # print 'CBAM CA :',cbam_feature._shape_as_list()
            cbam_feature = self.spacial_attention(cbam_feature,kernel)
            # print 'CBAM SA :',cbam_feature._shape_as_list()

        return cbam_feature

