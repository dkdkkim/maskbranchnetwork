import sys

sys.path.append('/home/dkkim/workplace/Multitask/abus')

import numpy as np
import tensorflow as tf
from models.model_densenet_DWA import convmodel as convmodel_multiV1_dense_DWA
from models.model_densenet_V1 import convmodel as convmodel_multiV1_dense
from models.model_densenet_V2 import convmodel as convmodel_multiV2_dense
from models.model_densenet_V3 import convmodel as convmodel_multiV3_dense
from models.model_densenet_V4 import convmodel as convmodel_multiV4_dense
from models.model_resnet_V1 import convmodel as convmodel_multiV1_res
from models.model_resnet_V2 import convmodel as convmodel_multiV2_res
from models.model_resnet_V3 import convmodel as convmodel_multiV3_res
from models.model_resnet_V4 import convmodel as convmodel_multiV4_res
import os,json,random,sys, math

from mask_utils import load_dataset, load_mass_train_data, load_mass_valid_data, load_mass_whole_valid_data, \
    load_final_train_data, load_final_valid_data, send_email, average_gradients, data_load
from utility_detection import data_load_woedge,data_load_woedge_spacing, data_load_spacing, \
    find_optimal_threshold, plot_roc_curve, plot_confusion_matrix
from utility_class import biopsy_data_setting
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

flags = tf.app.flags

flags.DEFINE_integer("num_gpu",1,help="Number of GPUs")
flags.DEFINE_integer("batch",48,help="batch size")
flags.DEFINE_float("lr_init",0.0001, help="init learning rate")
flags.DEFINE_string("save_dir", "exp01_ABUS_detection_test", help="save directory")
flags.DEFINE_string("CUDA_VISIBLE_DEVICES","1",help="GPU numbers")
flags.DEFINE_string("description","desc test", help="message which is sent by email")
# flags.DEFINE_string("network","network test", help="mask network used for training")
# flags.DEFINE_string("label_type","label_test", help="dataset used for training")
# flags.DEFINE_string("loss_weight","weight test", help="how to weigh loss")
# flags.DEFINE_string("data_type","dataset test", help="how to weigh loss")

flags.DEFINE_string("network","V3_dense", help="mask network used for training")
flags.DEFINE_string("label_type","mask_orient_0.7", help="dataset used for training")
flags.DEFINE_string("loss_weight","uncertainty", help="how to weigh loss")
flags.DEFINE_string("data_type","mass", help="how to weigh loss")

FLAGS = flags.FLAGS

print '*'*10,' / '.join([FLAGS.network, FLAGS.loss_weight, FLAGS.label_type])

save_dir = '/data/dk/exp/MaskAttention/'+FLAGS.save_dir

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.CUDA_VISIBLE_DEVICES

class Model():
    def __init__(self, loss_weight, network):
        '''
        :param loss_weight:'equal','uncertainty','revised_uncertainty'
        :param network: 'V1_dense', 'V1_res'
        :parameter network: 'V1_dense', 'V1_res'
        :raise asdfasdf
        :raises asdfsdfasdf
        '''
        if not os.path.exists(save_dir+'/weights'): os.makedirs(save_dir+'/weights')
        if not os.path.exists(save_dir+'/results'): os.makedirs(save_dir+'/results')
        if not os.path.exists(save_dir+'/log'): os.makedirs(save_dir+'/log')
        if not os.path.exists(save_dir+'/figure'): os.makedirs(save_dir+'/figure')
        self.network = network
        self.loss_weight = loss_weight
        self.cs = 48
        self.gpu_num = list(range(FLAGS.num_gpu))
        self.batch, self.hbatch = FLAGS.batch, FLAGS.batch/2
        self.res_check_step = 50
        # self.res_check_step = 1
        self.model_nums,self.steps = [],[]

        self.train_cls_accs, self.val_cls_paccs, self.val_cls_naccs, self.cls_losses, self.losses, \
        self.val_cls_plosses, self.val_plosses, self.val_aucs, \
        self.val_cls_nlosses, self.val_nlosses \
            = [],[],[],[],[],[],[],[],[],[]

        if self.network in ['V3_dense','V3_res']:
            self.train_det_accs_small, self.train_det_accs_large, \
            self.det_losses_small, self.det_losses_large = [], [], [], []
            self.val_det_paccs_small, self.val_det_paccs_large, \
            self.val_det_plosses_small, self.val_det_plosses_large, \
            self.val_psensitivities_small, self.val_psensitivities_large = [], [], [], [], [], []
            self.val_det_naccs_small, self.val_det_naccs_large, \
            self.val_det_nlosses_small, self.val_det_nlosses_large, self.val_nspecificities_small, self.val_nspecificities_large \
                = [], [], [], [], [], []
        else:
            self.train_det_accs, self.val_det_paccs, self.val_det_naccs, self.det_losses, self.val_det_nlosses, \
            self.val_det_plosses, self.val_psensitivities, self.val_nspecificities \
                = [], [], [], [], [], [], [], []

    def tower_loss(self,scope):
        if self.network == 'V1_dense':
            if self.loss_weight=='DWA':
                model = convmodel_multiV1_dense_DWA()
            else:
                model = convmodel_multiV1_dense(loss_weight=self.loss_weight)
        elif self.network == 'V2_dense':
            model = convmodel_multiV2_dense(loss_weight=self.loss_weight)
        elif self.network == 'V4_dense':
            model = convmodel_multiV4_dense(loss_weight=self.loss_weight)
        elif self.network == 'V3_dense':
            model = convmodel_multiV3_dense(loss_weight=self.loss_weight)
        elif self.network == 'V1_res':
            model = convmodel_multiV1_res(loss_weight=self.loss_weight)
        elif self.network == 'V2_res':
            model = convmodel_multiV2_res(loss_weight=self.loss_weight)
        elif self.network == 'V3_res':
            model = convmodel_multiV3_res(loss_weight=self.loss_weight)
        elif self.network == 'V4_res':
            model = convmodel_multiV4_res(loss_weight=self.loss_weight)

        if self.network in ['V3_dense','V3_res']:
            image, det_label_small, det_label_large, cls_label, det_loss_small, det_loss_large, cls_loss, loss, det_acc_small, det_acc_large, cls_acc, \
            is_training, expand_dims, det_prob_small, det_prob_large, cls_prob, log_vars = model.convnet_weighted(
                self.cs, self.cs * 2 / 3, self.cs, alpha=0.1)
            losses = tf.get_collection('losses', scope)
            total_loss = tf.add_n(losses, name='total_loss')
            return image, det_label_small, det_label_large, cls_label, det_loss_small, det_loss_large, cls_loss, loss, \
                   det_acc_small, det_acc_large, cls_acc, is_training, expand_dims, \
                   det_prob_small, det_prob_large, cls_prob, log_vars, total_loss
        else:
            if self.loss_weight == 'DWA':
                image, det_label, cls_label, det_loss, cls_loss, loss, det_acc, cls_acc, \
                is_training, expand_dims, det_prob, cls_prob, wk, cur_weights = model.convnet_weighted(self.cs,
                                                                                                       self.cs * 2 / 3,
                                                                                                       self.cs)
                losses = tf.get_collection('losses', scope)
                total_loss = tf.add_n(losses, name='total_loss')

                return image, det_label, cls_label, det_loss, cls_loss, loss, det_acc, cls_acc, is_training, expand_dims, \
                       det_prob, cls_prob, wk, cur_weights, total_loss
            else:
                image, det_label, cls_label, det_loss, cls_loss, loss, det_acc, cls_acc,\
                is_training, expand_dims, det_prob, cls_prob, log_vars = model.convnet_weighted(self.cs,self.cs*2/3,self.cs)

                losses = tf.get_collection('losses', scope)
                total_loss = tf.add_n(losses, name='total_loss')

                return image, det_label, cls_label, det_loss, cls_loss, loss, det_acc, cls_acc, is_training, expand_dims,\
                     det_prob, cls_prob, log_vars, total_loss

    def make_label(self):
        cs = 72 / 2
        ref_arr_oval = np.zeros((cs, cs, cs), dtype=np.float32)
        x_list = range(cs)
        y_list = range(cs)
        z_list = range(cs)
        for x in x_list:
            for y in y_list:
                for z in z_list:
                    ref_arr_oval[x, y, z] = (
                        math.sqrt((x - cs / 2) ** 2 + ((y - cs / 2) / 0.5) ** 2 + (z - cs / 2) ** 2))
        sp_arr = np.copy(ref_arr_oval)
        crop = np.where(sp_arr <= cs/2, 1., 0.)

        return crop
    def npy_load(self,npy_path):

        x = np.empty(shape=[len(npy_path), self.cs, self.cs*2/3, self.cs])

        for idx, cur_path in enumerate(npy_path):
            cur_arr = np.load(cur_path)
            shp = cur_arr.shape
            cur_arr = cur_arr[shp[0]/2-self.cs/2:shp[0]/2+self.cs/2,
                      shp[1]/2-self.cs/3:shp[1]/2+self.cs/3,
                      shp[2]/2-self.cs/2:shp[2]/2+self.cs/2]
            x[idx] = cur_arr
        return np.expand_dims(x, axis=4)

    def npy_load_lbl(self,npy_path):
        if FLAGS.network in ['V2_dense','V4_dense','V2_res','V4_res']:
            x = np.empty(shape=[len(npy_path), self.cs / 4, self.cs / 6, self.cs / 4])
        else:
            x = np.empty(shape=[len(npy_path), self.cs/2, self.cs/3, self.cs/2])

        for idx, cur_path in enumerate(npy_path):
            if self.label_type == 'fixed':
                cur_arr = np.copy(self.ref_lbl)
            else:
                cur_arr = np.load(cur_path)
            shp = cur_arr.shape
            cur_arr = cur_arr[shp[0]/2-self.cs/4:shp[0]/2+self.cs/4,
                      shp[1]/2-self.cs/6:shp[1]/2+self.cs/6,
                      shp[2]/2-self.cs/4:shp[2]/2+self.cs/4]
            if FLAGS.network in ['V2_dense', 'V4_dense','V2_res','V4_res']:
                cur_arr = cur_arr[::2, ::2, ::2]
                if np.sum(cur_arr) == 0:
                    cur_arr = np.zeros(shape=(self.cs / 4, self.cs / 6, self.cs / 4))
                    cur_arr[self.cs / 8, self.cs / 12, self.cs / 8] = 1.
            x[idx] = cur_arr

        return x

    def npy_load_lbl_small(self,npy_path):

        x = np.empty(shape=[len(npy_path), self.cs/4, self.cs/6, self.cs/4])

        for idx, cur_path in enumerate(npy_path):
            cur_arr = np.load(cur_path)
            shp = cur_arr.shape
            cur_arr = cur_arr[shp[0]/2-self.cs/4:shp[0]/2+self.cs/4,
                      shp[2]/2-self.cs/6:shp[2]/2+self.cs/6,
                      shp[2]/2-self.cs/4:shp[2]/2+self.cs/4]
            cur_arr = cur_arr[::2,::2,::2]
            if np.sum(cur_arr)==0:
                cur_arr = np.zeros(shape=(self.cs/4, self.cs/6, self.cs/4))
                cur_arr[self.cs/8, self.cs/12, self.cs/8] = 1.
            x[idx] = cur_arr
        return x

    def npy_load_lbl_large(self,npy_path):
        x = np.empty(shape=[len(npy_path), self.cs/2, self.cs/3, self.cs/2])

        for idx, cur_path in enumerate(npy_path):
            cur_arr = np.load(cur_path)
            shp = cur_arr.shape
            cur_arr = cur_arr[shp[0]/2-self.cs/4:shp[0]/2+self.cs/4,shp[2]/2-self.cs/6:shp[2]/2+self.cs/6,
                      shp[2]/2-self.cs/4:shp[2]/2+self.cs/4]
            x[idx] = cur_arr
        return x

    def npy_load_shift(self, img_path, lbl_path):
        if self.data_type in ['mass','final'] and len(img_path) != 2*len(lbl_path):
            print 'numbers of lbl,img is different!'

        img = np.empty(shape=[len(img_path), self.cs, self.cs*2/3, self.cs])

        if FLAGS.network in ['V2_dense','V4_dense','V2_res','V4_res']:
            lbl = np.empty(shape=[len(img_path), self.cs/4, self.cs/6, self.cs/4])
        else:
            lbl = np.empty(shape=[len(img_path), self.cs / 2, self.cs / 3, self.cs / 2])

        for idx in range(len(img_path)):
            if self.data_type in ['mass','final'] and idx >= len(img_path) / 2:
                cur_img = np.load(img_path[idx])
                img_shp = cur_img.shape
                cur_img = cur_img[img_shp[0] / 2 - self.cs/2 : img_shp[0] / 2 + self.cs/2 ,
                          img_shp[1] / 2 - self.cs/3 : img_shp[1] / 2 + self.cs/3 ,
                          img_shp[2] / 2 - self.cs/2 : img_shp[2] / 2 + self.cs/2 ]
                if FLAGS.network in ['V2_dense', 'V4_dense','V2_res','V4_res']:
                    cur_lbl = np.zeros(shape=[self.cs / 4, self.cs / 6, self.cs / 4])
                else:
                    cur_lbl = np.zeros(shape=[self.cs/2,self.cs/3,self.cs/2])
            else:
                cur_img = np.load(img_path[idx])
                if self.label_type == 'fixed':
                    cur_lbl = np.copy(self.ref_lbl)
                else:
                    cur_lbl = np.load(lbl_path[idx])
                shift = [random.sample([-6, 0, 6], 1)[0], random.sample([-3, 0, 3], 1)[0],
                         random.sample([-6, 0, 6], 1)[0]]
                img_shp = cur_img.shape
                lbl_shp = cur_lbl.shape
                cur_img = cur_img[img_shp[0] / 2 - self.cs/2 + shift[0] : img_shp[0] / 2 + self.cs/2 + shift[0],
                          img_shp[1] / 2 - self.cs/3 + shift[1] : img_shp[1] / 2 + self.cs/3 + shift[1],
                          img_shp[2] / 2 - self.cs/2 + shift[2] : img_shp[2] / 2 + self.cs/2 + shift[2]]
                cur_lbl = cur_lbl[lbl_shp[0] / 2 - self.cs/4 +shift[0] / 2 : lbl_shp[0] / 2 + self.cs/4 + shift[0] / 2,
                          lbl_shp[1] / 2 - self.cs/6 + shift[1] / 2 : lbl_shp[1] / 2 + self.cs/6 + shift[1] / 2,
                          lbl_shp[2] / 2 - self.cs/4 + shift[2] / 2 : lbl_shp[2] / 2 + self.cs/4 + shift[2] / 2]
                if FLAGS.network in ['V2_dense', 'V4_dense','V2_res','V4_res']:
                    cur_lbl = cur_lbl[::2, ::2, ::2]
                    if np.sum(cur_lbl) == 0:
                        cur_lbl = np.zeros(shape=(self.cs / 4, self.cs / 6, self.cs / 4))
                        cur_lbl[self.cs / 8, self.cs / 12, self.cs / 8] = 1.
            # else:
            #     cur_img = np.load(img_path[idx])
            #     img_shp = cur_img.shape
            #     cur_img = cur_img[img_shp[0] / 2 - self.cs/2 : img_shp[0] / 2 + self.cs/2 ,
            #               img_shp[1] / 2 - self.cs/3 : img_shp[1] / 2 + self.cs/3 ,
            #               img_shp[2] / 2 - self.cs/2 : img_shp[2] / 2 + self.cs/2 ]
            #     if FLAGS.network in ['V2_dense', 'V4_dense','V2_res','V4_res']:
            #         cur_lbl = np.zeros(shape=[self.cs / 4, self.cs / 6, self.cs / 4])
            #     else:
            #         cur_lbl = np.zeros(shape=[self.cs/2,self.cs/3,self.cs/2])

            img[idx] = cur_img
            lbl[idx] = cur_lbl

        return np.expand_dims(img,axis=4),lbl

    def npy_load_shift_cancer(self, img_path, lbl_path):
        if len(img_path) != 2*len(lbl_path):
            print 'numbers of lbl,img is different!'

        img = np.empty(shape=[len(img_path), self.cs, self.cs*2/3, self.cs])

        if FLAGS.network in ['V2_dense','V4_dense','V2_res','V4_res']:
            lbl = np.empty(shape=[len(img_path), self.cs/4, self.cs/6, self.cs/4])
        else:
            lbl = np.empty(shape=[len(img_path), self.cs / 2, self.cs / 3, self.cs / 2])

        for idx in range(len(img_path)):

            if idx < len(img_path) / 2:
                cur_img = np.load(img_path[idx])
                if self.label_type == 'fixed':
                    cur_lbl = np.copy(self.ref_lbl)
                else:
                    cur_lbl = np.load(lbl_path[idx])
                shift = [random.sample([-6, 0, 6], 1)[0], random.sample([-3, 0, 3], 1)[0],
                         random.sample([-6, 0, 6], 1)[0]]
                img_shp = cur_img.shape
                lbl_shp = cur_lbl.shape
                cur_img = cur_img[img_shp[0] / 2 - self.cs/2 + shift[0] : img_shp[0] / 2 + self.cs/2 + shift[0],
                          img_shp[1] / 2 - self.cs/3 + shift[1] : img_shp[1] / 2 + self.cs/3 + shift[1],
                          img_shp[2] / 2 - self.cs/2 + shift[2] : img_shp[2] / 2 + self.cs/2 + shift[2]]
                cur_lbl = cur_lbl[lbl_shp[0] / 2 - self.cs/4 +shift[0] / 2 : lbl_shp[0] / 2 + self.cs/4 + shift[0] / 2,
                          lbl_shp[1] / 2 - self.cs/6 + shift[1] / 2 : lbl_shp[1] / 2 + self.cs/6 + shift[1] / 2,
                          lbl_shp[2] / 2 - self.cs/4 + shift[2] / 2 : lbl_shp[2] / 2 + self.cs/4 + shift[2] / 2]
                if FLAGS.network in ['V2_dense', 'V4_dense','V2_res','V4_res']:
                    cur_lbl = cur_lbl[::2, ::2, ::2]
                    if np.sum(cur_lbl) == 0:
                        cur_lbl = np.zeros(shape=(self.cs / 4, self.cs / 6, self.cs / 4))
                        cur_lbl[self.cs / 8, self.cs / 12, self.cs / 8] = 1.
            else:
                cur_img = np.load(img_path[idx])
                img_shp = cur_img.shape
                cur_img = cur_img[img_shp[0] / 2 - self.cs/2 : img_shp[0] / 2 + self.cs/2 ,
                          img_shp[1] / 2 - self.cs/3 : img_shp[1] / 2 + self.cs/3 ,
                          img_shp[2] / 2 - self.cs/2 : img_shp[2] / 2 + self.cs/2 ]
                if FLAGS.network in ['V2_dense', 'V4_dense','V2_res','V4_res']:
                    cur_lbl = np.zeros(shape=[self.cs / 4, self.cs / 6, self.cs / 4])
                else:
                    cur_lbl = np.zeros(shape=[self.cs/2,self.cs/3,self.cs/2])

            img[idx] = cur_img
            lbl[idx] = cur_lbl

        return np.expand_dims(img,axis=4),lbl

    def npy_load_shift_V3(self, img_path, lbl_path):
        if len(img_path) != 2*len(lbl_path):
            print 'numbers of lbl,img is different!'

        img = np.empty(shape=[len(img_path), self.cs, self.cs*2/3, self.cs])
        lbl_small = np.empty(shape=[len(img_path), self.cs/4, self.cs/6, self.cs/4])
        lbl_large = np.empty(shape=[len(img_path), self.cs/2, self.cs/3, self.cs/2])

        for idx in range(len(img_path)):

            if idx < len(img_path) / 2:
                cur_img = np.load(img_path[idx])
                cur_lbl = np.load(lbl_path[idx])
                shift = [random.sample([-6, 0, 6], 1)[0], random.sample([-3, 0, 3], 1)[0],
                         random.sample([-6, 0, 6], 1)[0]]
                img_shp = cur_img.shape
                lbl_shp = cur_lbl.shape
                cur_img = cur_img[img_shp[0] / 2 - self.cs/2 + shift[0] : img_shp[0] / 2 + self.cs/2 + shift[0],
                          img_shp[1] / 2 - self.cs/3 + shift[1] : img_shp[1] / 2 + self.cs/3 + shift[1],
                          img_shp[2] / 2 - self.cs/2 + shift[2] : img_shp[2] / 2 + self.cs/2 + shift[2]]
                cur_lbl_large = cur_lbl[lbl_shp[0] / 2 - self.cs/4 +shift[0] / 2 : lbl_shp[0] / 2 + self.cs/4 + shift[0] / 2,
                          lbl_shp[1] / 2 - self.cs/6 + shift[1] / 2 : lbl_shp[1] / 2 + self.cs/6 + shift[1] / 2,
                          lbl_shp[2] / 2 - self.cs/4 + shift[2] / 2 : lbl_shp[2] / 2 + self.cs/4 + shift[2] / 2]
                cur_lbl_small = cur_lbl_large[::2, ::2, ::2]
                if np.sum(cur_lbl_small) == 0:
                    cur_lbl_small = np.zeros(shape=(self.cs / 4, self.cs / 6, self.cs / 4))
                    cur_lbl_small[self.cs / 8, self.cs / 12, self.cs / 8] = 1.
            else:
                cur_img = np.load(img_path[idx])
                shp = cur_img.shape
                cur_img = cur_img[shp[0]/2-self.cs/2: shp[0]/2+self.cs/2,
                          shp[1]/2 - self.cs/3: shp[1]/2 + self.cs/3,
                          shp[2] / 2 - self.cs / 2: shp[2] / 2 + self.cs / 2]
                cur_lbl_small = np.zeros(shape=[self.cs / 4, self.cs / 6, self.cs / 4])
                cur_lbl_large = np.zeros(shape=[self.cs / 2, self.cs / 3, self.cs / 2])

            img[idx] = cur_img
            lbl_small[idx] = cur_lbl_small
            lbl_large[idx] = cur_lbl_large

        return np.expand_dims(img,axis=4),lbl_small, lbl_large

    def tf_model(self):

        with tf.Graph().as_default():

            self.global_step = tf.get_variable(
                'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            lr_init = FLAGS.lr_init
            # self.lr = tf.train.exponential_decay(lr_init, self.global_step, 4000, 0.5, staircase=True)
            self.lr = tf.train.exponential_decay(lr_init, self.global_step, 10000, 0.1, staircase=True)

            opt = tf.train.RMSPropOptimizer(self.lr, decay=0.9, epsilon=0.1)

            tower_grads = []
            self.towers = {}

            with tf.variable_scope(tf.get_variable_scope()):
                for gpu in self.gpu_num:
                    with tf.device('/gpu:%d' % gpu):
                        with tf.name_scope('%s_%d' % ('tower', gpu)) as scope:
                            num = str(gpu)
                            if self.network in ['V3_dense', 'V3_res']:
                                self.towers['img' + num], self.towers['det_label_small' + num], self.towers[
                                    'det_label_large' + num], self.towers['cls_label' + num], \
                                self.towers['det_loss_small' + num], self.towers['det_loss_large' + num], self.towers[
                                    'cls_loss' + num], self.towers['loss' + num], \
                                self.towers['det_acc_small' + num], self.towers['det_acc_large' + num], self.towers[
                                    'cls_acc' + num], self.towers['is_training' + num], self.towers[
                                    'expand_dims' + num], \
                                self.towers['det_prob_small' + num], self.towers['det_prob_large' + num], self.towers[
                                    'cls_prob' + num], self.towers['log_vars' + num], total_loss \
                                    = self.tower_loss(scope)
                            else:
                                if self.loss_weight == 'DWA':
                                    self.towers['img' + num], self.towers['det_label' + num], self.towers[
                                        'cls_label' + num], \
                                    self.towers['det_loss' + num], self.towers['cls_loss' + num], self.towers[
                                        'loss' + num], \
                                    self.towers['det_acc' + num], self.towers['cls_acc' + num], self.towers[
                                        'is_training' + num], self.towers['expand_dims' + num], \
                                    self.towers['det_prob' + num], self.towers['cls_prob' + num], self.towers[
                                        'wk' + num], self.towers['log_vars' + num], total_loss \
                                        = self.tower_loss(scope)
                                else:
                                    self.towers['img' + num], self.towers['det_label' + num], self.towers['cls_label' + num], \
                                    self.towers['det_loss' + num], self.towers['cls_loss' + num], self.towers['loss' + num], \
                                    self.towers['det_acc' + num], self.towers['cls_acc' + num],self.towers['is_training' + num],self.towers['expand_dims' + num], \
                                    self.towers['det_prob' + num], self.towers['cls_prob' + num],self.towers['log_vars' + num], total_loss \
                                        = self.tower_loss(scope)
                            tf.get_variable_scope().reuse_variables()
                            batchnorm_updates = tf.get_collection('_update_ops_', scope)
                            tower_grads.append(opt.compute_gradients(total_loss))
            # print tower_grads
            grads = average_gradients(tower_grads)
            apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

            batchnorm_updates_op = tf.group(*batchnorm_updates)
            train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

            self.saver = tf.train.Saver(max_to_keep=1000)

            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

            init = tf.global_variables_initializer()
            self.sess.run(init)

            ## restore model
            # restore_dir = '/data/dk/exp/MaskAttention/MASK_EXP10/weights/model-28'
            # self.saver.restore(self.sess, restore_dir)
            # # self.sess.run(self.global_step.initializer)
            # print 'restore from %s' % restore_dir

        self.writer_train = tf.summary.FileWriter(save_dir + '/log/train')
        self.writer_pval = tf.summary.FileWriter(save_dir + '/log/pval')
        self.writer_nval = tf.summary.FileWriter(save_dir + '/log/nval')
        self.writer_val = tf.summary.FileWriter(save_dir + '/log/val')


        self.fetches, self.valFetches = [train_op], []
        for gpu in self.gpu_num:
            if self.network in ['V3_dense', 'V3_res']:
                self.fetches += [self.towers['det_loss_small' + str(gpu)], self.towers['det_loss_large' + str(gpu)],
                                 self.towers['cls_loss' + str(gpu)], self.towers['loss' + str(gpu)],
                                 self.towers['det_acc_small' + str(gpu)], self.towers['det_acc_large' + str(gpu)],
                                 self.towers['cls_acc' + str(gpu)], self.towers['log_vars' + str(gpu)]]
                self.valFetches += [self.towers['det_loss_small' + str(gpu)], self.towers['det_loss_large' + str(gpu)],
                                    self.towers['cls_loss' + str(gpu)], self.towers['loss' + str(gpu)],
                                    self.towers['det_acc_small' + str(gpu)], self.towers['det_acc_large' + str(gpu)],
                                    self.towers['cls_acc' + str(gpu)],
                                    self.towers['det_prob_small' + str(gpu)], self.towers['det_prob_large' + str(gpu)],
                                    self.towers['cls_prob' + str(gpu)]]
            else:
                self.fetches += [self.towers['det_loss' + str(gpu)], self.towers['cls_loss' + str(gpu)], self.towers['loss' + str(gpu)],
                                 self.towers['det_acc' + str(gpu)], self.towers['cls_acc' + str(gpu)], self.towers['log_vars' + str(gpu)]]
                self.valFetches += [self.towers['det_loss' + str(gpu)], self.towers['cls_loss' + str(gpu)], self.towers['loss' + str(gpu)],
                                 self.towers['det_acc' + str(gpu)], self.towers['cls_acc' + str(gpu)],
                                    self.towers['det_prob' + str(gpu)], self.towers['cls_prob' + str(gpu)]]

    def dataset(self, label_type, data_type):
        '''

        :param label_type: orientation, mask, mask_orient_acoust, mask_orient_0.5, mask_orient_0.7,
                            mask_orient_1.0, mask_orient_1.2, mask_acoust_0.7
        :return:
        '''
        self.label_type = label_type
        self.data_type = data_type
        if data_type == 'mass':
            if self.label_type == 'fixed':
                self.ref_lbl = self.make_label()
                self.train_det_plbl, self.train_pimg, self.train_nimg = load_mass_train_data(label_type='mask_orient_0.7')
                self.tpList, self.tnList = [i for i in range(len(self.train_pimg))], [i for i in
                                                                                      range(len(self.train_nimg))]
                # self.valid_det_plbl, self.valid_pimg, self.valid_nimg = load_mass_valid_data(label_type=label_type)
                self.valid_det_plbl, self.valid_pimg, self.valid_nimg = load_mass_whole_valid_data(
                    label_type='mask_orient_0.7')
                self.vpList, self.vnList = [i for i in range(len(self.valid_pimg))], [i for i in
                                                                                      range(len(self.valid_nimg))]
            else:
                self.train_det_plbl, self.train_pimg, self.train_nimg = load_mass_train_data(label_type=self.label_type)
                self.tpList, self.tnList = [i for i in range(len(self.train_pimg))], [i for i in range(len(self.train_nimg))]

            # self.valid_det_plbl, self.valid_pimg, self.valid_nimg = load_mass_valid_data(label_type=label_type)
            self.valid_det_plbl, self.valid_pimg, self.valid_nimg = load_mass_whole_valid_data(label_type=self.label_type)
            self.vpList, self.vnList = [i for i in range(len(self.valid_pimg))], [i for i in range(len(self.valid_nimg))]
            if self.network in ['V3_dense', 'V3_res']:
                self.det_nvlbl_small = np.zeros(shape=[self.batch, self.cs / 4, self.cs / 6, self.cs / 4])
                self.det_nvlbl_large = np.zeros(shape=[self.batch, self.cs / 2, self.cs / 3, self.cs / 2])
            elif FLAGS.network in ['V2_dense','V4_dense','V2_res','V4_res']:
                self.det_nvlbl = np.zeros(shape=[self.batch, self.cs / 4, self.cs / 6, self.cs / 4])
            else:
                self.det_nvlbl = np.zeros(shape=[self.batch, self.cs/2,self.cs/3,self.cs/2])
        elif data_type == 'cancer':
            self.train_det_plbl, self.train_det_nlbl, self.valid_det_plbl, self.valid_det_nlbl = biopsy_data_setting(
                '/data/dk/datasets_CROPS/crops_ABUS_lbl_' + label_type)
            self.train_pimg = [pimg.replace("crops_ABUS_lbl_" + label_type, "crops_ABUS_rescale") for pimg in
                               self.train_det_plbl]
            self.train_nimg = [nimg.replace("crops_ABUS_lbl_" + label_type, "crops_ABUS_rescale") for nimg in
                               self.train_det_nlbl]
            self.tpList, self.tnList = [i for i in range(len(self.train_pimg))], [i for i in
                                                                                  range(len(self.train_nimg))]
            self.valid_pimg = [pimg.replace("crops_ABUS_lbl_" + label_type, "crops_ABUS_rescale") for pimg in
                               self.valid_det_plbl]
            self.valid_nimg = [nimg.replace("crops_ABUS_lbl_" + label_type, "crops_ABUS_rescale") for nimg in
                               self.valid_det_nlbl]
            self.vpList, self.vnList = [i for i in range(len(self.valid_pimg))], [i for i in
                                                                                  range(len(self.valid_nimg))]
        elif data_type == 'final':
            self.train_det_plbl, self.train_pimg, self.train_nimg = load_final_train_data(label_type=self.label_type)
            self.tpList, self.tnList = [i for i in range(len(self.train_pimg))], [i for i in
                                                                                  range(len(self.train_nimg))]
            self.valid_det_plbl, self.valid_pimg, self.valid_nimg = load_final_valid_data(label_type=self.label_type)
            self.vpList, self.vnList = [i for i in range(len(self.valid_pimg))], [i for i in
                                                                                  range(len(self.valid_nimg))]
            if self.network in ['V3_dense', 'V3_res']:
                self.det_nvlbl_small = np.zeros(shape=[self.batch, self.cs / 4, self.cs / 6, self.cs / 4])
                self.det_nvlbl_large = np.zeros(shape=[self.batch, self.cs / 2, self.cs / 3, self.cs / 2])
            elif FLAGS.network in ['V2_dense','V4_dense','V2_res','V4_res']:
                self.det_nvlbl = np.zeros(shape=[self.batch, self.cs / 4, self.cs / 6, self.cs / 4])
            else:
                self.det_nvlbl = np.zeros(shape=[self.batch, self.cs/2,self.cs/3,self.cs/2])

        self.t_cls_label = np.append(np.ones([self.hbatch, 1]), np.zeros([self.hbatch, 1]), axis=0)
        self.pv_cls_label, self.nv_cls_label = np.ones([self.batch, 1]), np.zeros([self.batch, 1])

        print len(self.tpList), len(self.tnList), len(self.vpList), len(self.vnList)

    def run_model(self):
        self.cls_acc_sum, self.cls_loss_sum, self.loss_sum, self.model_num,self.idx = 0, 0, 0, 0, 0
        if self.loss_weight == 'DWA':
            self.loss0,self.loss1 = 0., 0.
            self.wk = np.array([1.,1.])

        if self.network in ['V3_dense', 'V3_res']:
            self.det_acc_small_sum, self.det_acc_large_sum, self.det_loss_small_sum, \
            self.det_loss_large_sum = 0, 0, 0, 0
        else:
            self.det_acc_sum, self.cls_acc_sum, self.det_loss_sum, self.cls_loss_sum, self.loss_sum, \
            self.model_num = 0, 0, 0, 0, 0, 0

        for idx, step in enumerate(range(40000), 1):
            # try:
            self.idx = idx
            feed_dict = {}

            for gpu in self.gpu_num:
                pIdx, nIdx = random.sample(self.tpList, self.hbatch), random.sample(self.tnList, self.hbatch)
                if self.network in ['V3_dense', 'V3_res']:
                    img, det_lbl_small, det_lbl_large = self.npy_load_shift_V3(
                        [self.train_pimg[i] for i in pIdx] + [self.train_nimg[i] for i in nIdx],
                        [self.train_det_plbl[i] for i in pIdx])
                    feed_dict[self.towers['det_label_small' + str(gpu)]] = det_lbl_small
                    feed_dict[self.towers['det_label_large' + str(gpu)]] = det_lbl_large
                else:
                    if self.data_type == 'mass' or self.data_type == 'final':
                        img, det_lbl = self.npy_load_shift([self.train_pimg[i] for i in pIdx] + [self.train_nimg[i] for i in nIdx],
                                                           [self.train_det_plbl[i] for i in pIdx])
                    elif self.data_type == 'cancer':
                        img, det_lbl = self.npy_load_shift(
                            [self.train_pimg[i] for i in pIdx] + [self.train_nimg[i] for i in nIdx],
                            [self.train_det_plbl[i] for i in pIdx] + [self.train_det_nlbl[i] for i in nIdx])

                    feed_dict[self.towers['det_label' + str(gpu)]] = det_lbl
                feed_dict[self.towers['cls_label' + str(gpu)]] = self.t_cls_label
                feed_dict[self.towers['img' + str(gpu)]] = img
                feed_dict[self.towers['is_training' + str(gpu)]] = True
                feed_dict[self.towers['expand_dims' + str(gpu)]] = False
                if self.loss_weight == 'DWA':
                    feed_dict[self.towers['wk' + str(gpu)]] = self.wk

            cur_res = self.sess.run(self.fetches, feed_dict=feed_dict)
            cur_res = cur_res[1:]
            if self.network in ['V3_dense', 'V3_res']:
                for tmp in cur_res[3::7]:
                    assert not np.isnan(tmp), 'Model diverged with loss'

                self.det_loss_small_sum += np.mean(cur_res[::8])
                self.det_loss_large_sum += np.mean(cur_res[1::8])
                self.cls_loss_sum += np.mean(cur_res[2::8])
                self.loss_sum += np.mean(cur_res[3::8])
                self.det_acc_small_sum += np.mean(cur_res[4::8]) * 100
                self.det_acc_large_sum += np.mean(cur_res[5::8]) * 100
                # print np.mean(cur_res[6::8]) * 100
                self.cls_acc_sum += np.mean(cur_res[6::8]) * 100
                self.log_vars = cur_res[7::8]

                if self.idx % self.res_check_step == 0:
                    if self.idx % (self.res_check_step * 10) == 0:
                        self.res_check_V3(is_save=True)
                    else:
                        self.res_check_V3(is_save=False)
            else:
                for tmp in cur_res[2::6]:
                    assert not np.isnan(tmp), 'Model diverged with loss'

                self.det_loss_sum += np.mean(cur_res[::6])
                self.cls_loss_sum += np.mean(cur_res[1::6])
                self.loss_sum += np.mean(cur_res[2::6])
                self.det_acc_sum += np.mean(cur_res[3::6]) * 100
                self.cls_acc_sum += np.mean(cur_res[4::6]) * 100
                self.log_vars = cur_res[5::6]
                self.log_vars = self.log_vars[0]

                if self.loss_weight == 'DWA':
                    self.loss0 = self.loss1
                    self.loss1 = np.array([self.det_loss_sum, self.cls_loss_sum])
                    if idx == 1 or idx == 2:
                        self.wk = np.array([1., 1.])
                    else:
                        self.wk = self.loss1 / self.loss0

                if self.idx % self.res_check_step == 0:
                    if self.idx % (self.res_check_step * 10) == 0:
                        self.res_check(is_save=True)
                    else:
                        self.res_check(is_save=False)

            # try:


    def res_check(self, is_save=False):

        # print self.sum_count
        self.det_loss_sum /= self.res_check_step
        self.cls_loss_sum /= self.res_check_step
        self.loss_sum /= self.res_check_step
        self.det_acc_sum /= self.res_check_step
        self.cls_acc_sum /= self.res_check_step
        print save_dir, '*' * 20
        print "step: %d, model: %d, det_acc: %.2f%%, cls_acc: %.2f%%, " \
              "det_loss: %.5f, cls_loss: %.5f, loss: %.5f, global_step: %d, lr: %.6f, log var: %.4f,%.4f" % \
              (self.idx, self.model_num, self.det_acc_sum,  self.cls_acc_sum,
               self.det_loss_sum, self.cls_loss_sum, self.loss_sum, self.sess.run(self.global_step),
               self.sess.run(self.lr), self.log_vars[0], self.log_vars[1])

        self.det_acc_summary = tf.Summary()
        self.cls_acc_summary = tf.Summary()
        self.det_loss_summary = tf.Summary()
        self.cls_loss_summary = tf.Summary()
        self.loss_summary = tf.Summary()

        self.det_acc_summary.value.add(tag='det_acc', simple_value=self.det_acc_sum / 100.)
        self.cls_acc_summary.value.add(tag='cls_acc', simple_value=self.cls_acc_sum / 100.)
        self.det_loss_summary.value.add(tag='det_loss', simple_value=self.det_loss_sum)
        self.cls_loss_summary.value.add(tag='cls_loss', simple_value=self.cls_loss_sum)
        self.loss_summary.value.add(tag='loss', simple_value=self.loss_sum)
        self.writer_train.add_summary(self.det_acc_summary, global_step=self.idx)
        self.writer_train.add_summary(self.cls_acc_summary, global_step=self.idx)
        self.writer_train.add_summary(self.det_loss_summary, global_step=self.idx)
        self.writer_train.add_summary(self.cls_loss_summary, global_step=self.idx)
        self.writer_train.add_summary(self.loss_summary, global_step=self.idx)
        self.writer_train.flush()

        if self.model_num == 0:
            self.saver.save(self.sess, save_path=save_dir + '/weights/model', global_step=self.model_num)

        if is_save:
            val_det_ploss, val_cls_ploss, val_ploss, val_det_pacc, val_cls_pacc, val_psensitivity, \
                pos_cls_prob, pos_cls_label = self.validation_run(is_pos=True)
            val_det_nloss, val_cls_nloss, val_nloss, val_det_nacc, val_cls_nacc, val_nspecificity, \
                neg_cls_prob, neg_cls_label = self.validation_run(is_pos=False)
            label_arr = np.vstack((pos_cls_label,neg_cls_label))
            prob_arr = np.vstack((pos_cls_prob,neg_cls_prob))

            fpr, tpr, thresholds = metrics.roc_curve(label_arr, prob_arr[:, 1])
            auc_value = metrics.auc(fpr, tpr)
            opt_th = find_optimal_threshold(fpr, tpr, thresholds)

            sens_opt = np.where(pos_cls_prob[..., 1] >= opt_th, 1., 0.)
            spec_opt = np.where(neg_cls_prob[..., 1] < opt_th, 1., 0.)
            pacc_opt = np.mean(sens_opt) * 100
            nacc_opt = np.mean(spec_opt) * 100
            sens = np.where(pos_cls_prob[..., 1] >= 0.5, 1., 0.)
            spec = np.where(neg_cls_prob[..., 1] < 0.5, 1., 0.)
            pacc = np.mean(sens) * 100
            nacc = np.mean(spec) * 100

            print '\t val det pos loss: %.5f, val cls pos loss: %.5f, val pos loss: %.5f, ' \
                  'val det pos acc: %.2f%%, val cls pos acc: %.2f%%, val cls pos acc opt: %.2f%%, val pos sens: %.2f%%' % (
            val_det_ploss, val_cls_ploss, val_ploss, val_det_pacc, pacc, pacc_opt, val_psensitivity)
            print '\t val det neg loss: %.5f, val cls neg loss: %.5f, val neg loss: %.5f, ' \
                  'val det neg acc: %.2f%%, val cls neg acc: %.2f%%, val cls neg acc opt: %.2f%%, val neg spec: %.2f%%' % (
                      val_det_nloss, val_cls_nloss, val_nloss, val_det_nacc, nacc, nacc_opt, val_nspecificity)

            print 'AUC = %0.4f\nopt_th = %0.3f' % (auc_value, opt_th)
            plot_roc_curve(fpr, tpr, FLAGS.save_dir,self.model_num, auc_value, opt_th)
            plt.savefig(save_dir + '/figure/AUC_model_' + str(self.model_num) + '.png')
            plt.close()

            np.save(save_dir + '/figure/fpr_' + str(self.model_num), fpr)
            np.save(save_dir + '/figure/tpr_' + str(self.model_num), tpr)
            # roc_dict={'fpr':fpr,'tpr':tpr}
            # with open(save_dir + '/figure/roc_dict_' + str(self.model_num),'w') as file:
            #     json.dump(roc_dict,file)

            labels = ['benign', 'cancer']
            prob_binary = np.where(prob_arr[:, 1] >= 0.5, 1, 0)
            cm = metrics.confusion_matrix(label_arr, prob_binary)
            plot_confusion_matrix(cm, labels,val_cls_pacc,val_cls_nacc, normalize=False,
                                  title="Model %s Confusion Matrix" % str(self.model_num))
            plt.savefig(save_dir + '/figure/confusion_mat_' + str(self.model_num) + '.png')
            plt.close()

            self.val_det_nacc_summary = tf.Summary()
            self.val_cls_nacc_summary = tf.Summary()
            self.val_det_nloss_summary = tf.Summary()
            self.val_cls_nloss_summary = tf.Summary()
            self.val_det_pacc_summary = tf.Summary()
            self.val_cls_pacc_summary = tf.Summary()
            self.val_det_ploss_summary = tf.Summary()
            self.val_cls_ploss_summary = tf.Summary()
            self.val_cls_total_loss_summary = tf.Summary()
            self.val_auc_summary = tf.Summary()
            self.val_sens_summary = tf.Summary()
            self.val_spec_summary = tf.Summary()

            self.val_auc_summary.value.add(tag='auc', simple_value=auc_value)
            self.writer_val.add_summary(self.val_auc_summary, global_step=self.idx)
            self.val_cls_total_loss_summary.value.add(tag='cls_loss', simple_value=val_cls_ploss + val_cls_nloss)
            self.writer_val.add_summary(self.val_cls_total_loss_summary, global_step=self.idx)
            self.writer_val.flush()

            self.val_sens_summary.value.add(tag='sensitivity', simple_value=val_psensitivity)
            self.writer_pval.add_summary(self.val_sens_summary, global_step=self.idx)
            self.writer_pval.flush()

            self.val_spec_summary.value.add(tag='specificity', simple_value=val_nspecificity)
            self.writer_nval.add_summary(self.val_spec_summary, global_step=self.idx)
            self.writer_nval.flush()

            self.val_det_pacc_summary.value.add(tag='det_acc', simple_value=val_det_pacc/100.)
            self.val_det_ploss_summary.value.add(tag='det_loss', simple_value=val_det_ploss)
            self.writer_pval.add_summary(self.val_det_pacc_summary, global_step=self.idx)
            self.writer_pval.add_summary(self.val_det_ploss_summary, global_step=self.idx)
            self.writer_pval.flush()

            self.val_det_nacc_summary.value.add(tag='det_acc', simple_value=val_det_nacc/100.)
            self.val_det_nloss_summary.value.add(tag='det_loss', simple_value=val_det_nloss)
            self.writer_nval.add_summary(self.val_det_nacc_summary, global_step=self.idx)
            self.writer_nval.add_summary(self.val_det_nloss_summary, global_step=self.idx)
            self.writer_nval.flush()

            self.val_cls_pacc_summary.value.add(tag='cls_acc', simple_value=val_cls_pacc/100.)
            self.val_cls_ploss_summary.value.add(tag='cls_loss', simple_value=val_cls_ploss)
            self.writer_pval.add_summary(self.val_cls_pacc_summary, global_step=self.idx)
            self.writer_pval.add_summary(self.val_cls_ploss_summary, global_step=self.idx)
            self.writer_pval.flush()

            self.val_cls_nacc_summary.value.add(tag='cls_acc', simple_value=val_cls_nacc/100.)
            self.val_cls_nloss_summary.value.add(tag='cls_loss', simple_value=val_cls_nloss)
            self.writer_nval.add_summary(self.val_cls_nacc_summary, global_step=self.idx)
            self.writer_nval.add_summary(self.val_cls_nloss_summary, global_step=self.idx)
            self.writer_nval.flush()

            self.saver.save(self.sess, save_path=save_dir + '/weights/model', global_step=self.model_num)
            print "Model saving ... \n model: %d \n save_path: %s" \
                  % (int(self.model_num), save_dir)
            Subject = FLAGS.save_dir
            Text = "description: %s \n"\
                   "step: %d, model_num: %d \n" \
                   "\t train_det_acc: %.2f%%, train_det_loss: %.5f \n" \
                   "\t train_cls_acc: %.2f%%, train_cls_loss: %.5f \n" \
                   "\t train_loss: %.5f, log_var: %.4f,%.4f \n" \
                   "\t val_det_pos_acc: %.2f%%, val_det_pos_loss: %.5f, val_pos_sens: %.2f%% \n" \
                   "\t val_det_neg_acc: %.2f%%, val_det_neg_loss: %.5f, val_neg_spec: %.2f%% \n" \
                   "\t val_cls_pos_acc: %.2f%%, val_cls_pos_loss: %.5f \n" \
                   "\t val_cls_neg_acc: %.2f%%, val_cls_neg_loss: %.5f \n" \
                   "\t auc: %.4f \n" \
                   % (FLAGS.description, self.idx, self.model_num,
                      self.det_acc_sum, self.det_loss_sum,self.cls_acc_sum, self.cls_loss_sum,self.loss_sum,self.log_vars[0],self.log_vars[1],
                      val_det_pacc, val_det_ploss, val_psensitivity, val_det_nacc, val_det_nloss, val_nspecificity,
                      val_cls_pacc, val_cls_ploss, val_cls_nacc, val_cls_nloss, auc_value
                      )

            send_email(Subject=Subject, Text=Text, To='big_respect@naver.com')

            self.model_nums.append(self.model_num)
            self.det_losses.append(self.det_loss_sum)
            self.cls_losses.append(self.cls_loss_sum)
            self.losses.append(self.loss_sum)
            self.train_det_accs.append(round(self.det_acc_sum, 2))
            self.train_cls_accs.append(round(self.cls_acc_sum, 2))
            self.steps.append(self.idx)
            self.val_det_paccs.append(val_det_nacc)
            self.val_cls_paccs.append(val_cls_pacc)
            self.val_det_naccs.append(val_det_nacc)
            self.val_cls_naccs.append(val_cls_nacc)
            self.val_det_plosses.append(val_det_ploss)
            self.val_cls_plosses.append(val_cls_ploss)
            self.val_plosses.append(val_ploss)
            self.val_det_nlosses.append(val_det_nloss)
            self.val_cls_nlosses.append(val_cls_nloss)
            self.val_nlosses.append(val_nloss)
            self.val_psensitivities.append(val_psensitivity)
            self.val_nspecificities.append(val_nspecificity)
            self.val_aucs.append(float(auc_value))

            save_data = [{'model_num': self.model_nums[q],
                          'train_det_acc': self.train_det_accs[q],
                          'train_cls_acc': self.train_cls_accs[q],
                          'train_loss': self.losses[q],
                          'train_det_loss': self.det_losses[q],
                          'train_cls_loss': self.cls_losses[q],
                          'step': self.steps[q],
                          'val_det_pos_acc': self.val_det_paccs[q],
                          'val_cls_pos_acc': self.val_cls_paccs[q],
                          'val_det_neg_acc': self.val_det_naccs[q],
                          'val_cls_neg_acc': self.val_cls_naccs[q],
                          'val_det_pos_loss': self.val_det_plosses[q],
                          'val_cls_pos_loss': self.val_cls_plosses[q],
                          'val_pos_loss': self.val_plosses[q],
                          'val_det_neg_loss': self.val_det_nlosses[q],
                          'val_cls_neg_loss': self.val_cls_nlosses[q],
                          'val_neg_loss': self.val_nlosses[q],
                          'val_pos_sensitivity': self.val_psensitivities[q],
                          'val_neg_specificity': self.val_nspecificities[q],
                          'val_auc': self.val_aucs[q]}
                         for q in range(len(self.steps))]
            with open(save_dir + '/results/record.json', 'wb') as f:
                json.dump(save_data, f)

            self.model_num += 1

        self.det_loss_sum,self.cls_loss_sum,self.loss_sum, self.det_acc_sum, self.cls_acc_sum = 0., 0., 0., 0., 0.
        # self.sum_count = 0

    def res_check_V3(self, is_save=False):

        # print self.sum_count
        self.det_loss_small_sum /= self.res_check_step
        self.det_loss_large_sum /= self.res_check_step
        self.cls_loss_sum /= self.res_check_step
        self.loss_sum /= self.res_check_step
        self.det_acc_small_sum /= self.res_check_step
        self.det_acc_large_sum /= self.res_check_step
        self.cls_acc_sum /= self.res_check_step
        self.log_vars = self.log_vars[0]

        print save_dir, '*' * 20
        print "step: %d, model: %d, det_acc_large: %.2f%%, det_acc_large: %.2f%%, cls_acc: %.2f%%, " \
              "det_loss_small: %.5f, det_loss_large: %.5f, cls_loss: %.5f, loss: %.5f, global_step: %d, lr: %.6f, log vars: %.4f,%.4f,%.4f" % \
              (self.idx, self.model_num, self.det_acc_small_sum, self.det_acc_large_sum,  self.cls_acc_sum,
               self.det_loss_small_sum, self.det_loss_large_sum,self.cls_loss_sum, self.loss_sum, self.sess.run(self.global_step),
               self.sess.run(self.lr), self.log_vars[0], self.log_vars[1], self.log_vars[2])

        self.det_acc_small_summary = tf.Summary()
        self.det_acc_large_summary = tf.Summary()
        self.cls_acc_summary = tf.Summary()
        self.det_loss_small_summary = tf.Summary()
        self.det_loss_large_summary = tf.Summary()
        self.cls_loss_summary = tf.Summary()
        self.loss_summary = tf.Summary()

        self.det_acc_small_summary.value.add(tag='det_acc', simple_value=self.det_acc_small_sum / 100.)
        self.det_acc_small_summary.value.add(tag='det_acc', simple_value=self.det_acc_large_sum / 100.)
        self.cls_acc_summary.value.add(tag='cls_acc', simple_value=self.cls_acc_sum / 100.)
        self.det_loss_small_summary.value.add(tag='det_loss_small', simple_value=self.det_loss_small_sum)
        self.det_loss_large_summary.value.add(tag='det_loss_large', simple_value=self.det_loss_large_sum)
        self.cls_loss_summary.value.add(tag='cls_loss', simple_value=self.cls_loss_sum)
        self.loss_summary.value.add(tag='loss', simple_value=self.loss_sum)
        self.writer_train.add_summary(self.det_acc_small_summary, global_step=self.idx)
        self.writer_train.add_summary(self.det_acc_large_summary, global_step=self.idx)
        self.writer_train.add_summary(self.cls_acc_summary, global_step=self.idx)
        self.writer_train.add_summary(self.det_loss_small_summary, global_step=self.idx)
        self.writer_train.add_summary(self.det_loss_large_summary, global_step=self.idx)
        self.writer_train.add_summary(self.cls_loss_summary, global_step=self.idx)
        self.writer_train.add_summary(self.loss_summary, global_step=self.idx)
        self.writer_train.flush()

        if self.model_num == 0:
            self.saver.save(self.sess, save_path=save_dir + '/weights/model', global_step=self.model_num)

        if is_save:
            val_det_ploss_small, val_det_ploss_large, val_cls_ploss, val_ploss, val_det_pacc_small, val_det_pacc_large, val_cls_pacc, val_psensitivity_small, val_psensitivity_large, \
                pos_cls_prob, pos_cls_label = self.validation_run(is_pos=True)
            print '\t val det pos loss small: %.5f, val det pos loss large: %.5f, val cls pos loss: %.5f, val pos loss: %.5f, ' \
                  'val det pos acc_small: %.2f%%, val det pos acc_large: %.2f%%, val cls pos acc: %.2f%%, val pos sens_small: %.2f%%, val pos sens_large: %.2f%%' % (
            val_det_ploss_small, val_det_ploss_large, val_cls_ploss, val_ploss, val_det_pacc_small, val_det_pacc_large, val_cls_pacc, val_psensitivity_small, val_psensitivity_large)
            val_det_nloss_small, val_det_nloss_large, val_cls_nloss, val_nloss, val_det_nacc_small, val_det_nacc_large, val_cls_nacc, val_nspecificity_small, \
            val_nspecificity_large, neg_cls_prob, neg_cls_label = self.validation_run(is_pos=False)
            print '\t val det neg loss small : %.5f, val det neg loss large: %.5f, val cls neg loss: %.5f, val neg loss: %.5f, ' \
                  'val det neg small acc: %.2f%%, val det neg acc large: %.2f%%, val cls neg acc: %.2f%%, val neg spec small: %.2f%%, val neg spec large: %.2f%%' % (
                      val_det_nloss_small, val_det_nloss_large, val_cls_nloss, val_nloss, val_det_nacc_small, val_det_nacc_large, val_cls_nacc, val_nspecificity_small, val_nspecificity_large)
            label_arr = np.vstack((pos_cls_label,neg_cls_label))
            prob_arr = np.vstack((pos_cls_prob,neg_cls_prob))

            fpr, tpr, thresholds = metrics.roc_curve(label_arr, prob_arr[:, 1])
            auc_value = metrics.auc(fpr, tpr)
            opt_th = find_optimal_threshold(fpr, tpr, thresholds)
            print 'AUC = %0.4f\nopt_th = %0.3f' % (auc_value,opt_th)
            plot_roc_curve(fpr,tpr, FLAGS.save_dir, self.model_num, auc_value, opt_th)

            plt.savefig(save_dir + '/figure/AUC_model_' + str(self.model_num) + '.png')
            plt.close()

            labels = ['benign', 'cancer']
            prob_binary = np.where(prob_arr[:,1]>=0.5,1,0)
            cm = metrics.confusion_matrix(label_arr, prob_binary)
            plot_confusion_matrix(cm,labels,val_cls_pacc, val_cls_nacc, normalize=False,\
                                  title="Model %s Confusion Matrix"%str(self.model_num))
            plt.savefig(save_dir + '/figure/confusion_mat_' + str(self.model_num) + '.png')
            plt.close()

            np.save(save_dir + '/figure/fpr_' + str(self.model_num), fpr)
            np.save(save_dir + '/figure/tpr_' + str(self.model_num), tpr)

            self.val_det_nacc_summary = tf.Summary()
            self.val_cls_nacc_summary = tf.Summary()
            self.val_det_nloss_small_summary = tf.Summary()
            self.val_det_nloss_large_summary = tf.Summary()
            self.val_cls_nloss_summary = tf.Summary()
            self.val_det_pacc_small_summary = tf.Summary()
            self.val_det_pacc_large_summary = tf.Summary()
            self.val_cls_pacc_summary = tf.Summary()
            self.val_det_ploss_small_summary = tf.Summary()
            self.val_det_ploss_large_summary = tf.Summary()
            self.val_cls_ploss_summary = tf.Summary()
            self.val_auc_summary = tf.Summary()
            self.val_sens_small_summary = tf.Summary()
            self.val_sens_large_summary = tf.Summary()
            self.val_spec_small_summary = tf.Summary()
            self.val_spec_large_summary = tf.Summary()

            self.val_auc_summary.value.add(tag='auc', simple_value=auc_value
                                           )
            self.writer_val.add_summary(self.val_auc_summary, global_step=self.idx)
            self.writer_val.flush()

            self.val_sens_small_summary.value.add(tag='sensitivity_small', simple_value=val_psensitivity_small)
            self.writer_pval.add_summary(self.val_sens_small_summary, global_step=self.idx)
            self.writer_pval.flush()

            self.val_spec_small_summary.value.add(tag='specificity_small', simple_value=val_nspecificity_small)
            self.writer_nval.add_summary(self.val_spec_small_summary, global_step=self.idx)
            self.writer_nval.flush()

            self.val_sens_large_summary.value.add(tag='sensitivity_large', simple_value=val_psensitivity_large)
            self.writer_pval.add_summary(self.val_sens_large_summary, global_step=self.idx)
            self.writer_pval.flush()

            self.val_spec_large_summary.value.add(tag='specificity_large', simple_value=val_nspecificity_large)
            self.writer_nval.add_summary(self.val_spec_large_summary, global_step=self.idx)
            self.writer_nval.flush()

            self.val_det_pacc_small_summary.value.add(tag='det_acc_small', simple_value=val_det_pacc_small/100.)
            self.val_det_pacc_large_summary.value.add(tag='det_acc_large', simple_value=val_det_pacc_large/100.)
            self.val_det_ploss_small_summary.value.add(tag='det_loss', simple_value=val_det_ploss_small)
            self.val_det_ploss_large_summary.value.add(tag='det_loss', simple_value=val_det_ploss_large)
            self.writer_pval.add_summary(self.val_det_pacc_small_summary, global_step=self.idx)
            self.writer_pval.add_summary(self.val_det_pacc_large_summary, global_step=self.idx)
            self.writer_pval.add_summary(self.val_det_ploss_small_summary, global_step=self.idx)
            self.writer_pval.add_summary(self.val_det_ploss_large_summary, global_step=self.idx)
            self.writer_pval.flush()

            self.val_det_nacc_summary.value.add(tag='det_acc_small', simple_value=val_det_nacc_small/100.)
            self.val_det_nacc_summary.value.add(tag='det_acc_large', simple_value=val_det_nacc_large
                                                                                  /100.)
            self.val_det_nloss_small_summary.value.add(tag='det_loss', simple_value=val_det_nloss_small)
            self.val_det_nloss_large_summary.value.add(tag='det_loss', simple_value=val_det_nloss_large)
            self.writer_nval.add_summary(self.val_det_nacc_summary, global_step=self.idx)
            self.writer_nval.add_summary(self.val_det_nloss_small_summary, global_step=self.idx)
            self.writer_nval.add_summary(self.val_det_nloss_large_summary, global_step=self.idx)
            self.writer_nval.flush()

            self.val_cls_pacc_summary.value.add(tag='cls_acc', simple_value=val_cls_pacc/100.)
            self.val_cls_ploss_summary.value.add(tag='cls_loss', simple_value=val_cls_ploss)
            self.writer_pval.add_summary(self.val_cls_pacc_summary, global_step=self.idx)
            self.writer_pval.add_summary(self.val_cls_ploss_summary, global_step=self.idx)
            self.writer_pval.flush()

            self.val_cls_nacc_summary.value.add(tag='cls_acc', simple_value=val_cls_nacc/100.)
            self.val_cls_nloss_summary.value.add(tag='cls_loss', simple_value=val_cls_nloss)
            self.writer_nval.add_summary(self.val_cls_nacc_summary, global_step=self.idx)
            self.writer_nval.add_summary(self.val_cls_nloss_summary, global_step=self.idx)
            self.writer_nval.flush()

            self.saver.save(self.sess, save_path=save_dir + '/weights/model', global_step=self.model_num)
            print "Model saving ... \n model: %d \n save_path: %s" \
                  % (int(self.model_num), save_dir)
            Subject = FLAGS.save_dir
            Text = "description: %s \n"\
                   "step: %d, model_num: %d \n" \
                   "\t train_det_acc_small: %.2f%%, train_det_acc_large: %.2f%%, \n" \
                   "\t train_det_loss_small: %.5f, train_det_loss_large: %.5f \n" \
                   "\t train_cls_acc: %.2f%%, train_cls_loss: %.5f \n" \
                   "\t train_loss: %.5f, log vars: %4f,%4f,%4f \n" \
                   "\t val_det_pos_acc_small: %.2f%%, val_det_pos_acc_large: %.2f%%, \n" \
                   "\t val_det_pos_loss_small: %.5f, val_det_pos_loss_large: %.5f, \n" \
                   "\t val_pos_sens_small: %.2f%% , val_pos_sens_large: %.2f%% \n" \
                   "\t val_det_neg_acc_small: %.2f%%, val_det_neg_acc_large: %.2f%%, \n " \
                   "\t val_det_neg_loss_small: %.5f, val_det_neg_loss: %.5f, \n" \
                   "\t val_neg_spec_small: %.2f%% , val_neg_spec_large: %.2f%% \n" \
                   "\t val_cls_pos_acc: %.2f%%, val_cls_pos_loss: %.5f \n" \
                   "\t val_cls_neg_acc: %.2f%%, val_cls_neg_loss: %.5f \n" \
                   "\t auc: %.4f" \
                   % (FLAGS.description, self.idx, self.model_num,
                      self.det_acc_small_sum, self.det_acc_large_sum, self.det_loss_small_sum,
                      self.det_loss_large_sum, self.cls_acc_sum, self.cls_loss_sum,self.loss_sum, self.log_vars[0], self.log_vars[1],self.log_vars[2],
                      val_det_pacc_small, val_det_pacc_large, val_det_ploss_small, val_det_ploss_large,
                      val_psensitivity_small, val_psensitivity_large, val_det_nacc_small, val_det_nacc_large,
                      val_det_nloss_small, val_det_nloss_large, val_nspecificity_small, val_nspecificity_large,
                      val_cls_pacc, val_cls_ploss, val_cls_nacc, val_cls_nloss, auc_value
                     )

            send_email(Subject=Subject, Text=Text, To='big_respect@naver.com')

            self.model_nums.append(self.model_num)
            self.det_losses_small.append(self.det_loss_small_sum)
            self.det_losses_large.append(self.det_loss_large_sum)
            self.cls_losses.append(self.cls_loss_sum)
            self.losses.append(self.loss_sum)
            self.train_det_accs_small.append(round(self.det_acc_small_sum, 2))
            self.train_det_accs_large.append(round(self.det_acc_large_sum, 2))
            self.train_cls_accs.append(round(self.cls_acc_sum, 2))
            self.steps.append(self.idx)
            self.val_det_paccs_small.append(val_det_nacc_small)
            self.val_det_paccs_large.append(val_det_nacc_large)
            self.val_cls_paccs.append(val_cls_pacc)
            self.val_det_naccs_small.append(val_det_nacc_small)
            self.val_det_naccs_large.append(val_det_nacc_large)
            self.val_cls_naccs.append(val_cls_nacc)
            self.val_det_plosses_small.append(val_det_ploss_small)
            self.val_det_plosses_large.append(val_det_ploss_large)
            self.val_cls_plosses.append(val_cls_ploss)
            self.val_plosses.append(val_ploss)
            self.val_det_nlosses_small.append(val_det_nloss_small)
            self.val_det_nlosses_large.append(val_det_nloss_large)
            self.val_cls_nlosses.append(val_cls_nloss)
            self.val_nlosses.append(val_nloss)
            self.val_psensitivities_small.append(val_psensitivity_small)
            self.val_psensitivities_large.append(val_psensitivity_large)
            self.val_nspecificities_small.append(val_nspecificity_small)
            self.val_nspecificities_large.append(val_nspecificity_large)
            self.val_aucs.append(float(auc_value))

            save_data = [{'model_num': self.model_nums[q],
                          'train_det_acc_small': self.train_det_accs_small[q],
                          'train_det_acc_large': self.train_det_accs_large[q],
                          'train_cls_acc': self.train_cls_accs[q],
                          'train_loss': self.losses[q],
                          'train_det_loss_small': self.det_losses_small[q],
                          'train_det_loss_large': self.det_losses_large[q],
                          'train_cls_loss': self.cls_losses[q],
                          'step': self.steps[q],
                          'val_det_pos_acc_small': self.val_det_paccs_small[q],
                          'val_det_pos_acc_large': self.val_det_paccs_large[q],
                          'val_cls_pos_acc': self.val_cls_paccs[q],
                          'val_det_neg_acc_small': self.val_det_naccs_small[q],
                          'val_det_neg_acc_large': self.val_det_naccs_large[q],
                          'val_cls_neg_acc': self.val_cls_naccs[q],
                          'val_det_pos_loss_small': self.val_det_plosses_small[q],
                          'val_det_pos_loss_large': self.val_det_plosses_large[q],
                          'val_cls_pos_loss': self.val_cls_plosses[q],
                          'val_pos_loss': self.val_plosses[q],
                          'val_det_neg_loss_small': self.val_det_nlosses_small[q],
                          'val_det_neg_loss_large': self.val_det_nlosses_large[q],
                          'val_cls_neg_loss': self.val_cls_nlosses[q],
                          'val_neg_loss': self.val_nlosses[q],
                          'val_pos_sensitivity_small': self.val_psensitivities_small[q],
                          'val_pos_sensitivity_large': self.val_psensitivities_large[q],
                          'val_neg_specificity_small': self.val_nspecificities_small[q],
                          'val_neg_specificity_large': self.val_nspecificities_large[q],
                          'val_auc': self.val_aucs[q]}
                         for q in range(len(self.steps))]
            with open(save_dir + '/results/record.json', 'wb') as f:
                json.dump(save_data, f)

            self.model_num += 1

        self.det_loss_small_sum, self.det_acc_large_sum, self.cls_loss_sum, self.loss_sum, self.det_acc_large_sum, \
        self.det_acc_small_sum, self.cls_acc_sum = 0., 0., 0., 0., 0., 0., 0.

    def validation_run(self, is_pos=True):
        if is_pos:
            cur_list = self.valid_pimg
            cls_label = self.pv_cls_label
        else:
            cur_list = self.valid_nimg
            cls_label = self.nv_cls_label
        cnt = 0
        cls_prob_arr = np.zeros((1, 2))
        cls_label_arr = np.zeros((1, 1))
        if self.network in ['V3_dense', 'V3_res']:
            val_det_loss_small_sum, val_det_loss_large_sum, val_cls_loss_sum, val_loss_sum, val_det_acc_small_sum, \
            val_det_acc_large_sum, val_cls_acc_sum, val_metric_small, val_metric_large = 0., 0., 0., 0., 0., 0., 0., 0., 0
        else:
            val_det_loss_sum, val_cls_loss_sum, val_loss_sum, val_det_acc_sum, val_cls_acc_sum, val_metric = 0., 0., 0., 0., 0., 0.
        for step in range(0, len(cur_list), self.batch * FLAGS.num_gpu):
            cnt += 1
            sys.stdout.write('val_step: %d/%d\r'%(step,len(cur_list)))
            sys.stdout.flush()
            feed_dict = {}
            if len(cur_list) - step < self.batch * FLAGS.num_gpu:
                cur_batch = int(len(cur_list) - step)
                cur_cls_label = cls_label[:cur_batch]
                if self.data_type == 'mass' or self.data_type == 'final':
                    if self.network in ['V3_dense', 'V3_res']:
                        cur_det_nlbl_small = self.det_nvlbl_small[:cur_batch, ...]
                        cur_det_nlbl_large = self.det_nvlbl_large[:cur_batch, ...]
                    else:
                        cur_det_nlbl = self.det_nvlbl[:cur_batch,...]
            else:
                cur_batch = self.batch
                cur_cls_label = cls_label
                if self.data_type == 'mass' or self.data_type == 'final':
                    if self.network in ['V3_dense', 'V3_res']:
                        cur_det_nlbl_small = self.det_nvlbl_small
                        cur_det_nlbl_large = self.det_nvlbl_large
                    else:
                        cur_det_nlbl = self.det_nvlbl
            for gpu in self.gpu_num:
                feed_dict[self.towers['img' + str(gpu)]] = self.npy_load(cur_list[step + cur_batch * gpu:
                                                                             step + cur_batch * (gpu + 1)])
                feed_dict[self.towers['cls_label' + str(gpu)]] = cur_cls_label
                feed_dict[self.towers['is_training' + str(gpu)]] = False
                if self.loss_weight == 'DWA':
                    feed_dict[self.towers['wk' + str(gpu)]] = self.wk

                if is_pos:
                    if self.network in ['V3_dense', 'V3_res']:
                        feed_dict[self.towers['det_label_small' + str(gpu)]] = self.npy_load_lbl_small(
                            self.valid_det_plbl[step + gpu * cur_batch:step + (gpu + 1) * cur_batch])
                        feed_dict[self.towers['det_label_large' + str(gpu)]] = self.npy_load_lbl_large(
                            self.valid_det_plbl[step + gpu * cur_batch:step + (gpu + 1) * cur_batch])
                    else:
                        feed_dict[self.towers['det_label' + str(gpu)]] = self.npy_load_lbl(
                            self.valid_det_plbl[step + gpu * cur_batch:step + (gpu + 1) * cur_batch])
                else:
                    if self.network in ['V3_dense', 'V3_res']:
                        if self.data_type == 'mass' or self.data_type == 'final':
                            feed_dict[self.towers['det_label_small' + str(gpu)]] = cur_det_nlbl_small
                            feed_dict[self.towers['det_label_large' + str(gpu)]] = cur_det_nlbl_large
                        elif self.data_type == 'cancer':
                            feed_dict[self.towers['det_label_small' + str(gpu)]] = self.npy_load_lbl_small(
                                self.valid_det_nlbl[step + gpu * cur_batch:step + (gpu + 1) * cur_batch])
                            feed_dict[self.towers['det_label_large' + str(gpu)]] = self.npy_load_lbl_large(
                                self.valid_det_nlbl[step + gpu * cur_batch:step + (gpu + 1) * cur_batch])
                    else:
                        if self.data_type == 'mass' or self.data_type == 'final':
                            feed_dict[self.towers['det_label' + str(gpu)]] = cur_det_nlbl
                        elif self.data_type == 'cancer':
                            feed_dict[self.towers['det_label' + str(gpu)]] = self.npy_load_lbl(
                                self.valid_det_nlbl[step + gpu * cur_batch:step + (gpu + 1) * cur_batch])

                if cur_batch ==1:
                    feed_dict[self.towers['expand_dims' + str(gpu)]] = True
                else:
                    feed_dict[self.towers['expand_dims' + str(gpu)]] = False

            cur_res = self.sess.run(self.valFetches, feed_dict=feed_dict)
            if self.network in ['V3_dense', 'V3_res']:
                val_det_loss_small_sum += np.mean(cur_res[::10])
                val_det_loss_large_sum += np.mean(cur_res[1::10])
                val_cls_loss_sum += np.mean(cur_res[2::10])
                val_loss_sum += np.mean(cur_res[3::10])
                val_det_acc_small_sum += np.mean(cur_res[4::10]) * 100
                val_det_acc_large_sum += np.mean(cur_res[5::10]) * 100
                val_cls_acc_sum += np.mean(cur_res[6::10]) * 100
                cls_probs = np.array(cur_res[9::10])
            else:
                val_det_loss_sum += np.mean(cur_res[::7])
                val_cls_loss_sum += np.mean(cur_res[1::7])
                val_loss_sum += np.mean(cur_res[2::7])
                val_det_acc_sum += np.mean(cur_res[3::7]) * 100
                val_cls_acc_sum += np.mean(cur_res[4::7]) * 100
                cls_probs = np.array(cur_res[6::7])

            probs_reshp = np.reshape(cls_probs, (-1, 2))
            cls_prob_arr = np.vstack((cls_prob_arr, probs_reshp))
            for gpu in self.gpu_num:
                cls_label_arr = np.vstack((cls_label_arr, cur_cls_label))
            if self.network in ['V3_dense', 'V3_res']:
                if is_pos:
                    for gpu in self.gpu_num:
                        pred_small = np.argmax(cur_res[7 + gpu * 10], axis=-1) * feed_dict[
                            self.towers['det_label_small' + str(gpu)]]
                        pred_small = np.sum(np.reshape(pred_small, newshape=[self.batch, -1]), axis=-1)
                        pred_small = np.where(pred_small >= 1, 1., 0.)
                        val_metric_small += np.mean(pred_small[:]) * 100

                        pred_large = np.argmax(cur_res[8 + gpu * 10], axis=-1) * feed_dict[
                            self.towers['det_label_large' + str(gpu)]]
                        pred_large = np.sum(np.reshape(pred_large, newshape=[self.batch, -1]), axis=-1)
                        pred_large = np.where(pred_large >= 1, 1., 0.)
                        val_metric_large += np.mean(pred_large[:]) * 100
                else:
                    for gpu in self.gpu_num:
                        pred_small = np.argmax(cur_res[7 + gpu * 10], axis=-1)
                        pred_small = np.sum(np.reshape(pred_small, newshape=[self.batch, -1]), axis=-1)
                        pred_small = np.where(pred_small > 0, 1., 0.)
                        val_metric_small += (1 - np.mean(pred_small)) * 100

                        pred_large = np.argmax(cur_res[8 + gpu * 10], axis=-1)
                        pred_large = np.sum(np.reshape(pred_large, newshape=[self.batch, -1]), axis=-1)
                        pred_large = np.where(pred_large > 0, 1., 0.)
                        val_metric_large += (1 - np.mean(pred_large)) * 100
            else:
                if is_pos:
                    for gpu in self.gpu_num:
                        pred = np.argmax(cur_res[5 + gpu * 7], axis=-1) * feed_dict[self.towers['det_label' + str(gpu)]]
                        pred = np.sum(np.reshape(pred, newshape=[self.batch, -1]), axis=-1)
                        pred = np.where(pred >= 1, 1., 0.)
                        val_metric += np.mean(pred[:]) * 100
                else:
                    for gpu in self.gpu_num:
                        pred = np.argmax(cur_res[5 + gpu * 7], axis=-1)
                        pred = np.sum(np.reshape(pred, newshape=[self.batch, -1]), axis=-1)
                        pred = np.where(pred > 0, 1., 0.)
                        val_metric += (1-np.mean(pred)) * 100

        if self.network in ['V3_dense', 'V3_res']:
            val_det_loss_small_sum /= cnt
            val_det_loss_large_sum /= cnt
            val_cls_loss_sum /= cnt
            val_loss_sum /= cnt
            val_det_acc_small_sum /= cnt
            val_det_acc_large_sum /= cnt
            val_cls_acc_sum /= cnt
            val_metric_small /= cnt * FLAGS.num_gpu
            val_metric_large /= cnt * FLAGS.num_gpu

            return round(val_det_loss_small_sum, 5), round(val_det_loss_large_sum, 5), round(val_cls_loss_sum, 5), \
                   round(val_loss_sum, 5), round(val_det_acc_small_sum, 2), round(val_det_acc_large_sum, 2), \
                   round(val_cls_acc_sum, 2), round(val_metric_small, 2), round(val_metric_large, 2), cls_prob_arr[1:], \
                   cls_label_arr[1:]
        else:
            val_det_loss_sum /= cnt
            val_cls_loss_sum /= cnt
            val_loss_sum /= cnt
            val_det_acc_sum /= cnt
            val_cls_acc_sum /= cnt
            val_metric /= cnt * FLAGS.num_gpu

            return round(val_det_loss_sum, 5), round(val_cls_loss_sum, 5), round(val_loss_sum, 5), \
                   round(val_det_acc_sum, 2), round(val_cls_acc_sum, 2), round(val_metric, 2), \
                    cls_prob_arr[1:], cls_label_arr[1:]

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    detection = Model(loss_weight=FLAGS.loss_weight, network=FLAGS.network)
    detection.dataset(label_type=FLAGS.label_type+'_new', data_type=FLAGS.data_type)
    detection.tf_model()
    detection.run_model()