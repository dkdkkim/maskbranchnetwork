import sys
sys.path.append('/home/dkkim/workplace/Multitask/abus')

import os
from models.model_densenet_V2 import convmodel
import sys
import tensorflow as tf
import numpy as np
# matplotlib.use("Agg")

from ABUS_utility import data_load
from utility_class import  data_load_spacing, npy_load, data_load_woedge_spacing, fig_3views_cam, npy_load_cube

flags = tf.app.flags
flags.DEFINE_integer("num_gpu", 1, help="Number of GPUs")
flags.DEFINE_integer("batch", 2, help="batch_size")

flags.DEFINE_string("save_dir", "/data/dk/exp/ABUS_nodule_class/exp_test", help="Directory name to save the weights and records")
flags.DEFINE_string("CUDA_VISIBLE_DEVICES", "2", help="GPU number")
flags.DEFINE_string("mlpBackend", "TkAgg", help="GPU number")
FLAGS = flags.FLAGS

mlpBack = FLAGS.mlpBackend
import matplotlib
matplotlib.use(mlpBack)
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.CUDA_VISIBLE_DEVICES

class Model():
    def __init__(self,exp_name,model_no):
        self.exp_name = exp_name
        self.model_no = model_no
        self.fig_save_dir = '/'.join(['/data/dk/images/GRAD_CAM_jet',self.exp_name+'_'+self.model_no+'_cube'])
        print self.fig_save_dir
        if not os.path.exists(self.fig_save_dir):
            os.makedirs(self.fig_save_dir+'/pos')
            os.makedirs(self.fig_save_dir+'/neg')

        self.gpu_num = list(range(FLAGS.num_gpu))
        self.batch, self.hbatch = FLAGS.batch, FLAGS.batch / 2
        self.res_check_step = 20
        self.val_psensitivities = []
        self.pvals, self.nvals, self.accs, self.pvloss, self.nvloss, self.modelnums, self.losses, self.starts = [], [], [], [], [], [], [], []

    def tower_loss(self, scope):

        model = convmodel()
        image, is_training, prob, conv = model.convcam()
        losses = tf.get_collection('losses', scope)
        total_loss = tf.add_n(losses, name='total_loss')

        return image, is_training, prob, conv, total_loss

    def dataset(self):

        self.valid_det_plbl = data_load_spacing(
            '/data/dk/datasets_CROPS/crops_ABUS_lbl_rescale_small_rev/valid/TP/*/*/*/*/*/*/*')
        self.valid_det_plbl += data_load_spacing(
            '/data/dk/datasets_CROPS/crops_ABUS_lbl_rescale_small_rev/valid/oversize/*/*/*/*/*/*/*')
        self.valid_det_plbl += data_load_spacing(
            '/data/dk/datasets_CROPS/crops_ABUS_lbl_rescale_small_rev/valid/benign/*/*/*/*/*/*/*')
        self.v_pos = [pimg.replace("crops_ABUS_lbl_rescale_small_rev", "crops_ABUS_rescale") for pimg in
                           self.valid_det_plbl]
        print 'valid positive data loaded'
        ng = data_load_spacing('/data/dk/datasets_CROPS/crops_ABUS_rescale/valid/negative/*/*/*/*/*/*/*')
        ng = ng[::2]
        fp = data_load_woedge_spacing(
            '/data/dk/datasets_CROPS/crops_ABUS_rescale/valid/FP_integ_1805/*/*/*/*/*/*/*')
        fp = fp[::2]
        self.v_neg = ng+fp
        self.pv_label, self.nv_label = np.ones([self.batch, 1]), np.zeros([self.batch, 1])
        print 'pos %d, neg %d' % (len(self.v_pos), len(self.v_neg))

    def tf_model(self):

        with tf.Graph().as_default():
            self.global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)

            self.towers = {}

            with tf.variable_scope(tf.get_variable_scope()):
                model = convmodel(loss_weight='uncertainty')
                # model = convmodel()
                self.image, self.label, self.is_training, self.prob, self.conv = model.convcam()
                tf.get_variable_scope().reuse_variables()
            label = tf.one_hot(indices=self.label, depth=2, on_value=1.0, off_value=0.0)
            cost = (-1) * tf.reduce_sum(tf.multiply(tf.cast(label,dtype=tf.float32), tf.log(self.prob)), axis=1)
            # y_c = tf.reduce_sum(tf.multiply(self.prob, tf.cast(self.label,dtype=tf.float32)), axis=1)
            # y_c = tf.reduce_sum(self.prob, axis=1)
            y_c = tf.reduce_sum(tf.multiply(self.prob,label), axis=1)
            # conv_grad = tf.gradients(self.prob, self.conv)[0]
            conv_grad = tf.gradients(y_c, self.conv)[0]
            gb_grad = tf.gradients(cost, self.image)[0]

            self.saver = tf.train.Saver(max_to_keep=1000)

            self.sess = tf.Session(config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True)))
            print '/'.join(['/data/dk/exp/MaskAttention',self.exp_name,'weights','model-'+self.model_no])
            self.saver.restore(self.sess,'/'.join(['/data/dk/exp/MaskAttention',self.exp_name,'weights','model-'+self.model_no]))

        self.valFetches = []
        for gpu in self.gpu_num:
            self.valFetches = [gb_grad, conv_grad, self.conv, self.prob]

    def visualize(self, image, conv, conv_grad, gb_grad):
        from scipy.ndimage.interpolation import zoom

        # print 'grad shape :', conv_grad.shape #6x3x6x256
        # print 'gb_grad shape :', gb_grad.shape

        weights = np.mean(conv_grad, axis = (0,1,2,)) #256
        cam = np.zeros(conv.shape[0:3], dtype=np.float32) #6x3x6

        ## all feature
        # for i, w in enumerate(weights):
        #     print w
        #     cam += w * conv[...,i]

        ## top k feature
        topk = np.argsort(weights)[::-1][:10]
        for i in topk:
            cam += weights[i] * conv[...,i]

        # cam = np.maximum(cam,0)

        cam = zoom(cam, np.array([8., 8., 8.]))
        # cam = zoom(cam, np.array([4., 4., 4.]))
        # cam = zoom(cam, np.array([2., 2., 2.]))
        cam -= np.min(cam)
        cam = cam / np.max(cam)
        print cam.shape, image.shape
        fig_3views_cam(image,cam,mlpBack=FLAGS.mlpBackend)



    def GradCam(self, is_pos):

        if is_pos:
            npy = self.v_pos
            label = self.pv_label
            cur_save_dir = self.fig_save_dir+'/'+'pos'
        else:
            npy = self.v_neg
            label = self.nv_label
            cur_save_dir = self.fig_save_dir+'/'+'neg'

        val_acc, val_loss = 0, 0

        cnt=0
        for vstart in range(0,len(npy),self.batch):
            if len(npy) - vstart < self.batch: continue

            cnt+=1

            sys.stdout.write('val_step: %d/%d\r'%(vstart,len(npy)))

            feed_dict = {}
            feed_dict[self.image] = npy_load_cube(npy[vstart : vstart + self.batch])
            feed_dict[self.label] = label
            feed_dict[self.is_training] = False

            cur_gb_grad, cur_conv_grad, cur_conv, cur_prob = self.sess.run(self.valFetches, feed_dict=feed_dict)
            for i in range(self.batch):
                cur_path = npy[vstart + i]
                cur_sopUID = cur_path.split('/')[-2]
                cur_name = cur_path.split('/')[-1]
                cur_name = cur_name.replace('.npy','')
                self.visualize(np.squeeze(feed_dict[self.image][i],axis=-1), cur_conv[i], cur_conv_grad[i], cur_gb_grad[i])
                plt.savefig(cur_save_dir+'/'+cur_sopUID + '_' + cur_name+'.png')

                plt.close()

        val_loss /= cnt
        val_acc /= cnt

        if is_pos:
            print "pos_val_loss: %f, pos_val_acc: %.2f%%" % (val_loss, val_acc)
        else:
            print "neg_val_loss: %f, neg_val_acc: %.2f%%" % (val_loss, val_acc)


def main():

    model = Model(exp_name='MASKV4_EXP03',model_no='33')
    model.dataset()
    model.tf_model()
    model.GradCam(is_pos=True)

if __name__ == '__main__':
    main()
