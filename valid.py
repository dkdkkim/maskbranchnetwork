import sys
sys.path.append('/home/dkkim/workplace/Multitask/abus')

from models.model_densenet_DWA import convmodel as convmodel_multiV1_dense_DWA
from models.model_densenet_V1 import convmodel as convmodel_multiV1_dense
from models.model_densenet_V2 import convmodel as convmodel_multiV2_dense
from models.model_densenet_V3 import convmodel as convmodel_multiV3_dense
from models.model_densenet_V4 import convmodel as convmodel_multiV4_dense
from models.model_resnet_V1 import convmodel as convmodel_multiV1_res
from models.model_resnet_V2 import convmodel as convmodel_multiV2_res
from models.model_resnet_V3 import convmodel as convmodel_multiV3_res
from models.model_resnet_V4 import convmodel as convmodel_multiV4_res
from utility_class import biopsy_data_setting
import matplotlib
matplotlib.use("Agg")
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os, math, datetime, scipy, json
import tensorflow as tf
import numpy as np
from glob import glob

from utility_class import fig_3views, data_load_origin, npy_load, data_load_woedge_spacing, data_load_spacing, \
    plot_roc_curve, find_optimal_threshold


flags = tf.app.flags
flags.DEFINE_integer("num_gpu", 1, help="Number of GPUs")
flags.DEFINE_integer("batch", 48, help="batch_size")
flags.DEFINE_float("lr_init", 0.01, help="lr init")
flags.DEFINE_string("save_dir", "/data/dk/exp/ABUS_nodule_class/exp_test", help="Directory name to save the weights and records")
flags.DEFINE_string("CUDA_VISIBLE_DEVICES", "2", help="GPU number")
flags.DEFINE_string("fig_dir","fig_test",help="Directory name to save the figures")
flags.DEFINE_string("network","V1_dense", help="mask network used for training")
flags.DEFINE_string("loss_weight","weight test", help="how to weigh loss")
flags.DEFINE_string("data_type","mass", help="how to weigh loss")
flags.DEFINE_string("exp_no","exp_no test", help="how to weigh loss")
flags.DEFINE_string("model_no","model_no test", help="how to weigh loss")
flags.DEFINE_string("split","valid", help="how to weigh loss")

FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.CUDA_VISIBLE_DEVICES

class Model():
    def __init__(self,exp_no,model_no,network,loss_weight):
        self.network = network
        self.loss_weight = loss_weight
        self.exp_no = exp_no
        self.model_no = model_no
        self.model_name=self.exp_no+'/weights/model-'+self.model_no
        self.gpu_num = list(range(FLAGS.num_gpu))
        self.batch, self.hbatch = FLAGS.batch, FLAGS.batch / 2
        self.res_check_step = 20
        self.val_psensitivities = []
        self.pvals, self.nvals, self.accs, self.pvloss, self.nvloss, self.modelnums, self.losses, self.starts = [], [], [], [], [], [], [], []

    def npy_load(self,npy_path):
        x = np.empty(shape=[len(npy_path), 48, 32, 48])

        for idx, cur_path in enumerate(npy_path):
            cur_arr = np.load(cur_path)
            shp= cur_arr.shape
            cur_arr = cur_arr[shp[0]/2-24:shp[0]/2+24,shp[1]/2-16:shp[1]/2+16, shp[2]/2-24:shp[2]/2+24]

            x[idx] = cur_arr

        return np.expand_dims(x, axis=4)

    def data_load_origin(self, *args):
        dirs = []
        for idx, arg in enumerate(args):
            if isinstance(arg, list):
                for cur in arg:
                    if cur[-1] == '/': cur = cur[:-1]
                    dirs += glob(cur + '/*_1.0_0.npy')

            else:
                if arg[-1] == '/': arg = arg[:-1]
                dirs += glob(arg + '/*_1.0_0.npy')

        if len(dirs) == 0:
            print "Please check dirs (len(dir)==0)"
            raw_input("Enter")

        return dirs

    def tower_loss(self, scope, th=0.5):
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
        elif self.network == 'res':
            model = convmodel_res()
        elif self.network == 'dense':
            model = convmodel_dense()
        image, label, loss, acc, is_training, expand_dims, prob, conv = model.convval(th)

        losses = tf.get_collection('losses', scope)
        total_loss = tf.add_n(losses, name='total_loss')

        return image, label, loss, acc, is_training, expand_dims, prob, conv, total_loss

    def dataset(self, mode, split):
        # ################ mass
        if mode == 'mass' and split=='whole':
            self.valid_det_plbl = data_load_spacing(
                '/data/dk/datasets_CROPS/crops_ABUS_lbl_rescale_small_rev/valid/TP/*/*/*/*/*/*/*')
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
            self.v_neg = fp+ng
        elif mode == 'cancer' and split=='whole':
            self.t_pos, self.t_neg, self.v_pos, self.v_neg = biopsy_data_setting('/data/dk/datasets_CROPS/crops_ABUS_rescale')
        else:
            with open('/data/dk/datasets_CROPS/'+mode+'_'+split+'_pos.json','r') as file:
                self.v_pos=json.load(file)
            with open('/data/dk/datasets_CROPS/'+mode+'_'+split+'_neg.json','r') as file:
                self.v_neg=json.load(file)

        self.vpList, self.vnList = [i for i in range(len(self.v_pos))], [i for i in
                                                                         range(len(self.v_neg))]
        print 'valid negative data loaded'

        self.pv_label, self.nv_label = np.ones([self.batch, 1]), np.zeros([self.batch, 1])
        print 'pos %d, neg %d' % (len(self.v_pos), len(self.v_neg))

    def tf_model(self,th):

        with tf.Graph().as_default():
            self.global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)

            self.towers = {}

            with tf.variable_scope(tf.get_variable_scope()):
                for gpu in self.gpu_num:
                    with tf.device('/gpu:%d' % gpu):
                        with tf.name_scope('%s_%d' % ('tower', gpu)) as scope:
                            num = str(gpu)
                            self.towers['img' + num], self.towers['label' + num], self.towers['loss' + num], \
                            self.towers['tacc' + num], self.towers['is_training' + num], self.towers['expand_dims' + num], \
                            self.towers['prob' + num], self.towers['conv' + num], self.total_loss \
                                = self.tower_loss(scope,th)
                            tf.get_variable_scope().reuse_variables()

            self.saver = tf.train.Saver(max_to_keep=1000)

            self.sess = tf.Session(config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True)))

            self.saver.restore(self.sess,'/data/dk/exp/MaskAttention/' + self.model_name)



        self.valFetches = []
        for gpu in self.gpu_num:
            self.valFetches += [self.towers['loss' + str(gpu)], self.towers['tacc' + str(gpu)],
                                self.towers['prob' + str(gpu)]]

    def res_check(self):
        import sklearn.metrics as metrics

        save_dir = '/data/dk/exp/MaskAttention'+'/'+self.exp_no
        ploss, pacc, plabel, pprob = self.validation(is_pos=True)

        print "pos_val_loss: %f, pos_val_acc: %.2f%%" % (ploss, pacc)
        nloss, nacc, nlabel, nprob= self.validation(is_pos=False)

        print "neg_val_loss: %f, neg_val_acc: %.2f%%" % (nloss, nacc)

        label_arr = np.vstack((plabel, nlabel))
        prob_arr = np.vstack((pprob, nprob))
        print label_arr.shape, prob_arr.shape
        fpr, tpr, thresholds = metrics.roc_curve(label_arr,prob_arr[...,1])
        auc_value = metrics.auc(fpr, tpr)
        opt_th = find_optimal_threshold(fpr, tpr, thresholds)

        sens = np.where(pprob[...,1]>=opt_th,1.,0.)
        spec = np.where(nprob[...,1]<opt_th,1.,0.)
        print 'optimal',np.mean(sens)*100,np.mean(spec)*100
        sens = np.where(pprob[...,1]>=0.5,1.,0.)
        spec = np.where(nprob[...,1]<0.5,1.,0.)

        print '0.5 th',np.mean(sens)*100,np.mean(spec)*100#,np.mean(sss)
        print 'AUC = %0.4f\nopt_th = %0.3f' % (auc_value, opt_th)
        plot_roc_curve(fpr, tpr, self.model_no, auc_value, opt_th)
        plt.savefig(save_dir + '/figure/AUC_model_valid' + str(self.model_no) + '.png')
        plt.close()

        np.save(save_dir + '/figure/fpr_' +FLAGS.split+ str(self.model_no), fpr)
        np.save(save_dir + '/figure/tpr_' +FLAGS.split+ str(self.model_no), tpr)

    def validation(self, is_pos):

        if is_pos:
            npy = self.v_pos
            label = self.pv_label
        else:
            npy = self.v_neg
            label = self.nv_label
        val_acc, val_loss, val_sensitivity = 0, 0, 0
        prob_arr = np.zeros((1,2))
        label_arr = np.zeros((1,1))
        cnt=0
        for vstart in range(0,len(npy),self.batch * len(self.gpu_num)):

            cnt+=1

            sys.stdout.write('val_step: %d/%d\r'%(vstart,len(npy)))

            feed_dict = {}
            if len(npy) - vstart < self.batch * FLAGS.num_gpu:
                cur_batch = int(len(npy) - vstart)
                cur_label = label[:cur_batch]
            else:
                cur_batch = self.batch
                cur_label = label
            for gpu in self.gpu_num:
                feed_dict[self.towers['img' + str(gpu)]] = self.npy_load(npy[vstart + cur_batch * gpu:
                                                                             vstart + cur_batch * (gpu + 1)])
                feed_dict[self.towers['label' + str(gpu)]] = cur_label
                feed_dict[self.towers['is_training' + str(gpu)]] = False
                if cur_batch ==1:
                    feed_dict[self.towers['expand_dims' + str(gpu)]] = True
                else:
                    feed_dict[self.towers['expand_dims' + str(gpu)]] = False


            cur_res = self.sess.run(self.valFetches, feed_dict=feed_dict)

            val_loss += np.mean(cur_res[::3])
            val_acc += np.mean(cur_res[1::3])

            probs = np.array(cur_res[2::3])
            probs = np.reshape(probs, (-1, 2))
            prob_arr = np.vstack((prob_arr, probs))
            for gpu in self.gpu_num:
                label_arr = np.vstack((label_arr, cur_label))

        val_loss /= cnt
        val_acc /= cnt

        return round(val_loss, 6), round(val_acc * 100, 2),label_arr[1:,],prob_arr[1:,]


def main():
    model = Model(FLAGS.exp_no,FLAGS.model_no,FLAGS.network,FLAGS.loss_weight)
    model.dataset(FLAGS.data_type,FLAGS.split)
    model.tf_model(th=0.5)
    model.res_check()

if __name__ == '__main__':
    main()
