import matplotlib
# matplotlib.use('Agg')
matplotlib.rcParams['figure.dpi'] = 200
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def plot_roc_curve(fpr,tpr,label_name,auc_value=0., opt_th=0., marker='o'):
    plt.plot(fpr, tpr, marker=marker, label=label_name)
    plt.legend(fontsize=18)
    plt.title('Mask attention loss', fontsize=20)
    plt.legend(loc='lower right')
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

def plot_roc_checkpoint(expno,modelno,label_name,split='valid',marker='o'):
    fpr = np.load('/data/dk/exp/MaskAttention/MASKV5_'+expno+'/figure/fpr_'+split+str(modelno)+'.npy')
    tpr = np.load('/data/dk/exp/MaskAttention/MASKV5_'+expno+'/figure/tpr_'+split+str(modelno)+'.npy')
    plot_roc_curve(fpr, tpr, label_name,marker=marker)

# fpr = np.load('/data/dk/exp/MaskAttention/MASKV5_EXP01/figure/fpr_valid56.npy')
# tpr = np.load('/data/dk/exp/MaskAttention/MASKV5_EXP01/figure/tpr_valid56.npy')
# plot_roc_curve(fpr,tpr,'DenseNet')
# fpr = np.load('/data/dk/exp/MaskAttention/MASKV5_EXP02/figure/fpr_valid52.npy')
# tpr = np.load('/data/dk/exp/MaskAttention/MASKV5_EXP02/figure/tpr_valid52.npy')
# plot_roc_curve(fpr,tpr,'DenseNet+MALV1')
# fpr = np.load('/data/dk/exp/MaskAttention/MASKV5_EXP03/figure/fpr_valid39.npy')
# tpr = np.load('/data/dk/exp/MaskAttention/MASKV5_EXP03/figure/tpr_valid39.npy')
# plot_roc_curve(fpr,tpr,'DenseNet+MALV2')
# fpr = np.load('/data/dk/exp/MaskAttention/MASKV5_EXP04/figure/fpr_valid38.npy')
# tpr = np.load('/data/dk/exp/MaskAttention/MASKV5_EXP04/figure/tpr_valid38.npy')
# plot_roc_curve(fpr,tpr,'DenseNet+MALV3')
# fpr = np.load('/data/dk/exp/MaskAttention/MASKV5_EXP05/figure/fpr_valid47.npy')
# tpr = np.load('/data/dk/exp/MaskAttention/MASKV5_EXP05/figure/tpr_valid47.npy')
# plot_roc_curve(fpr,tpr,'DenseNet+MALV4')
# plt.show()
# plt.close()

# plot_roc_checkpoint(expno='EXP01',modelno=56,label_name='DenseNet', marker='.')
# plot_roc_checkpoint(expno='EXP02',modelno=52,label_name='DenseNet+MBNV1', marker='*')
# plot_roc_checkpoint(expno='EXP03',modelno=39,label_name='DenseNet+MBNV2', marker='o')
# plot_roc_checkpoint(expno='EXP04',modelno=38,label_name='DenseNet+MBNV3', marker='s')
# plot_roc_checkpoint(expno='EXP05',modelno=47,label_name='DenseNet+MBNV4', marker='x')
# plt.savefig('/data/dk/images/ROC_mass_densenet_rev04.png',dpi=300)

# plot_roc_checkpoint(expno='EXP01',modelno=56,label_name='DenseNet', marker='.')
# plot_roc_checkpoint(expno='EXP02',modelno=52,label_name='DenseNet+MBN', marker='*')

# plot_roc_checkpoint(expno='EXP16',modelno=40,label_name='ResNet', marker='.')
# plot_roc_checkpoint(expno='EXP17_2',modelno=40,label_name='ResNet+MBNV1', marker='*')
# plot_roc_checkpoint(expno='EXP18_2',modelno=39,label_name='ResNet+MBNV2', marker='o')
# plot_roc_checkpoint(expno='EXP19_2',modelno=42,label_name='ResNet+MBNV3', marker='s')
# plot_roc_checkpoint(expno='EXP20_2',modelno=43,label_name='ResNet+MBNV4', marker='x')
# plt.savefig('/data/dk/images/ROC_mass_resnet_rev04.png',dpi=300)

# plot_roc_checkpoint(expno='EXP16',modelno=40,label_name='ResNet', marker='.')
# plot_roc_checkpoint(expno='EXP17_2',modelno=40,label_name='ResNet+MBN', marker='*')

# plot_roc_checkpoint(expno='EXP06',modelno=58,label_name='equal', marker='.')
# plot_roc_checkpoint(expno='EXP02',modelno=52,label_name='uncertainty', marker='*')
# plot_roc_checkpoint(expno='EXP07',modelno=65,label_name='reviced uncertainty', marker='s')
# plot_roc_checkpoint(expno='EXP08',modelno=27,label_name='DWA', marker='o')
# plt.savefig('/data/dk/images/ROC_loss_weighting_rev04.png',dpi=300)


# plot_roc_checkpoint(expno='EXP01',modelno=56,label_name='no mask', marker='.')
# plot_roc_checkpoint(expno='EXP09',modelno=56,label_name='TM-V1', marker='o')
# plot_roc_checkpoint(expno='EXP10',modelno=39,label_name='TM-V2', marker='s')
# plot_roc_checkpoint(expno='EXP02',modelno=52,label_name='TM-V3', marker='*')
# plot_roc_checkpoint(expno='EXP11',modelno=43,label_name='TM-V4', marker='x')
# plot_roc_checkpoint(expno='EXP12',modelno=53,label_name='TM-V5', marker='+')
# plt.savefig('/data/dk/images/ROC_label_case_rev04.png',dpi=300)

##### cancer
# plot_roc_checkpoint(expno='EXP21_1',modelno=35,label_name='DenseNet',split='test', marker='o')
# plot_roc_checkpoint(expno='EXP22',modelno=49,label_name='DenseNet+MALV1', marker='*')
# plot_roc_checkpoint(expno='EXP23',modelno=50,label_name='DenseNet+MALV2', marker='s')
# plot_roc_checkpoint(expno='EXP24',modelno=53,label_name='DenseNet+MALV3', marker='.')
# plot_roc_checkpoint(expno='EXP25',modelno=36,label_name='DenseNet+MALV4', marker='x')

####### MICCAI
plot_roc_checkpoint(expno='EXP01',modelno=56,label_name='DenseNet', marker='.')
plot_roc_checkpoint(expno='EXP03',modelno=39,label_name='DenseNet+MBN', marker='o')

# plot_roc_checkpoint(expno='EXP16',modelno=40,label_name='ResNet', marker='.')
# plot_roc_checkpoint(expno='EXP18_2',modelno=39,label_name='ResNet+MBN', marker='o')

plt.show()
plt.close()