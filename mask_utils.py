from utility_class import plot_roc_curve
from utils.sql_utils import Database
from utility_class import data_load_spacing, data_load_woedge, data_load_woedge_spacing
from glob import glob
from ABUS_utility import data_load
import numpy as np
import json

def plot_rocs():
    import matplotlib,datetime
    matplotlib.use('Agg')
    # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    save_dir = '/data/dk/exp/MaskAttention/'
    model_name = 'MASKV3_EXP01'
    model_no = '79'
    fpr = np.load('/data/dk/exp/MaskAttention/'+model_name+'/figure/fpr_valid79.npy')
    tpr = np.load('/data/dk/exp/MaskAttention/'+model_name+'/figure/tpr_valid79.npy')
    plot_roc_curve(fpr, tpr, model_name)

    model_name = 'MASKV3_EXP02'
    fpr = np.load('/data/dk/exp/MaskAttention/'+model_name+'/figure/fpr_valid79.npy')
    tpr = np.load('/data/dk/exp/MaskAttention/'+model_name+'/figure/tpr_valid79.npy')
    plot_roc_curve(fpr, tpr, model_name)

    plt.savefig('/data/dk/exp/MaskAttention/roc_test_'+datetime.datetime.now().strftime('%y%m%d_%H%M')+'.png')
    plt.close()

def set_neg_data():
    split_list = ['valid','train']
    db = Database()
    keys = ['caseType','iskind', 'patientID', 'studyDate',
            'studyUID', 'seriesUID', 'sopUID','intervals',
            'spacingX', 'spacingY','viewName', 'split', 'cropType',
            'coordZ', 'coordY', 'coordX', 'id', 'diameter1', 'diameter2', 'diameter3','score']
    crop_save_dir = '/data/dk/datasets_CROPS/crops_ABUS_rescale'
    for split in split_list:
        if split == 'valid':
            num_limit = 15
            PIDs = ['13389932','10214116','11305363','10858136']
        elif split == 'train':
            num_limit = 100
            PIDs = ['31150288','31120720','30706323','21523168','19280824','19280824','14310883']

        npy_list = []

        for PID in PIDs:
            # print PID
            try:

                rows = db.select("select " + ','.join(keys) + " from crops where cropType in ('negative')"
                                                              "and patientID = '%s' and split='%s'"%(PID,split))
            except Exception as e:
                print e

            rows = sorted(list(rows))
            # print len(rows)
            # raw_input()
            # continue
            for ridx,row in enumerate(rows,1):
                cur_info = {keys[i]:row[i] for i in range(len(keys))}

                if ridx>num_limit: continue
                try:
                    db.update("update crops set dataType = '%s' where id = %s"%(split,cur_info['id']))
                except Exception as e:
                    print e

                crop_dir = '/'.join(
                    [crop_save_dir, cur_info['split'], cur_info['cropType'], cur_info['score'],
                     cur_info['iskind'], str(cur_info['patientID']),
                     str(cur_info['studyDate']), cur_info['studyUID'], cur_info['seriesUID'],
                     cur_info['sopUID']])
                crop_name = '_'.join(
                    [str(cur_info['viewName']), str(round(cur_info['coordZ'], 3)), str(round(cur_info['coordY'], 3)),
                     str(round(cur_info['coordX'], 3)),
                     str(cur_info['intervals']), str(cur_info['spacingY']), str(cur_info['spacingX'])])
                # print crop_dir
                npy = glob(crop_dir+'/'+crop_name+'*.npy')
                if len(npy)==0:
                    print crop_dir, crop_name
                    raw_input()
                # print len(npy)
                npy_list+= npy
        print split,'npy list:',len(npy_list)

def set_FP_data():
    db = Database()
    keys = ['caseType','iskind', 'patientID', 'studyDate',
            'studyUID', 'seriesUID', 'sopUID','intervals',
            'spacingX', 'spacingY','viewName', 'split', 'cropType',
            'coordZ', 'coordY', 'coordX', 'id', 'diameter1', 'diameter2', 'diameter3','score']
    crop_save_dir = '/data/dk/datasets_CROPS/crops_ABUS_rescale'
    num_limit = 10
    valid_PIDs = ['26130460','15445807','25916762']
    try:
        PIDs = db.select("select distinct patientID from crops where cropType in ('FP_integ_1805')")
    except Exception as e:
        print e
    npy_dict = {'valid':[],'train':[]}

    for PID in PIDs:
        PID = PID[0]
        if PID in valid_PIDs:
            split = 'valid'
        else:
            split= 'train'
        try:
            rows = db.select("select " + ','.join(keys) + " from crops where cropType in ('FP_integ_1805')"
                                                          "and patientID = '%s' and split = '%s'"%(PID,split))
        except Exception as e:
            print e

        rows = sorted(list(rows))
        # print len(rows)
        # raw_input()
        # continue
        for ridx,row in enumerate(rows,1):
            cur_info = {keys[i]:row[i] for i in range(len(keys))}
            if ridx>num_limit: continue
            try:
                db.update("update crops set dataType = '%s' where id = %s"%(split,cur_info['id']))
            except Exception as e:
                print e

            crop_dir = '/'.join(
                [crop_save_dir, cur_info['split'], cur_info['cropType'], cur_info['score'],
                 cur_info['iskind'], str(cur_info['patientID']),
                 str(cur_info['studyDate']), cur_info['studyUID'], cur_info['seriesUID'],
                 cur_info['sopUID']])
            crop_name = '_'.join(
                [str(cur_info['viewName']), str(round(cur_info['coordZ'], 3)), str(round(cur_info['coordY'], 3)),
                 str(round(cur_info['coordX'], 3)),
                 str(cur_info['intervals']), str(cur_info['spacingY']), str(cur_info['spacingX'])])
            # print crop_dir
            npy = glob(crop_dir+'/'+crop_name+'*.npy')
            if len(npy)==0:
                print crop_dir, crop_name
                raw_input()
            # print len(npy)
            npy_dict[split] += npy
    print 'valid',len(npy_dict['valid'])
    print 'train',len(npy_dict['train'])

def set_pos_valid_data():
    db = Database()
    plbl = data_load_spacing(
        '/data/dk/datasets_CROPS/crops_ABUS_lbl_rescale_small_rev/valid/TP/*/*/*/*/*/*/*')
    plbl += data_load_spacing(
        '/data/dk/datasets_CROPS/crops_ABUS_lbl_rescale_small_rev/valid/benign/*/*/*/*/*/*/*')
    plbl += data_load(
        '/data/dk/datasets_CROPS/crops_ABUS_lbl_orientation/train/TP/*/*/*/*/*/*/*')
    plbl += data_load(
        '/data/dk/datasets_CROPS/crops_ABUS_lbl_orientation/train/benign/*/*/*/*/*/*/*')
    plbl = sorted(plbl)
    prev_coordZ = ''
    print len(plbl)
    for lidx, lbl in enumerate(plbl,1):
        if lidx%100==0:
            print lidx,'/',len(plbl)
        dir_list = lbl.split('/')
        split = dir_list[5]
        cropType = dir_list[6]
        sopUID = dir_list[-2]
        cropName = dir_list[-1]
        cropName = cropName.replace('.npy','')
        name_list = cropName.split('_')
        viewName = name_list[0]
        coordZ = name_list[1]
        coordY = name_list[2]
        coordX = name_list[3]
        if coordZ == prev_coordZ:
            continue
        else:
            prev_coordZ = coordZ
        # print split,cropType,cropName,sopUID
        # print viewName,coordZ,coordY,coordX
        # raw_input()

        try:
            db.update("update crops set dataType = '%s' where sopUID = '%s' and coordZ=%s and coordY=%s and"
                      " coordX=%s" % (split, sopUID, coordZ,coordY,coordX))
        except Exception as e:
            print e

def load_dataset(label_mode):
    label_dir = 'crops_ABUS_lbl_'+label_mode
    data_dir = '/data/dk/datasets_CROPS/'
    t_plbl,t_pos, t_neg,v_plbl, v_pos,v_neg=[],[],[],[],[],[]
    db = Database()
    keys = ['iskind', 'patientID', 'studyDate',
            'studyUID', 'seriesUID', 'sopUID', 'intervals',
            'spacingX', 'spacingY', 'viewName', 'split', 'cropType',
            'coordZ', 'coordY', 'coordX', 'id', 'diameter1', 'diameter2', 'diameter3', 'score','dataType']
    try:
        rows = db.select("select " + ','.join(keys) + " from crops where dataType in ('train','valid')")
    except Exception as e:
        print e
    for row in rows:
        cur_info = {keys[i]: row[i] for i in range(len(keys))}
        if cur_info['cropType'] in ['benign','TP']:
            if cur_info['split'] == 'train':
                crop_save_dir = data_dir + label_dir
            elif cur_info['split'] == 'valid':
                crop_save_dir = data_dir + 'crops_ABUS_lbl_rescale_small_rev'
        elif cur_info['cropType'] in ['negative','FP_integ_1805']:
            crop_save_dir = data_dir + 'crops_ABUS_rescale'
        else:
            raw_input('cropType error')
        crop_dir = '/'.join(
            [crop_save_dir, cur_info['split'], cur_info['cropType'], cur_info['score'],
             cur_info['iskind'], str(cur_info['patientID']),
             str(cur_info['studyDate']), cur_info['studyUID'], cur_info['seriesUID'],
             cur_info['sopUID']])
        crop_name = '_'.join(
            [str(cur_info['viewName']), str(round(cur_info['coordZ'], 3)), str(round(cur_info['coordY'], 3)),
             str(round(cur_info['coordX'], 3)),
             str(cur_info['intervals']), str(cur_info['spacingY']), str(cur_info['spacingX'])])
        if cur_info['dataType'] == 'train':
            npy = glob(crop_dir + '/' + crop_name + '*.npy')
        elif cur_info['dataType'] == 'valid':
            npy = glob(crop_dir + '/' + crop_name + '*_1.0_0.npy')
            npy += glob(crop_dir + '/' + crop_name + '*_1.0_0_012.npy')
        else:
            raw_input('wrong dataType')
        if len(npy) == 0:
            print cur_info
            print crop_dir, crop_name
            raw_input()
        if cur_info['cropType'] in ['benign','TP']:
            if cur_info['dataType'] == 'train':
                t_plbl+=npy
            elif cur_info['dataType'] == 'valid':
                v_plbl+=npy
        elif cur_info['cropType'] in ['negative','FP_integ_1805']:
            if cur_info['dataType'] == 'train':
                t_neg+=npy
            elif cur_info['dataType'] == 'valid':
                v_neg+=npy
    t_pos = [pimg.replace(label_dir, "crops_ABUS_rescale") for pimg in
                       t_plbl]
    t_pos = [pimg.replace('crops_ABUS_lbl_rescale_small_rev', "crops_ABUS_rescale") for pimg in
                       t_pos]
    v_pos = [pimg.replace('crops_ABUS_lbl_rescale_small_rev', "crops_ABUS_rescale") for pimg in
                       v_plbl]
    print len(t_plbl), len(t_neg), len(v_plbl), len(v_neg)

    return t_plbl,t_pos, t_neg,v_plbl, v_pos,v_neg


def load_mass_train_data(label_type):
    '''

    :param label_type:
    :return:
    '''
    train_det_plbl = data_load('/data/dk/datasets_CROPS/crops_ABUS_lbl_' + label_type + '/train/TP/*/*/*/*/*/*/*')
    train_det_plbl += data_load(
        '/data/dk/datasets_CROPS/crops_ABUS_lbl_' + label_type + '/train/oversize/*/*/*/*/*/*/*')
    train_det_plbl += data_load(
        '/data/dk/datasets_CROPS/crops_ABUS_lbl_' + label_type + '/train/benign/*/*/*/*/*/*/*')
    train_pimg = [pimg.replace("crops_ABUS_lbl_" + label_type, "crops_ABUS_rescale") for pimg in train_det_plbl]
    train_nimg = data_load('/data/dk/datasets_CROPS/crops_ABUS_rescale/train/negative/*/*/*/*/*/*/*')
    train_nimg += data_load('/data/dk/datasets_CROPS/crops_ABUS_rescale/train/nipple/*/*/*/*/*/*')
    train_nimg += data_load_woedge('/data/dk/datasets_CROPS/crops_ABUS_rescale/train/FP_integ_1805/*/*/*/*/*/*/*')

    return train_det_plbl, train_pimg, train_nimg


def load_mass_valid_data(label_type):
    with open('/data/dk/datasets_CROPS/mass_valid_pos.json', 'r') as file:
        valid_pimg = json.load(file)
    valid_det_plbl = [pimg.replace("crops_ABUS_lbl_rescale_small_rev", "crops_ABUS_rescale") for pimg in
                      valid_pimg]

    with open('/data/dk/datasets_CROPS/mass_valid_neg.json', 'r') as file:
        valid_nimg = json.load(file)
    return valid_det_plbl, valid_pimg, valid_nimg

def load_final_train_data(label_type):
    '''
    cancer or not
    :param label_type:
    :return:
    '''
    train_det_plbl_bio, train_det_nlbl_bio, _, _ = biopsy_data_setting(
        '/data/dk/datasets_CROPS/crops_ABUS_lbl_' + label_type)

    train_det_plbl = data_load('/data/dk/datasets_CROPS/crops_ABUS_lbl_' + label_type + '/train/TP/6/*/*/*/*/*/*')
    train_det_plbl += data_load(
        '/data/dk/datasets_CROPS/crops_ABUS_lbl_' + label_type + '/train/oversize/6/*/*/*/*/*/*')
    train_det_plbl += train_det_plbl_bio
    train_pimg = [pimg.replace("crops_ABUS_lbl_" + label_type, "crops_ABUS_rescale") for pimg in train_det_plbl]

    train_nimg = data_load('/data/dk/datasets_CROPS/crops_ABUS_rescale/train/negative/*/*/*/*/*/*/*')
    train_nimg += data_load('/data/dk/datasets_CROPS/crops_ABUS_rescale/train/benign/2/*/*/*/*/*/*')
    train_nimg += data_load('/data/dk/datasets_CROPS/crops_ABUS_rescale/train/nipple/*/*/*/*/*/*')
    train_nimg += data_load_woedge('/data/dk/datasets_CROPS/crops_ABUS_rescale/train/FP_integ_1805/*/*/*/*/*/*/*')

    train_nimg_bio = [nimg.replace("crops_ABUS_lbl_" + label_type, "crops_ABUS_rescale") for nimg in train_det_nlbl_bio]
    train_nimg += train_nimg_bio
    return train_det_plbl, train_pimg, train_nimg


def load_final_valid_data(label_type):
    with open('/data/dk/datasets_CROPS/final_valid_pos.json', 'r') as file:
        valid_pimg = json.load(file)
    valid_det_plbl = [pimg.replace("crops_ABUS_lbl_rescale_small_rev", "crops_ABUS_rescale") for pimg in
                      valid_pimg]

    with open('/data/dk/datasets_CROPS/final_valid_neg.json', 'r') as file:
        valid_nimg = json.load(file)
    return valid_det_plbl, valid_pimg, valid_nimg

def load_mass_whole_valid_data(label_type):
    valid_det_plbl = data_load_spacing(
        '/data/dk/datasets_CROPS/crops_ABUS_lbl_rescale_small_rev/valid/TP/*/*/*/*/*/*/*')
    valid_det_plbl += data_load_spacing(
        '/data/dk/datasets_CROPS/crops_ABUS_lbl_rescale_small_rev/valid/oversize/*/*/*/*/*/*/*')
    valid_det_plbl += data_load_spacing(
        '/data/dk/datasets_CROPS/crops_ABUS_lbl_rescale_small_rev/valid/benign/*/*/*/*/*/*/*')
    valid_pimg = [pimg.replace("crops_ABUS_lbl_rescale_small_rev", "crops_ABUS_rescale") for pimg in
                       valid_det_plbl]
    ng = data_load_spacing('/data/dk/datasets_CROPS/crops_ABUS_rescale/valid/negative/*/*/*/*/*/*/*')
    ng = ng[::2]
    fp = data_load_woedge_spacing(
        '/data/dk/datasets_CROPS/crops_ABUS_rescale/valid/FP_integ_1805/*/*/*/*/*/*/*')
    fp = fp[::2]
    valid_nimg = ng + fp

    return valid_det_plbl, valid_pimg, valid_nimg

def biopsy_data_setting(crops_dir):
    db=Database()
    t_pos, t_neg, v_pos, v_neg = [], [], [], []
    for bx in [0,1]:
        for split in ['train','valid']:
            cur_list = []
            try:
                rows = db.select("select split, cropType, score, iskind, patientID, studyDate ,studyUID, seriesUID,"
                                   "sopUID from crops where biopsy = '%s' and split = '%s'"%(str(bx),split))
            except Exception as e:
                print 'select error :',e

            for row in rows:
                cur_path = '/'.join([crops_dir,row[0],row[1],row[2],row[3],row[4],str(row[5]),row[6],row[7],row[8]])
                cur_npys = glob(cur_path+'/*.npy')
                cur_list+=cur_npys

            # print split,bx,len(cur_list)
            if bx == 1 and split == 'train':
                t_pos = cur_list
            elif bx == 0 and split == 'train':
                t_neg = cur_list
            elif bx == 1 and split == 'valid':
                v_pos = cur_list
            elif bx ==0 and split == 'valid':
                v_neg = cur_list

    return t_pos,t_neg,v_pos,v_neg