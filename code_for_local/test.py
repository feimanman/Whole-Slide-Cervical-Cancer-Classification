import time, os, argparse, shutil
import numpy as np
import json 
from glob import glob
import pandas as pd
import torch.nn.functional as F
import torch
import torch.optim as optim
from glob import glob
from adni_GraphDataset import get_test_dataloader
from net_utils_for_test import test_epoch
from sklearn.metrics import roc_curve
from adni_tools import get_logger, visualize_training_history, draw_attentionmap,plot_confusion_matrix
from model.adni_model_GA_DIFFPOOL import GAN
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,roc_auc_score
from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.seterr(divide='ignore',invalid='ignore')
import warnings
warnings.filterwarnings("ignore")
def openjson(path):
    with open(path, 'r')as f:
        a = json.load(f)
        f.close()
    return a
def writejson(path, f1):
    with open(path, 'w')as f:
        json.dump(f1, f)
        f.close()
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_decay = 0.0001
    lr = 3e-4
    lr_decay = 4e-5
    train_epochs = 500
    risk=111
    node_num = 5
    feat_dim = 2048
    hidden_num = 512
    class_num = 2
    nheads = 8
    dropout = 0.2
    
    model = GAN(feat_dim=feat_dim, node_num=node_num, hidden_num=hidden_num, class_num=class_num,nheads=nheads).to(DEVICE)
    save_path= '/mnt/data/zhangxin/Journal/code_for_local/experiment/seed2022lr3.0e-044.0e-05_epoch500_wd0.0001_node5_feature2048_hidden_num512_bs512_nheads8_dropout0.2-002/seed2022lr3.0e-044.0e-05_epoch500_wd0.0001_node5_feature2048_hidden_num512_bs512_nheads8_dropout0.2-002_best_score.pth'
    model.load_state_dict(torch.load(save_path)["state_dict"])
    model.eval()

    for sample in os.listdir('/mnt/data/zhangxin/Journal/data_for_global/npy_data')[7500:]:
        sample_dir = '/mnt/data/zhangxin/Journal/data_for_global/npy_data'+'/'+sample
        print(sample)
        for tile in os.listdir(sample_dir):
            # print(tile)
            tile_dir = sample_dir+'/'+tile
            json_p = glob(os.path.join(tile_dir, '*.json'))[0]
            a=openjson(json_p)
            new = a.items()

            with torch.no_grad():
                fc_matrixs_batch = np.load(tile_dir+'/fcmatrix_5.npy').astype(np.float32)
                fc_matrixs_batch = torch.tensor(fc_matrixs_batch).unsqueeze(0).cuda()
                feature_batch = np.load(tile_dir+'/feature_5.npy').astype(np.float32)
                # print(fc_matrixs_batch.size())
                feature_batch = torch.tensor(feature_batch).unsqueeze(0).cuda()
                preds, link_loss, attentions , ypred_1,ypred_2,ypred_3,ypred_4,ypred_5= model(feature_batch, fc_matrixs_batch)
                # print(ypred_1)
                # print(ypred_2)
                preds_patch_score_1 =  F.softmax(ypred_1, dim=1).detach().cpu().numpy()
                preds_patch_score_2 =  F.softmax(ypred_2, dim=1).detach().cpu().numpy()
                preds_patch_score_3 =  F.softmax(ypred_3, dim=1).detach().cpu().numpy()
                preds_patch_score_4 =  F.softmax(ypred_4, dim=1).detach().cpu().numpy()
                preds_patch_score_5 =  F.softmax(ypred_5, dim=1).detach().cpu().numpy()
                # print(float(preds_patch_score_1[:,1]))
                score = []
                # print(preds_patch_score_1[:,1])
                # print(preds_patch_score_2[:,1])
                score.append(preds_patch_score_1[:,1])
                score.append(preds_patch_score_2[:,1])
                score.append(preds_patch_score_3[:,1])
                score.append(preds_patch_score_4[:,1])
                score.append(preds_patch_score_5[:,1])
                # a[imgs[0]] = str(preds_patch_score_1)
                # a[imgs[1]] = str(preds_patch_score_2)
                # a[imgs[2]] = str(preds_patch_score_3)
                # a[imgs[3]] = str(preds_patch_score_4)
                # a[imgs[4]] = str(preds_patch_score_5)
                # print(a[str(imgs[0])])
                for i,imgs in enumerate(new):
                    if imgs[0].endswith('.jpg'):
                        a[imgs[0]] = str(float(score[i]))
                    # print(i)
                # print(a['8_4_72_01.jpg'])
                # print(imgs(0))
                writejson(json_p,a)