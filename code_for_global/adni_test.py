from posixpath import dirname
import time, os, argparse, shutil
import numpy as np
import pandas as pd
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



def test(cosine_loss,save_path,DEVICE,node_num,feat_dim,hidden_num,class_num,nheads,dropout,data_path):

    class_num = 2
    
    TestLoader = get_test_dataloader(node_num, batch_size,data_path)

    # model = GCN(feat_dim=2, node_num=116, assign_ratio=0.5, class_num=3).to(DEVICE)
    test_model = GAN(feat_dim=feat_dim, node_num=node_num, hidden_num=hidden_num, class_num=class_num,nheads=nheads,dropout=dropout).to(DEVICE)
    test_model.load_state_dict(torch.load(save_path)["state_dict"])
    # model = GraphConvModel(feat_dim=3, node_num=116, assign_ratio=0.5, class_num=2).to(DEVICE)
    # model = GAT(nfeat=3, nhid=3, nclass=2).to(DEVICE)

    test_loss, test_acc, test_auc, test_preds, test_preds_score, test_targets, test_risks,_,_,_,test_score_pre,test_score_rec,test_score_f1,test_dirs=test_epoch(cosine_loss,test_model,TestLoader,DEVICE)


    return test_acc, test_auc, test_preds, test_preds_score, test_targets, test_risks, test_dirs


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0', help='gpu')
    parser.add_argument('-seed', type=int, default=664, help='Random seed.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # plt.figure(figsize=(6, 3))
    # plt.figure(2)
    cosine_loss = False
    node_num = 20
    feat_dim = 2048
    hidden_num = 512
    nheads=8
    class_num = 2
    dropout = 0
    batch_size=1
    save_path='/mnt/data/zhangxin/MICCAI/GAT/output/加cos_初始化全1_seed2021_lr4.0e-05_lr_decay4.0e-05_wd0.0001_node20_hidden_num512_bs128_nheads8_dropout0.2-001/fold2/seed2021_lr4.0e-05_lr_decay4.0e-05_wd0.0001_node20_hidden_num512_bs128_nheads8_dropout0.2-001_best_score.pth'
    # save_path='/mnt/data/zhangxin/MICCAI/GAT/output/不加cos_初始化全1_seed2021_lr1.0e-03_lr_decay1.0e-05_wd0.0001_node20_hidden_num512_bs128_nheads8_dropout0-000/fold5/seed2021_lr1.0e-03_lr_decay1.0e-05_wd0.0001_node20_hidden_num512_bs128_nheads8_dropout0-000_best_score.pth'
    
    score_acc, score_auc, preds, preds_score, targets, risks,dirs = test(cosine_loss,save_path,DEVICE,node_num,feat_dim,hidden_num,class_num,nheads,dropout,'data_csv/427.csv')
    print('427:\n','score_acc:',format(score_acc,'.4f'),'score_auc:', format(score_auc,'.4f'))
    matrix = confusion_matrix(targets,preds)
    print('confusion_matrix:\n',matrix)
    risk = 0
    for i,num in enumerate(targets):
        if targets[i] == 1:
            if risks[i]==2 and preds[i]==0:
                risk+=1
                print(dirs[i])
    print('高级别漏诊：',risk)
    print('特异性：',matrix[0][0]/(matrix[0][0]+matrix[0][1]))
    # ax_427 = plt.subplot(121)
    # ax_427 = plt.plot()
    # ax_427.set_title('427')
    plt.figure()
    plot_confusion_matrix(matrix)
    plt.savefig('result_3.png')
    # ax_roc_curve = plt.subplot(122)
    # ax_roc_curve = plt.plot()
    # ax_roc_curve.set_title('roc curve')
    plt.figure()
    fpr_total,tpr_total,threshold_total = roc_curve(targets,preds_score)
    score_auc_total = roc_auc_score(targets,preds_score)
    plt.plot(fpr_total,tpr_total,color='blue',linestyle='-', lw=1,alpha=.8,label=f'AUC={score_auc_total:.4f}')
    plt.legend()
    plt.savefig('result_4.png')
    
    


