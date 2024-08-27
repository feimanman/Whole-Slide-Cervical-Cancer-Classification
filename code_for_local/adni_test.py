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



def test(DEVICE,batch_size,node_num,feat_dim,hidden_num,nheads,save_path,data_path):

    class_num = 2
    class_name = '01'
    TestLoader = get_test_dataloader(node_num, batch_size,data_path)

    # model = GCN(feat_dim=2, node_num=116, assign_ratio=0.5, class_num=3).to(DEVICE)
    model = GAN(feat_dim=feat_dim, node_num=node_num, hidden_num=hidden_num, class_num=class_num,nheads=nheads).to(DEVICE)
    model.load_state_dict(torch.load(save_path)["state_dict"])
    # model = GraphConvModel(feat_dim=3, node_num=116, assign_ratio=0.5, class_num=2).to(DEVICE)
    # model = GAT(nfeat=3, nhid=3, nclass=2).to(DEVICE)

    score_acc, score_auc, preds, preds_score, targets, risks,attentions,dirname_list = test_epoch(model, TestLoader, DEVICE)


    return score_acc, score_auc, preds, preds_score, targets, risks,attentions,dirname_list


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='1', help='gpu')
    parser.add_argument('-seed', type=int, default=664, help='Random seed.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plt.figure(figsize=(12, 3))
    
    node_num = 20
    feat_dim = 2048
    hidden_num = 512
    nheads=8
    batch_size=1
    # save_path='/mnt/data/zhangxin/GAT/output/0.7488risk1210seed2lr4.0e-054.0e-05_epoch100_wd0.0001_node20_feature2048_hidden_num512_bs128_nheads8_dropout0-000/risk1210seed2lr4.0e-054.0e-05_epoch100_wd0.0001_node20_feature2048_hidden_num512_bs128_nheads8_dropout0-000_best_score.pth'
    save_path='/mnt/data/zhangxin/GAT/output/gat2_0.80seed11lr4.0e-054.0e-05_epoch100_wd0.0001_node20_feature2048_hidden_num512_bs128_nheads8_dropout0-000/seed11lr4.0e-054.0e-05_epoch100_wd0.0001_node20_feature2048_hidden_num512_bs128_nheads8_dropout0-000_best_score.pth'
    score_acc, score_auc, preds, preds_score, targets, risks,attentions,dirname_list = test(DEVICE,batch_size,node_num,feat_dim,hidden_num,nheads,save_path,'data_csv/308.csv')
    print('308:\n','score_acc:',format(score_acc,'.4f'),'score_auc:', format(score_auc,'.4f'))
    matrix = confusion_matrix(targets,preds)
    print(matrix)
    risk = 0
    for i,num in enumerate(targets):
        if targets[i] == 1:
            if risks[i]==2 and preds[i]==0:
                risk+=1
                print(dirname_list[i])
    print('高级别漏诊：',risk)
    print('特异性：',matrix[0][0]/(matrix[0][0]+matrix[0][1]))
    # img = ConfusionMatrixDisplay(matrix)
    # img.plot()
    ax_308=plt.subplot(141)
    ax_308.set_title('308')
    plot_confusion_matrix(matrix)
    # total_targets_308 = np.load('total_targets_0950.npy')[:308]
    # total_preds_scores_308 = np.load('total_preds_scores_0950.npy')[:308]
    # fpr_308,tpr_308,threshold_308 = roc_curve(total_targets_308,total_preds_scores_308)
    # score_auc_308 = roc_auc_score(total_targets_308,total_preds_scores_308)
    # plt.plot(fpr_308,tpr_308,color='blue',linestyle='-', lw=1,alpha=.8,label=f'0950(AUC={score_auc_308:.4f})')
    # total_targets_308 = np.load('total_targets_0974.npy')[:308]
    # total_preds_scores_308 = np.load('total_preds_scores_0974.npy')[:308]
    # fpr_308,tpr_308,threshold_308 = roc_curve(total_targets_308,total_preds_scores_308)
    # score_auc_308 = roc_auc_score(total_targets_308,total_preds_scores_308)
    # plt.plot(fpr_308,tpr_308,color='black',linestyle='-', lw=1,alpha=.8,label=f'0974(AUC={score_auc_308:.4f})')
    # fpr_308,tpr_308,threshold_308 = roc_curve(targets,preds_score)
    # plt.plot(fpr_308,tpr_308,color='red',linestyle='-', lw=1,alpha=.8,label=f'GAT(AUC={score_auc:.4f})')
    # plt.legend()




    score_acc, score_auc, preds, preds_score, targets, risks,attentions,dirname_list = test(DEVICE,batch_size,node_num,feat_dim,hidden_num,nheads,save_path,'data_csv/427.csv')
    print('427:\n','score_acc:',format(score_acc,'.4f'),'score_auc:', format(score_auc,'.4f'))
    matrix = confusion_matrix(targets,preds)
    print('confusion_matrix:\n',matrix)
    risk = 0
    for i,num in enumerate(targets):
        if targets[i] == 1:
            if risks[i]==2 and preds[i]==0:
                risk+=1
                print(dirname_list[i])
    print('高级别漏诊：',risk)
    print('特异性：',matrix[0][0]/(matrix[0][0]+matrix[0][1]))
    ax_427 = plt.subplot(142)
    ax_427.set_title('427')
    plot_confusion_matrix(matrix)
    # total_targets_427 = np.load('total_targets_0950.npy')[308:735]
    # total_preds_scores_427 = np.load('total_preds_scores_0950.npy')[308:735]
    # fpr_427,tpr_427,threshold_427 = roc_curve(total_targets_427,total_preds_scores_427)
    # score_auc_427 = roc_auc_score(total_targets_427,total_preds_scores_427)
    # plt.plot(fpr_427,tpr_427,color='blue',linestyle='-', lw=1,alpha=.8,label=f'0950(AUC={score_auc_427:.4f})')
    # total_targets_427 = np.load('total_targets_0974.npy')[308:735]
    # total_preds_scores_427 = np.load('total_preds_scores_0974.npy')[308:735]
    # fpr_427,tpr_427,threshold_427 = roc_curve(total_targets_427,total_preds_scores_427)
    # score_auc_427 = roc_auc_score(total_targets_427,total_preds_scores_427)
    # plt.plot(fpr_427,tpr_427,color='black',linestyle='-', lw=1,alpha=.8,label=f'0974(AUC={score_auc_427:.4f})')
    # fpr_427,tpr_427,threshold_427 = roc_curve(targets,preds_score)
    # plt.plot(fpr_427,tpr_427,color='red',linestyle='-', lw=1,alpha=.8,label=f'GAT(AUC={score_auc:.4f})')
    # plt.legend()




    score_acc, score_auc, preds, preds_score, targets, risks,attentions,dirname_list = test(DEVICE,batch_size,node_num,feat_dim,hidden_num,nheads,save_path,'data_csv/328.csv')
    print('328:\n','score_acc:',format(score_acc,'.4f'),'score_auc:', format(score_auc,'.4f'))
    matrix = confusion_matrix(targets,preds)
    print('confusion_matrix:\n',matrix)
    risk = 0
    for i,num in enumerate(targets):
        if targets[i] == 1:
            if risks[i]==2 and preds[i]==0:
                risk+=1
                print(dirname_list[i])
    print('高级别漏诊：',risk)
    print('特异性：',matrix[0][0]/(matrix[0][0]+matrix[0][1]))
    ax_328=plt.subplot(143)
    ax_328.set_title('328')
    plot_confusion_matrix(matrix)
    # total_targets_328 = np.load('total_targets_0950.npy')[-328:]
    # total_preds_scores_328 = np.load('total_preds_scores_0950.npy')[-328:]
    # fpr_328,tpr_328,threshold_328 = roc_curve(total_targets_328,total_preds_scores_328)
    # score_auc_328 = roc_auc_score(total_targets_328,total_preds_scores_328)
    # plt.plot(fpr_328,tpr_328,color='blue',linestyle='-', lw=1,alpha=.8,label=f'0950(AUC={score_auc_328:.4f})')
    # total_targets_328 = np.load('total_targets_0974.npy')[-328:]
    # total_preds_scores_328 = np.load('total_preds_scores_0974.npy')[-328:]
    # fpr_328,tpr_328,threshold_328 = roc_curve(total_targets_328,total_preds_scores_328)
    # score_auc_328 = roc_auc_score(total_targets_328,total_preds_scores_328)
    # plt.plot(fpr_328,tpr_328,color='black',linestyle='-', lw=1,alpha=.8,label=f'0974(AUC={score_auc_328:.4f})')
    # fpr_328,tpr_328,threshold_328 = roc_curve(targets,preds_score)
    # plt.plot(fpr_328,tpr_328,color='red',linestyle='-', lw=1,alpha=.8,label=f'GAT(AUC={score_auc:.4f})')
    # plt.legend()


 
    score_acc, score_auc, preds, preds_score, targets, risks,attentions,dirname_list = test(DEVICE,batch_size,node_num,feat_dim,hidden_num,nheads,save_path,'data_csv/alltest.csv')
    print('total:\n','score_acc:',format(score_acc,'.4f'),'score_auc:', format(score_auc,'.4f'))
    matrix = confusion_matrix(targets,preds)
    print('confusion_matrix:\n',matrix)
    risk = 0
    for i,num in enumerate(targets):
        if targets[i] == 1:
            if risks[i]==2 and preds[i]==0:
                risk+=1
                print(dirname_list[i])
        # print(dirname_list[i],preds[i])
    print('高级别漏诊：',risk)
    print('特异性：',matrix[0][0]/(matrix[0][0]+matrix[0][1]))
    ax_total = plt.subplot(144)
    ax_total.set_title('total')
    
    plot_confusion_matrix(matrix)
    plt.savefig('result_3.png', transparent=True, dpi=800)
    # total_targets = np.load('total_targets_0950.npy')
    # total_preds_scores = np.load('total_preds_scores_0950.npy')
    # fpr_total,tpr_total,threshold_total = roc_curve(total_targets,total_preds_scores)
    # score_auc_total = roc_auc_score(total_targets,total_preds_scores)
    # plt.plot(fpr_total,tpr_total,color='blue',linestyle='-', lw=1,alpha=.8,label=f'0950(AUC={score_auc_total:.4f})')
    # total_targets = np.load('total_targets_0974.npy')
    # total_preds_scores = np.load('total_preds_scores_0974.npy')
    # fpr_total,tpr_total,threshold_total = roc_curve(total_targets,total_preds_scores)
    # score_auc_total = roc_auc_score(total_targets,total_preds_scores)
    # plt.plot(fpr_total,tpr_total,color='black',linestyle='-', lw=1,alpha=.8,label=f'0974(AUC={score_auc_total:.4f})')
    # fpr,tpr,threshold = roc_curve(targets,preds_score)
    # plt.plot(fpr,tpr,color='red',linestyle='-', lw=1,alpha=.8,label=f'GAT(AUC={score_auc:.4f})')
    # plt.legend()
    # plt.savefig('test1208.png')