import time, os, argparse, shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from glob import glob
from adni_GraphDataset import get_data_loader,get_test_dataloader
from adni_net_utils import train_epoch, val_epoch, test_epoch
from adni_tools import get_logger, visualize_training_history, draw_attentionmap
from model.adni_model_GA_DIFFPOOL import GAN

from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime

np.seterr(divide='ignore',invalid='ignore')
import warnings
warnings.filterwarnings("ignore")

def open_log(log_savepath):
    # log_savepath = os.path.join(log_path, name)
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)
    log_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if os.path.isfile(os.path.join(log_savepath, '{}.log'.format(log_name))):
        os.remove(os.path.join(log_savepath, '{}.log'.format(log_name)))
    initLogging(os.path.join(log_savepath, '{}.log'.format(log_name)))
def increment_dir(dir, e=None):
    # Increments a directory runs/exp1 --> runs/exp2_comment
    n = 0  # number
    d = sorted(glob('experiment/'+dir + '*'))  # directories
    print(d)
    if len(d):
        d = d[-1].split('-')[-1][:3]
        n = int(d) + 1  # increment
        print(n)
    if e is not None:
#        os.makedirs(dir + '-' + str(n).zfill(3) + e, exist_ok=True)
        return dir + '-' + str(n).zfill(3) + e
    else:
#        os.makedirs(dir + '-' + str(n).zfill(3), exist_ok=True)
        return dir + '-' + str(n).zfill(3)
# Init for logging
def initLogging(logFilename):
    logging.basicConfig(level    = logging.INFO,
                        format   = '[%(asctime)s-%(levelname)s] %(message)s',
                        datefmt  = '%y-%m-%d %H:%M:%S',
                        filename = logFilename,
                        filemode = 'w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s-%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def train(seed,DEVICE):
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
    bs = 512
    network_name = increment_dir(f'seed{seed}lr{lr:.1e}{lr_decay:.1e}_epoch{train_epochs}_wd{weight_decay}' \
                   f'_node{node_num}_feature{feat_dim}_hidden_num{hidden_num}_bs{bs}_nheads{nheads}_dropout{dropout}')
    save_path = f'experiment/{network_name}'
    os.makedirs(os.path.join(save_path), exist_ok=True)
    writer1 = SummaryWriter('./resultandinew/log1')
    writer2 = SummaryWriter('./resultandinew/log2')
    open_log(save_path)
    os.system(f'cp -r model {save_path}/code')
    os.system(f'cp  adni_train.py {save_path}/code')
    os.system(f'cp  adni_GraphDataset.py {save_path}/code')
    os.system(f'cp  adni_tools.py {save_path}/code')
    os.system(f'cp  adni_net_utils.py {save_path}/code')

    best_val_loss, is_best_loss, best_test_loss, is_test_best_loss,best_loss_epoch,best_test_loss_epoch = 2 ** 20, False,2 ** 20,False,0, 0
    best_score_acc, is_best_score, best_score_epoch ,best_test_acc,is_test_best_score,best_test_score_epoch= -2 ** 20, False, 0,-2 ** 20, False, 0
    best_score_auc,best_test_score_auc = -2 ** 20,-2 ** 20
    history = pd.DataFrame()

    def save_model(model, optimizer, lr_scheduler, attentions, best_score_acc,is_best_loss=False, is_best_score=False,is_test_best_loss=False,is_test_best_score=False):

        model_state = {'state_dict': model.state_dict(), 'epoch': epoch,
                       'history': history, 
                       'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc,
                       'best_val_loss': best_val_loss, 'best_loss_epoch': best_loss_epoch,
                       'best_score_acc': best_score_acc, 'best_score_epoch': best_score_epoch,
                       'preds': preds, 'targets': targets, 'preds_score': preds_score,
                       'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict()
                       }

        model_path = os.path.join(save_path, f"{network_name}.pth")
        torch.save(model_state, model_path)
        if is_best_loss:
            best_model_path = os.path.join(save_path, f"{network_name}_best_loss.pth")
            shutil.copy(model_path, best_model_path)

            attentionlist = []
            for attention in attentions:
                attentionlist.append(attention.detach().cpu().numpy())
            attentionarray = np.array(attentionlist, dtype=np.float32)
            np.save(os.path.join(save_path, 'attention_best_loss.npy'), attentionarray)
            draw_attentionmap(attentionarray, os.path.join(save_path, 'attention_best_loss.png'))
        if is_best_score:
            best_model_path = os.path.join(save_path, f"{network_name}_best_score.pth")
            shutil.copy(model_path, best_model_path)

            attentionlist = []
            for attention in attentions:
                attentionlist.append(attention.detach().cpu().numpy())
            attentionarray = np.array(attentionlist, dtype=np.float32)
            np.save(os.path.join(save_path, 'attention_best_score.npy'), attentionarray)
            draw_attentionmap(attentionarray, os.path.join(save_path, 'attention_best_score.png'))
        if is_test_best_score:
            test_best_model_path = os.path.join(save_path, f"{network_name}_{best_score_acc:.2f}.pth")
            shutil.copy(model_path, test_best_model_path)
    TrainLoader, ValLoader = get_data_loader( node_num,bs)
    # TestLoader = get_test_dataloader(node_num,1)
    # model = GCN(feat_dim=2, node_num=116, assign_ratio=0.5, class_num=3).to(DEVICE)
    model = GAN(feat_dim=feat_dim, node_num=node_num, hidden_num=hidden_num, class_num=class_num,nheads=nheads,dropout=dropout).to(DEVICE)
    # model = GraphConvModel(feat_dim=3, node_num=116, assign_ratio=0.5, class_num=2).to(DEVICE)
    # model = GAT(nfeat=3, nhid=3, nclass=2).to(DEVICE)

    # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs, eta_min=lr_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=83, gamma=0.5)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20)

    

    for epoch in range(train_epochs):
        cur_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc, train_patch_loss, train_patch_acc = train_epoch(model, TrainLoader, DEVICE, optimizer)
        val_loss, val_acc, score_auc, preds, preds_score, targets, attentions, val_patch_loss, val_patch_acc, val_patch_auc, preds_patch, preds_score_patch, targets_patch = val_epoch(model, ValLoader, DEVICE)
        # test_loss, test_acc, test_score_auc, test_preds, test_preds_score, test_targets, test_risks,test_attentions = test_epoch(model, TestLoader, DEVICE)

        is_best_loss, is_best_score = val_patch_loss < best_val_loss, val_patch_acc > best_score_acc
        best_val_loss, best_score_acc = min(val_patch_loss, best_val_loss), max(val_patch_acc, best_score_acc)

        # is_test_best_loss, is_test_best_score = test_loss < best_test_loss, test_acc > best_test_acc
        # best_test_loss, best_test_acc = min(test_loss, best_test_loss), max(test_acc, best_test_acc)
        
        lr_scheduler.step()

        _h = pd.DataFrame(
            {'lr': [cur_lr], 'train_loss': [train_loss], 'train_acc': [train_acc], 'val_loss': [val_loss],'val_acc':[val_acc],
             'train_patch_loss':[train_patch_loss],'train_patch_acc':[train_patch_acc],'val_patch_loss':[val_patch_loss],'val_patch_acc':[val_patch_acc],})
        """ _h = pd.DataFrame(
            {'lr': [cur_lr], 'train_loss': [train_loss], 'train_acc': [train_acc], 'val_loss': [val_loss], 'val_acc': [val_acc],
            'test_loss':[test_loss],'test_acc':[test_acc]}) """
        history = history.append(_h, ignore_index=True)
        visualize_training_history(history, save_path=os.path.join(save_path, f"history_{network_name}.png"))
        history.to_csv(os.path.join(save_path, f"history_{network_name}.csv"))

        # msg = f"Epoch{epoch}, lr:{cur_lr:.4f}, train_loss:{train_loss:.4f}, train_acc:{train_acc:.4f}, val_loss:{val_loss:.4f}, val_acc:{val_acc:.4f}, test_loss:{test_loss:.4f}, test_acc:{test_acc:.4f}"
        msg = f"Epoch{epoch}, lr:{cur_lr:.4f}, train_loss:{train_loss:.4f}, train_acc:{train_acc:.4f}, val_loss:{val_loss:.4f}, val_acc:{val_acc:.4f}, val_auc:{score_auc:.4f}, train_patch_loss:{train_patch_loss:.4f}, train_patch_acc:{train_patch_acc:.4f},val_patch_loss:{val_patch_loss:.4f},val_patch_acc:{val_patch_acc:.4f},val_patch_auc:{val_patch_auc:.4f}"
        if is_best_loss:
            best_loss_epoch, msg = epoch, msg + "  => best loss"
        if is_best_score:
            best_score_epoch, msg = epoch, msg + "  => best score"
            best_score_auc = score_auc
        # if is_test_best_score:
        #     best_test_score_epoch,msg = epoch, msg+"  => best test score "
        logging.info(msg)
        save_model(model, optimizer, lr_scheduler, attentions,best_score_acc, is_best_loss, is_best_score,is_test_best_loss,is_test_best_score)

    # train_acc1 = history['train_acc'].dropna()
    # val_acc1 = history['val_acc'].dropna()
    # train_loss1 = history['train_loss'].dropna()
    # val_loss1 = history['val_loss'].dropna()
    # for i in range(np.array(train_loss1).shape[0]):
    #     writer1.add_scalars(f'loss', {'train_loss':np.array(train_loss1)[i], 'val_loss':np.array(val_loss1)[i]}, i)
    #     # writer1.add_scalar('loss/val_loss', np.array(val_loss1)[i], i)
    #     writer2.add_scalars(f'acc', {'train_acc':np.array(train_acc1)[i], 'val_acc':np.array(val_acc1)[i]}, i)
    #     # writer2.add_scalar('acc/val_acc', np.array(val_acc1)[i], i)

    return best_score_acc, best_score_auc


if __name__ == '__main__':
    # Training settings
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accs = []
    aucs = []
        
    best_score_acc, best_score_auc = train(args.seed,DEVICE)
        

    accs.append(best_score_acc)
    aucs.append(best_score_auc)
    logging.info(accs)
    logging.info(aucs)
    logging.info(f'ACC {np.array(accs).mean():.4f}±{np.array(accs).std():.4f}')
    logging.info(f'AUC {np.array(aucs).mean():.4f}±{np.array(aucs).std():.4f}')