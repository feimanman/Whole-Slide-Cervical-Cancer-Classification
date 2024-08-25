import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm as tqdmauto
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score


def train_epoch(model, loader, device, optimizer, verbose=False):
    model.train()
    train_losses = []
    train_losses_patch = []
    pred_patch_list, target_patch_list = [], []
    pred_list, target_list = [], []
    optimizer.zero_grad()
    progress_bar = tqdmauto(loader) if verbose else None

    for batch_idx, (fc_matrixs, feature, targets, label_1, label_2, label_3, label_4, label_5) in enumerate(loader):
        # print(fc_matrixs.shape, feature.shape, targets.shape)
        # print('train')
        # batchsize, node_num, _ = fc_matrixs.shape
        fc_matrixs_batch = fc_matrixs.to(device)
        targets_batch = targets.to(device)
        feature_batch = feature.to(device)
        label_1_batch = label_1.to(device)
        label_2_batch = label_2.to(device)
        label_3_batch = label_3.to(device)
        label_4_batch = label_4.to(device)
        label_5_batch = label_5.to(device)
        targets_batch = torch.as_tensor(targets_batch, dtype=torch.long).to(device)
        label_1_batch = torch.as_tensor(label_1_batch, dtype=torch.long).to(device)
        label_2_batch = torch.as_tensor(label_2_batch, dtype=torch.long).to(device)
        label_3_batch = torch.as_tensor(label_3_batch, dtype=torch.long).to(device)
        label_4_batch = torch.as_tensor(label_4_batch, dtype=torch.long).to(device)
        label_5_batch = torch.as_tensor(label_5_batch, dtype=torch.long).to(device)
        preds, link_loss, _ , ypred_1,ypred_2,ypred_3,ypred_4,ypred_5= model(feature_batch, fc_matrixs_batch)
        # print(preds)
        loss = nn.CrossEntropyLoss(reduction='none')(preds, targets_batch) 
        # print(loss)
        # loss = loss * risks_batch
        # print(loss)
        loss = loss.mean()
        # print(loss)
        loss_1=nn.CrossEntropyLoss(reduction='none')(ypred_1, label_1_batch).mean() 
        loss_2=nn.CrossEntropyLoss(reduction='none')(ypred_2, label_2_batch).mean() 
        loss_3=nn.CrossEntropyLoss(reduction='none')(ypred_3, label_3_batch).mean() 
        loss_4=nn.CrossEntropyLoss(reduction='none')(ypred_4, label_4_batch).mean() 
        loss_5=nn.CrossEntropyLoss(reduction='none')(ypred_5, label_5_batch).mean() 

        # print(loss)
        loss_np = loss.detach().cpu().item()
        train_losses.append(loss_np)
        loss_np_patch = loss_1.detach().cpu().item() + loss_2.detach().cpu().item()+loss_3.detach().cpu().item()+loss_4.detach().cpu().item()+loss_5.detach().cpu().item()
        train_losses_patch.append(loss_np_patch)


        preds_score = F.softmax(preds, dim=1).detach().cpu().numpy()
        preds = np.argmax(preds_score, axis=1)
        pred_list.append(preds)
        targets = targets.detach().cpu().numpy()
        target_list.append(targets)

        preds_patch_score_1 =  F.softmax(ypred_1, dim=1).detach().cpu().numpy()
        preds_1 = np.argmax(preds_patch_score_1, axis=1)
        preds_patch_score_2 =  F.softmax(ypred_2, dim=1).detach().cpu().numpy()
        preds_2 = np.argmax(preds_patch_score_2, axis=1)
        preds_patch_score_3 =  F.softmax(ypred_3, dim=1).detach().cpu().numpy()
        preds_3 = np.argmax(preds_patch_score_3, axis=1)
        preds_patch_score_4 =  F.softmax(ypred_4, dim=1).detach().cpu().numpy()
        preds_4 = np.argmax(preds_patch_score_4, axis=1)
        preds_patch_score_5 =  F.softmax(ypred_5, dim=1).detach().cpu().numpy()
        preds_5 = np.argmax(preds_patch_score_5, axis=1)

        pred_patch_list.append(preds_1)
        pred_patch_list.append(preds_2)
        pred_patch_list.append(preds_3)
        pred_patch_list.append(preds_4)
        pred_patch_list.append(preds_5)

        target_1 = label_1.detach().cpu().numpy()
        target_2 = label_2.detach().cpu().numpy()
        target_3 = label_3.detach().cpu().numpy()
        target_4 = label_4.detach().cpu().numpy()
        target_5 = label_5.detach().cpu().numpy()
        target_patch_list.append(target_1)
        target_patch_list.append(target_2)
        target_patch_list.append(target_3)
        target_patch_list.append(target_4)
        target_patch_list.append(target_5)
        

        loss = loss+loss_1+loss_2+loss_3+loss_4+loss_5
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if verbose:
            progress_bar.set_postfix_str(f"loss: {loss_np:.4f}, smooth_loss: {np.mean(train_losses[-20:]):.4f}")
            progress_bar.update(1)
    if verbose:
        progress_bar.close()

    preds = np.concatenate(pred_list)
    targets = np.concatenate(target_list)
    score_acc = accuracy_score(targets, preds)

    preds_patch = np.concatenate(pred_patch_list)
    targets_patch = np.concatenate(target_patch_list)
    score_acc_patch = accuracy_score(targets_patch,preds_patch)
    return np.asarray(train_losses).mean(), score_acc , np.asarray(train_losses_patch).mean(), score_acc_patch


def val_epoch(model, loader, device):
    model.eval()
    val_losses = []
    val_losses_patch = []
    pred_list, target_list = [], []
    pred_patch_list , target_patch_list = [],[]
    preds_score_list = []
    preds_patch_score_list = []
    with torch.no_grad():
        for fc_matrixs, feature, targets, label_1, label_2, label_3, label_4, label_5 in loader:
            # batchsize, node_num = fc_matrixs.shape
            # print('val')
            # test= []
            fc_matrixs_batch = fc_matrixs.to(device)
            targets_batch = targets.to(device)
            feature_batch = feature.to(device)
            label_1_batch = label_1.to(device)
            label_2_batch = label_2.to(device)
            label_3_batch = label_3.to(device)
            label_4_batch = label_4.to(device)
            label_5_batch = label_5.to(device)
            targets_batch = torch.as_tensor(targets_batch, dtype=torch.long).to(device)
            label_1_batch = torch.as_tensor(label_1_batch, dtype=torch.long).to(device)
            label_2_batch = torch.as_tensor(label_2_batch, dtype=torch.long).to(device)
            label_3_batch = torch.as_tensor(label_3_batch, dtype=torch.long).to(device)
            label_4_batch = torch.as_tensor(label_4_batch, dtype=torch.long).to(device)
            label_5_batch = torch.as_tensor(label_5_batch, dtype=torch.long).to(device)
            preds, link_loss, attentions , ypred_1,ypred_2,ypred_3,ypred_4,ypred_5= model(feature_batch, fc_matrixs_batch)

            loss = nn.CrossEntropyLoss(reduction='none')(preds, targets_batch) 
            
            loss = loss.mean()
            # print(loss)
            loss_1=nn.CrossEntropyLoss(reduction='none')(ypred_1, label_1_batch).mean() 
            loss_2=nn.CrossEntropyLoss(reduction='none')(ypred_2, label_2_batch).mean() 
            loss_3=nn.CrossEntropyLoss(reduction='none')(ypred_3, label_3_batch).mean() 
            loss_4=nn.CrossEntropyLoss(reduction='none')(ypred_4, label_4_batch).mean() 
            loss_5=nn.CrossEntropyLoss(reduction='none')(ypred_5, label_5_batch).mean() 

            loss_np = loss.detach().cpu().item()
            loss_np_patch = loss_1.detach().cpu().item() + loss_2.detach().cpu().item()+loss_3.detach().cpu().item()+loss_4.detach().cpu().item()+loss_5.detach().cpu().item()
            
            preds_score = F.softmax(preds, dim=1).cpu().numpy()
            preds = np.argmax(preds_score, axis=1)
            
            targets = targets.detach().cpu().numpy()
            val_losses.append(loss_np)
            val_losses_patch.append(loss_np_patch)
            pred_list.append(preds)
            target_list.append(targets)
            
            preds_score_list.append(preds_score[:, 1])#

            preds_patch_score_1 =  F.softmax(ypred_1, dim=1).detach().cpu().numpy()
            preds_1 = np.argmax(preds_patch_score_1, axis=1)
            preds_patch_score_2 =  F.softmax(ypred_2, dim=1).detach().cpu().numpy()
            preds_2 = np.argmax(preds_patch_score_2, axis=1)
            preds_patch_score_3 =  F.softmax(ypred_3, dim=1).detach().cpu().numpy()
            preds_3 = np.argmax(preds_patch_score_3, axis=1)
            preds_patch_score_4 =  F.softmax(ypred_4, dim=1).detach().cpu().numpy()
            preds_4 = np.argmax(preds_patch_score_4, axis=1)
            preds_patch_score_5 =  F.softmax(ypred_5, dim=1).detach().cpu().numpy()
            preds_5 = np.argmax(preds_patch_score_5, axis=1)

            pred_patch_list.append(preds_1)
            pred_patch_list.append(preds_2)
            pred_patch_list.append(preds_3)
            pred_patch_list.append(preds_4)
            pred_patch_list.append(preds_5)

            target_1 = label_1.detach().cpu().numpy()
            target_2 = label_2.detach().cpu().numpy()
            target_3 = label_3.detach().cpu().numpy()
            target_4 = label_4.detach().cpu().numpy()
            target_5 = label_5.detach().cpu().numpy()
            target_patch_list.append(target_1)
            target_patch_list.append(target_2)
            target_patch_list.append(target_3)
            target_patch_list.append(target_4)
            target_patch_list.append(target_5)
            preds_patch_score_list.append(preds_patch_score_1[:, 1])
            preds_patch_score_list.append(preds_patch_score_2[:, 1])
            preds_patch_score_list.append(preds_patch_score_3[:, 1])
            preds_patch_score_list.append(preds_patch_score_4[:, 1])
            preds_patch_score_list.append(preds_patch_score_5[:, 1])
            # print(preds_patch_score_list)
            # test=[]
            # test.append(preds_patch_score_1[:, 1])
            # test.append(preds_patch_score_2[:, 1])
            # test.append(preds_patch_score_3[:, 1])
            # test.append(preds_patch_score_4[:, 1])
            # test.append(preds_patch_score_5[:, 1])
            # print(test)

    preds = np.concatenate(pred_list)
    preds_score = np.concatenate(preds_score_list)
    targets = np.concatenate(target_list)
    preds = preds.squeeze()
    targets = targets.squeeze()

    preds_patch =np.concatenate(pred_patch_list)
    preds_patch_score = np.concatenate(preds_patch_score_list)
    targets_patch = np.concatenate(target_patch_list)
    preds_patch = preds_patch.squeeze()
    targets_patch = targets_patch.squeeze()
    
   
    score_acc = accuracy_score(targets, preds)
    score_auc = roc_auc_score(targets, preds_score)

    patch_score_acc = accuracy_score(targets_patch,preds_patch)
    patch_score_auc = roc_auc_score(targets_patch,preds_patch_score)

    return np.asarray(val_losses).mean(), score_acc, score_auc, preds, preds_score, targets, attentions, np.asarray(val_losses_patch).mean(), patch_score_acc, patch_score_auc, preds_patch,preds_patch_score,targets_patch

def test_epoch(model, loader, device):
    model.eval()
    test_losses=[]
    pred_list, target_list, risk_list = [], [], []
    preds_score_list = []
    with torch.no_grad():
        for fc_matrixs, feature, targets, risks in loader:
            # batchsize, node_num = fc_matrixs.shape
            fc_matrixs_batch = fc_matrixs.to(device)
            targets_batch = targets.to(device)
            feature_batch = feature.to(device)
            #risk_batch=risks.to(device)
            targets_batch = torch.as_tensor(targets_batch, dtype=torch.long).to(device)

            preds, link_loss, attentions = model(feature_batch, fc_matrixs_batch)
            #print(preds)
            # loss = nn.CrossEntropyLoss()(preds, targets_batch) + 0.5 * link_loss
            loss = nn.CrossEntropyLoss(reduction='none')(preds, targets_batch) 
            # print(loss)
            # loss = loss * risks_batch
            # print(loss)
            loss = loss.mean()
            # print(loss)
            loss +=  0.5 * link_loss
            
            loss_np = loss.detach().cpu().item()
            preds_score = F.softmax(preds, dim=1).cpu().numpy()
            preds = np.argmax(preds_score, axis=1)
            risks = risks.cpu().numpy()
            targets = targets.cpu().numpy()
            test_losses.append(loss_np)
            pred_list.append(preds)
            target_list.append(targets)
            preds_score_list.append(preds_score[:, 1])
            risk_list.append(risks)
    preds = np.concatenate(pred_list)
    preds_score = np.concatenate(preds_score_list)
    targets = np.concatenate(target_list)
    risks=np.concatenate(risk_list)
    #print(preds.shape)
    #print(targets.shape)
    score_acc = accuracy_score(targets, preds)
    score_auc = roc_auc_score(targets, preds_score)
    return np.asarray(test_losses).mean(),score_acc, score_auc, preds, preds_score, targets, risks,attentions