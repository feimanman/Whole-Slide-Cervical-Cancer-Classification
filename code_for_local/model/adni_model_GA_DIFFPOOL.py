import torch
from torch import nn
import torch.nn.functional as F
from model.GraphAttention import GraphAttentionLayer
# from GraphAttention import GraphAttentionLayer
# from DiffPool import dense_diff_pool

class GAN(nn.Module):
    def __init__(self, feat_dim, node_num, hidden_num, class_num, nheads,dropout=0.5): #2
        super(GAN, self).__init__()
        self.class_num = class_num
        self.node_num = node_num
        self.bn = False
        self.dropout = dropout

        # self.linear_feature = nn.Linear(feat_dim, 32)

        self.attentions = [GraphAttentionLayer(feat_dim, hidden_num, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(hidden_num * nheads, hidden_num * nheads, concat=False)

        # ## diff_pool_block1
        # self.conv_pool1 = GraphAttentionLayer(hidden_num * nheads, 32) #16
        # # self.linear_pool1 = nn.Linear(32, 32)

        # # ## diff_pool_block2
        # self.conv_pool2 = GraphAttentionLayer(hidden_num * nheads, 4) #2
        # self.linear_pool2 = nn.Linear(4, 4)
        # self.linear_patch = nn.Linear(hidden_num * nheads, self.class_num)
        self.linear_patch_1 = nn.Linear(hidden_num * nheads, self.class_num)
        self.linear_patch_2 = nn.Linear(hidden_num * nheads, self.class_num)
        self.linear_patch_3 = nn.Linear(hidden_num * nheads, self.class_num)
        self.linear_patch_4 = nn.Linear(hidden_num * nheads, self.class_num)
        self.linear_patch_5 = nn.Linear(hidden_num * nheads, self.class_num)
        self.pooling = nn.MaxPool2d((node_num,1))
        # self.pooling = nn.AvgPool2d((20,1))
        self.linear = nn.Linear(hidden_num * nheads, self.class_num)

    def apply_bn(self, x):
        '''
        Batch normalization of 3D tensor x
        '''
        if self.bn:
            bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
            return bn_module(x)
        else:
            return x

    def forward(self, x, adj):
        loss = 0

        # x = self.linear_feature(x)

        x = F.dropout(x, self.dropout, training=self.training)
        attentions = [att(x, adj)[1] for att in self.attentions]
        # print(len(attentions))
        # print(attentions[0])
        x = torch.cat([att(x, adj)[0] for att in self.attentions], dim=2)
        out_0 = self.pooling(x).view(x.size()[0],-1)
        out_0 = F.tanh(out_0)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj)[0])  # b, 116, 15
        # print(x.shape)
        # print(x[:,0].shape)
        # x_patch = self.linear_patch(x)
        # print(x_patch)
        out = self.pooling(x).view(x.size()[0],-1)
        # print(out.shape)
        out = F.tanh(out)
        ypred_1 = self.linear_patch_1(x[:,0])
        # print(ypred_1)
        ypred_2 = self.linear_patch_2(x[:,1])
        ypred_3 = self.linear_patch_3(x[:,2])
        ypred_4 = self.linear_patch_4(x[:,3])
        ypred_5 = self.linear_patch_5(x[:,4])

        ypred = self.linear(out)
        # print(ypred.shape)
        return ypred, loss,attentions, ypred_1,ypred_2,ypred_3,ypred_4,ypred_5

if __name__ == "__main__":
    #model = GAN(feat_dim=3, node_num=90, assign_ratio=0.5)
    model = GAN(feat_dim=2048, node_num=5, hidden_num=512, class_num=2,nheads=8,dropout=0.5)
    # print(model)
    x = torch.randn(2,5,2048)
    adj = torch.ones(2, 5, 5, dtype=torch.float)
    print(x.shape)
    ypred, loss,attentions, ypred_1,ypred_2,ypred_3,ypred_4,ypred_5 = model(x, adj)

    print(ypred_1)
    print(ypred_2)