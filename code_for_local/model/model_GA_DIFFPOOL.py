import torch
from torch import nn
import torch.nn.functional as F

from .GraphAttention import GraphAttentionLayer
from .DiffPool import dense_diff_pool


class GAN(nn.Module):
    def __init__(self, feat_dim, node_num, hidden_num=3, class_num=2, nheads=5):
        super(GAN, self).__init__()
        self.class_num = class_num
        assign_dim = node_num
        self.bn = False
        self.dropout = 0.5

        # self.linear_feature = nn.Linear(feat_dim, 32)

        self.attentions = [GraphAttentionLayer(feat_dim, hidden_num, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(hidden_num * nheads, hidden_num * nheads, concat=False)

        ## diff_pool_block1
        self.conv_pool1 = GraphAttentionLayer(hidden_num * nheads, 32)
        self.linear_pool1 = nn.Linear(32, 32)

        ## diff_pool_block2
        self.conv_pool2 = GraphAttentionLayer(hidden_num * nheads, 4)
        self.linear_pool2 = nn.Linear(4, 4)

        self.linear = nn.Linear(4 * hidden_num * nheads, self.class_num)

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
        x = torch.cat([att(x, adj)[0] for att in self.attentions], dim=2)

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj)[0])  # b, 116, 15

        ## diffpool_block1
        x_pool1 = self.apply_bn(self.conv_pool1(x, adj)[0])
        s = self.linear_pool1(x_pool1)
        x, adj, link_loss, ent_loss = dense_diff_pool(x, adj, s)  # b, 32, 15
        loss = loss + link_loss

        ## diffpool_block2
        x_pool2 = self.apply_bn(self.conv_pool2(x, adj)[0])
        s = self.linear_pool2(x_pool2)
        x, adj, link_loss, ent_loss = dense_diff_pool(x, adj, s)  # b, 4, 15
        loss = loss + link_loss

        out = x.view(x.size()[0], -1)
        ypred = self.linear(out)

        return ypred, loss, attentions


if __name__ == "__main__":
    model = GAN(feat_dim=3, node_num=90, assign_ratio=0.5)
    print(model)
    x = torch.tensor([list(range(0, 90)), list(range(4, 94))[::-1]]).float().unsqueeze(-1)
    x = torch.cat([x, x, x], dim=-1)
    adj = torch.ones(2, 90, 90, dtype=torch.float)

    a = model(x, adj)
