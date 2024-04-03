import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn2_conv.html
    def __init__(self, feats, classes, hidden=64, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(feats, hidden // heads, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden, classes, heads=1, dropout=0.6)

    def forward(self, data):
        x, edges = data.x, data.edges
        x = F.dropout(x, training=self.training, p=0.6)
        x = self.conv1(x, edges)
        x = self.conv2(x, edges)
        return x
