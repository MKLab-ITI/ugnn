import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from ugnn.utils import HashedModule


class GCN(HashedModule):
    def __init__(self, feats, classes, hidden=64, dropout=0.6):
        super().__init__()
        self.conv1 = GCNConv(feats, hidden)
        self.conv2 = GCNConv(hidden, classes)
        self.dropout = dropout

    def _forward(self, data):
        x, edges = data.x, data.edges
        x = F.dropout(x, training=self.training and x.shape[1] > 1, p=self.dropout)
        x = F.relu(self.conv1(x, edges))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edges)
        return x
