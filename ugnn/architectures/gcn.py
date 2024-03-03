import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, feats, classes, hidden):
        super().__init__()
        self.conv1 = GCNConv(feats, hidden)
        self.conv2 = GCNConv(hidden, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, training=self.training)
        x = self.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
