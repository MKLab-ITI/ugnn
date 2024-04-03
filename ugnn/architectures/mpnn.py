import torch
import torch.nn.functional as F
from ugnn.utils import GraphConv


class MPNN(torch.nn.Module):
    def __init__(self, feats, classes, hidden=64, alpha=0.1, depth=10):
        super().__init__()
        self.module_list1 = torch.nn.ModuleList()
        for k in range(depth):
            self.module_list2.append(torch.nn.Linear(feats, hidden))

        self.module_list1 = torch.nn.ModuleList()
        self.module_list2.append(torch.nn.Linear(feats, hidden))


        self.conv = GraphConv(
            edge_dropout=lambda x: F.dropout(x, p=0.5, training=self.training)
        )
        self.alpha = alpha
        self.depth = depth
        self.feats_dropout = True

    def forward(self, data):
        x, edges = data.x, data.edges
        if self.feats_dropout:
            x = F.dropout(x, training=self.training and x.shape[1] > 1, p=0.6)
        x = self.embed1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.6)
        x = self.embed2(x)
        h0 = x
        for _ in range(self.depth):
            x = self.conv(x, edges) * (1 - self.alpha) + self.alpha * h0

        return x
