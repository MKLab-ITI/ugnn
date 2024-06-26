import torch
import torch.nn.functional as F
from ugnn.utils import GraphConv
from ugnn.utils import HashedModule


class APPNP(HashedModule):
    def __init__(self, feats, classes, hidden=64, alpha=0.1, depth=10):
        super().__init__()
        self.embed1 = torch.nn.Linear(feats, hidden)
        self.embed2 = torch.nn.Linear(hidden, classes)
        self.conv = GraphConv(
            edge_dropout=lambda x: F.dropout(x, p=0.5, training=self.training)
        )
        self.alpha = alpha
        self.depth = depth
        self.feats_dropout = True

    def _forward(self, data):
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
