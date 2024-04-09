import torch
import torch.nn.functional as F
from ugnn.utils import GraphConv


class MPNN(torch.nn.Module):
    def __init__(self, feats, classes, hidden=64, dropout=0.6):
        super().__init__()
        self.linear1 = torch.nn.Linear(feats, hidden)
        self.linear2 = torch.nn.Linear(hidden*2, hidden)
        self.linear3 = torch.nn.Linear(hidden*2, hidden)
        self.linear4 = torch.nn.Linear(hidden*2, classes)
        self.dropout = dropout
        self.random_embeddings = None
        self.conv = GraphConv(cached=True)
        #    edge_dropout=lambda x: F.dropout(x, p=0.5, training=self.training)
        #)

    def forward(self, data):
        x, edges = data.x, data.edges
        x = F.dropout(x, training=self.training and x.shape[1] > 1, p=self.dropout)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.relu(self.linear2(torch.cat([x, self.conv(x, edges)], dim=1)))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.relu(self.linear3(torch.cat([x, self.conv(x, edges)], dim=1)))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.relu(self.linear4(torch.cat([x, self.conv(x, edges)], dim=1)))
        return x
