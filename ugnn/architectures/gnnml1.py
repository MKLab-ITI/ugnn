import torch
import torch.nn.functional as F
from ugnn.utils import GraphConv


class GNNML1(torch.nn.Module):
    def __init__(self, feats, classes, hidden=64, dropout=0.6):
        super().__init__()
        self.linear1 = torch.nn.Linear(feats, hidden)
        self.linear2 = torch.nn.Linear(hidden*2, hidden, bias=False)
        self.att2a = torch.nn.Linear(hidden, hidden, bias=False)
        self.att2b = torch.nn.Linear(hidden, hidden, bias=False)
        self.linear3 = torch.nn.Linear(hidden*2, hidden, bias=False)
        self.att3a = torch.nn.Linear(hidden, hidden, bias=False)
        self.att3b = torch.nn.Linear(hidden, hidden, bias=False)
        self.linear4 = torch.nn.Linear(hidden*2, classes, bias=False)
        self.att4a = torch.nn.Linear(hidden, classes, bias=False)
        self.att4b = torch.nn.Linear(hidden, classes, bias=False)
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
        x = F.relu(self.linear2(torch.cat([x, self.conv(x, edges)], dim=1))+torch.mul(self.att2a(x), self.att2b(x)))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.relu(self.linear3(torch.cat([x, self.conv(x, edges)], dim=1))+torch.mul(self.att3a(x), self.att3b(x)))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.relu(self.linear4(torch.cat([x, self.conv(x, edges)], dim=1))+torch.mul(self.att4a(x), self.att4b(x)))
        return x
