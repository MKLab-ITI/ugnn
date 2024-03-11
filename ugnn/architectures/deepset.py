import torch
import torch.nn.functional as F
from ugnn.utils import GraphConv


class DeepSet(torch.nn.Module):
    # "Deep Sets" Zaheer et al., 2017
    def __init__(self, feats, classes, hidden=64):
        super().__init__()
        self.equiv1 = torch.nn.Parameter(
            torch.tensor([[1.0 for _ in range(hidden)] for _ in range(feats)])
        )
        self.equiv2 = torch.nn.Parameter(
            torch.tensor([[1.0 for _ in range(hidden)] for _ in range(feats)])
        )
        torch.nn.init.kaiming_uniform_(self.equiv1)
        torch.nn.init.kaiming_uniform_(self.equiv2)
        # self.equiv3 = torch.nn.Parameter(torch.tensor([[1. for _ in range(classes)] for _ in range(hidden)]))
        # self.equiv4 = torch.nn.Parameter(torch.tensor([[1. for _ in range(classes)] for _ in range(hidden)]))
        # torch.nn.init.xavier_uniform_(self.equiv3)
        # torch.nn.init.xavier_uniform_(self.equiv4)
        self.conv = GraphConv()
        self.layer1 = torch.nn.Linear(hidden, hidden)
        self.layer2 = torch.nn.Linear(hidden, classes)

    def forward(self, data):
        x, edges = data.x, data.edges
        x = F.dropout(x, training=self.training and x.shape[1] > 1)
        x = torch.matmul(x, self.equiv1) + torch.matmul(
            self.conv(x, edges), self.equiv2
        )
        x = F.relu(x / 2)
        # x = F.dropout(x, training=self.training)
        # x = torch.matmul(x, self.equiv3) + torch.matmul(self.conv(x, edges), self.equiv4)
        # x = x/2
        x = F.dropout(x, training=self.training)
        x = F.relu(self.layer1(x))
        x = F.dropout(x, training=self.training)
        x = self.layer2(x)
        return x
