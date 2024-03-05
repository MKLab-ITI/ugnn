import torch
import torch.nn.functional as F
from ugnn.utils import GraphConv


class Universal(torch.nn.Module):
    def __init__(self, feats, classes, hidden=64, depth=20):
        super().__init__()
        self.linear1 = torch.nn.Linear(feats, hidden)
        self.linear2 = torch.nn.Linear(hidden, classes)

        self.adjust1 = torch.nn.ModuleList()
        self.adjust2 = torch.nn.ModuleList()
        for _ in range(classes):
            self.adjust1.append(torch.nn.Linear(1 + feats, hidden))
            self.adjust2.append(torch.nn.Linear(hidden, 1))

        self.conv = GraphConv()
        self.diffusion = [0.9 for _ in range(depth)]

    def forward(self, data):
        x, edges = data.x, data.edges
        # predict
        x = F.dropout(x, training=self.training)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, training=self.training)
        x = self.linear2(x)

        # propagate
        h0 = x
        for diffusion in self.diffusion:
            x = self.conv(x, edges) * diffusion + (1.0 - diffusion) * h0
        # x = x-x.min()
        # create a transformation for each class to serve as new propagation features
        transformed = list()
        for cl in range(data.classes):
            xcl = x[:, cl].reshape(-1, 1)
            xcl = torch.cat([xcl, data.x], dim=1)
            xcl = F.relu(self.adjust1[cl](xcl))
            xcl = self.adjust2[cl](xcl)
            transformed.append(xcl)
        x = torch.cat(transformed, dim=1)
        # x = x-x.min()

        # propagate again
        h0 = x
        for diffusion in self.diffusion:
            x = self.conv(x, edges) * diffusion + (1.0 - diffusion) * h0
        return x
