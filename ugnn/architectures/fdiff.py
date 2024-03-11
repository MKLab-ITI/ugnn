import torch
import torch.nn.functional as F
from ugnn.utils import GraphConv


class FDiff(torch.nn.Module):
    def __init__(self, feats, classes, traindata, hidden=64, alpha=0.1, depth=10):
        super().__init__()
        self.embed1 = torch.nn.Linear(feats, hidden)
        self.embed2 = torch.nn.Linear(hidden, classes)
        self.conv = GraphConv()
        self.converr = GraphConv()
        self.alpha = alpha
        self.depth = depth
        self.traindata = traindata

    def forward(self, data):
        x, edges = data.x, data.edges
        x = F.dropout(x, training=self.training)
        x = self.embed1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.embed2(x)
        if self.training:
            return x
        x = torch.softmax(x, dim=1)
        trueonehot = torch.zeros(x.shape[0], self.traindata.classes, device=x.device)
        nz = self.traindata.mask.nonzero()
        trueonehot[nz, self.traindata.labels] = 1
        err = trueonehot-x
        h0 = err
        for _ in range(self.depth):
            err = self.converr(err, edges)
            err[nz, :] = h0[nz, :]
        x = x + err

        h0 = x
        for _ in range(self.depth):
            x = self.conv(x, edges)*0.9 + h0*0.1

        return torch.log(x+1)
