import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, feats, classes, hidden=64):
        super().__init__()
        self.layer1 = torch.nn.Linear(feats, hidden)
        self.layer2 = torch.nn.Linear(hidden, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edges
        x = F.dropout(x, training=self.training)
        x = F.relu(self.layer1(x))
        x = F.dropout(x, training=self.training)
        x = self.layer2(x)
        return x
