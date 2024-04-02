import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn2_conv import GCN2Conv


class GCNII(torch.nn.Module):
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn2_conv.html
    def __init__(
        self, feats, classes, nlayers=64, hidden=64, theta=0.6, alpha=0.1, cached=True
    ):
        super(GCNII, self).__init__()
        self.convs = torch.nn.ModuleList()
        for layer in range(nlayers):
            self.convs.append(
                GCN2Conv(
                    hidden, layer=layer + 1, theta=theta, alpha=alpha, cached=cached
                )
            )
        self.layer1 = torch.nn.Linear(feats, hidden)
        self.layer2 = torch.nn.Linear(hidden, classes)

    def forward(self, data):
        x, edges = data.x, data.edges
        x = F.dropout(x, training=self.training, p=0.6)
        x = F.relu(self.layer1(x))
        h0 = x
        for i, con in enumerate(self.convs):
            x = F.dropout(x, training=self.training, p=0.6)
            x = F.relu(con(x, h0, edges))
        x = F.dropout(x, training=self.training, p=0.6)
        x = self.layer2(x)
        return x
