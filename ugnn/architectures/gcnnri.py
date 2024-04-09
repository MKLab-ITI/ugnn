import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear
from ugnn.utils import HashedModule


class GCNNRI(HashedModule):
    def __init__(self, feats, classes, hidden=64, dropout=0.5):
        super().__init__()
        self.hidden = hidden
        self.conv1 = GCNConv(feats, hidden)
        self.conv2 = GCNConv(hidden, hidden//2)
        self.inter1 = GCNConv(hidden, hidden)
        self.inter2 = GCNConv(hidden, hidden)
        self.inter3 = GCNConv(hidden, hidden)
        self.linear1 = Linear(hidden, hidden)
        self.linear2 = Linear(hidden, hidden)
        self.linear3 = Linear(hidden, classes)
        self.dropout = dropout

    def _forward(self, data):
        x, edges = data.x, data.edges
        x = F.tanh(self.conv1(x, edges))
        x = F.tanh(self.conv2(x, edges))
        random_dims = torch.empty(x.shape[0], self.hidden-self.hidden//2, device=x.device)
        #torch.nn.init.uniform_(random_dims, a=-1.0, b=1.0)
        torch.nn.init.xavier_uniform_(random_dims)
        x = torch.cat([x, random_dims], dim=1)

        x = F.tanh(self.inter1(x, edges))
        x = F.tanh(self.inter2(x, edges))
        x = F.tanh(self.inter3(x, edges))

        x = F.elu(self.linear1(x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.elu(self.linear2(x))
        x = self.linear3(x)
        return x
