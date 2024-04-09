import torch
import torch.nn.functional as F
from ugnn.utils import GraphConv
import math
from torch_geometric.nn import GCNConv
from ugnn.utils import HashedModule


class Universal(HashedModule):
    def __init__(self, feats, classes, hidden=64, depth=10, cached=True, dropout=0.6, nri=0):
        super().__init__()
        self.nri = nri

        self.inconv1 = GCNConv(feats, hidden)
        self.inconv2 = GCNConv(hidden,  hidden-nri)
        self.dimreduce = torch.nn.Linear(feats, hidden-nri)
        feats = hidden

        #self.toclasses1 = torch.nn.Linear(feats, hidden)
        #self.toclasses2 = torch.nn.Linear(hidden, hidden)
        self.toclasses3 = torch.nn.Linear(hidden, classes)

        embedding_dim = int(1 + math.log2(feats))
        hidden = 4 + embedding_dim
        self.class_indicator_embedding = torch.nn.Embedding(feats, embedding_dim)
        self.adjust1 = torch.nn.Linear(2 + embedding_dim, hidden)
        self.adjust2 = torch.nn.Linear(hidden, 1)
        self.adjust2 = torch.nn.Linear(hidden, 1)
        self.conv = GraphConv(cached=cached)
        self.diffusion = [0.9 for _ in range(depth)]
        self.dropout = dropout
        self.hashed = None

    def train(self, train = True):
        super().train(train)
        self.hashed = None

    def _forward(self, data):
        x, edges = data.x, data.edges
        dropout = self.dropout

        # convert features to lower representation

        if self.nri > 0:
            x = self.dimreduce(x)
            #x = F.tanh(self.inconv1(x, edges))
            #x = self.inconv2(x, edges)
            random_dims = torch.empty(x.shape[0], self.nri, device=x.device)
            torch.nn.init.kaiming_uniform_(random_dims)
            #torch.nn.init.uniform_(random_dims, a=-1.0, b=1.0)
            x = torch.cat([x, random_dims], dim=1)
        else:
            x = self.dimreduce(x)

        # diffuse feature representations
        h0 = x
        # propagate
        for diffusion in self.diffusion:
            x = self.conv(x, edges) * diffusion + (1.0 - diffusion) * h0

        x = x.t()
        self.original_size = x.size()
        x = x.reshape(-1, 1)
        self.diffused_transformed_x = x
        self.transformed_h0 = h0.t().reshape(-1, 1)

        # create class indicator about which dims are folded (these are NOT the dataset classes)
        if not hasattr(self, "class_indicator"):
            num_samples = x.shape[0]
            num_classes = x.shape[1]  # diffusion classes = feature column ids
            class_indicator = torch.zeros(
                num_samples * num_classes, device=x.device, dtype=torch.int
            )
            for cl in range(num_classes):
                class_indicator[
                    (cl * num_samples) : (cl * num_samples + num_samples)
                ] = cl
            class_indicator.requires_grad_(False)
            self.class_indicator = class_indicator

        # create universal local attractor features: fold initial columns, diffused columns, indicate which dimension the folding refers to
        x = torch.concat(
            [
                self.diffused_transformed_x,
                self.transformed_h0,
                self.class_indicator_embedding(self.class_indicator),
            ],
            dim=1,
        )

        # transform and get back to original shape
        x = F.leaky_relu(self.adjust1(x))
        x = self.adjust2(x)/2
        x = x.reshape(self.original_size).t()
        x = F.dropout(x, training=self.training, p=dropout)

        # propagate again
        h0 = x
        for diffusion in self.diffusion:
            x = self.conv(x, edges) * diffusion + (1.0 - diffusion) * h0

        # reduce to classes
        #x = F.elu(self.toclasses1(x))
        #x = F.dropout(x, training=self.training, p=dropout)
        #x = F.elu(self.toclasses2(x))
        x = self.toclasses3(x)

        return x
