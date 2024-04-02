import torch
import torch.nn.functional as F
from ugnn.utils import GraphConv
import math


class Universal(torch.nn.Module):
    def __init__(self, feats, classes, hidden=64, depth=10, cached=True, dropout=0.6):
        super().__init__()
        self.dimreduce = torch.nn.Linear(feats, hidden)
        feats = hidden

        #self.toclasses1 = torch.nn.Linear(feats, hidden)
        self.toclasses2 = torch.nn.Linear(feats, classes)

        embedding_dim = int(1 + math.log2(feats))
        hidden = 4 + embedding_dim
        self.class_indicator_embedding = torch.nn.Embedding(feats, embedding_dim)
        self.adjust1 = torch.nn.Linear(2 + embedding_dim, hidden)
        self.adjust2 = torch.nn.Linear(hidden, 1)
        self.conv = GraphConv(cached=cached)
        self.diffusion = [0.9 for _ in range(depth)]
        self.dropout = dropout

    def forward(self, data):
        x, edges = data.x, data.edges
        dropout = self.dropout

        # convert features to lower representation
        x = self.dimreduce(x)

        # diffuse feature representations
        h0 = x
        if True:
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
        x = self.adjust2(x) / 2
        x = x.reshape(self.original_size).t()
        x = F.dropout(x, training=self.training, p=dropout)

        # propagate again
        h0 = x
        for diffusion in self.diffusion:
            x = self.conv(x, edges) * diffusion + (1.0 - diffusion) * h0

        # reduce to classes
        #x = F.leaky_relu(self.toclasses1(x))
        #x = F.dropout(x, training=self.training, p=dropout)
        x = self.toclasses2(x)

        return x
