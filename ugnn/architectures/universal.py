import torch
import torch.nn.functional as F
from ugnn.utils import GraphConv
import math


class Universal(torch.nn.Module):
    def __init__(self, feats, classes, hidden=64, depth=10, cached=True):
        super().__init__()

        self.linear1 = torch.nn.Linear(feats * 2, hidden)
        self.linear2 = torch.nn.Linear(hidden, classes)
        embedding_dim = int(1 + math.log2(feats))
        hidden = 4 + embedding_dim
        self.adjust1 = torch.nn.Linear(2 + embedding_dim, hidden)
        self.adjust2 = torch.nn.Linear(hidden, 1)
        self.class_indicator_embedding = torch.nn.Embedding(feats, embedding_dim)
        self.conv = GraphConv(cached=cached)
        self.diffusion = [0.9 for _ in range(depth)]

    def forward(self, data):
        x, edges = data.x, data.edges

        # propagate
        h0 = x
        for diffusion in self.diffusion:
            x = self.conv(x, edges) * diffusion + (1.0 - diffusion) * h0

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

        # create repeated node features
        x = x.t()
        original_size = x.size()
        x = x.reshape(-1, 1)
        x = torch.concat(
            [
                x,
                h0.t().reshape(-1, 1),
                self.class_indicator_embedding(self.class_indicator),
            ],
            dim=1,
        )

        dropout = 0.5
        # transform and get back to original shape
        x = F.relu(self.adjust1(x))
        x = self.adjust2(x) / 2
        x = x.reshape(original_size).t()
        x = F.dropout(x, training=self.training, p=dropout)

        x = torch.cat([x, h0], dim=1)
        # reduce to classes
        x = F.dropout(x, training=self.training, p=dropout)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, training=self.training, p=dropout)
        x = self.linear2(x)

        # propagate again
        h0 = x
        for diffusion in self.diffusion:
            x = self.conv(x, edges) * diffusion + (1.0 - diffusion) * h0

        return x
