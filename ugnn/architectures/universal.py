import math

import torch
import torch.nn.functional as F
from ugnn.utils import GraphConv
import math

class Universal(torch.nn.Module):
    def __init__(self, feats, classes, hidden=64, depth=10, layers=2, cached=True):
        super().__init__()
        #self.linear1 = torch.nn.Linear(feats, hidden)
        self.linear1 = torch.nn.Linear(feats, classes)
        embedding_dim = int(1+math.log2(feats))
        hidden = 3+embedding_dim
        self.adjust1 = torch.nn.Linear(1+embedding_dim, hidden)
        self.adjust2 = torch.nn.Linear(hidden, 1)
        self.additional_layers = torch.nn.ModuleList()
        self.class_indicator_embedding = torch.nn.Embedding(feats, embedding_dim)
        for _ in range(layers - 2):
            self.additional_layers.append(torch.nn.Linear(hidden, hidden))
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
            num_samples = data.x.shape[0]
            num_classes = x.shape[1]  # diffusion classes = feature column ids
            class_indicator = torch.zeros(num_samples * num_classes, device=x.device, dtype=torch.int)
            for cl in range(num_classes):
                class_indicator[(cl * num_samples) : (cl * num_samples + num_samples)] = cl
            class_indicator.requires_grad_(False)
            self.class_indicator = class_indicator

        # create repeated node features
        x = x.t()
        original_size = x.size()
        x = x.reshape(-1, 1)
        x = torch.concat([x, self.class_indicator_embedding(self.class_indicator)], dim=1)

        # transform and get back to original shape
        x = F.relu(self.adjust1(x))
        for layer in self.additional_layers:
            x = F.relu(layer(x))
        x = self.adjust2(x)
        x = x.reshape(original_size).t()

        # propagate again
        h0 = x
        for diffusion in self.diffusion:
            x = self.conv(x, edges) * diffusion + (1.0 - diffusion) * h0

        x = self.linear1(x)

        return x
