import torch
import torch.nn.functional as F
from ugnn.utils import GraphConv


class UniversalP(torch.nn.Module):
    def __init__(
        self,
        feats,
        classes,
        hidden=64,
        depth=10,
        layers=2,
        cached=True,
        reuse_feats=True,
    ):
        super().__init__()
        self.reuse_feats = reuse_feats
        self.linear1 = torch.nn.Linear(feats, hidden)
        self.linear2 = torch.nn.Linear(hidden, classes)
        reused_feats = feats if reuse_feats else 0
        hidden = 3 + reused_feats
        self.adjust1 = torch.nn.Linear(1 + classes + reused_feats, hidden)
        self.adjust2 = torch.nn.Linear(hidden, 1)
        self.additional_layers = torch.nn.ModuleList()
        for _ in range(layers - 2):
            self.additional_layers.append(torch.nn.Linear(hidden, hidden))
        self.conv = GraphConv(cached=cached)
        self.diffusion = [0.9 for _ in range(depth)]

    def forward(self, data):
        x, edges = data.x, data.edges

        # predict
        x = F.dropout(x, training=self.training and x.shape[1] > 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        # propagate
        h0 = x
        for diffusion in self.diffusion:
            x = self.conv(x, edges) * diffusion + (1.0 - diffusion) * h0

        # create class indicator about which dims are folded (these are NOT the dataset classes)
        if not hasattr(self, "class_indicator"):
            num_samples = data.x.shape[0]
            class_indicator = torch.zeros(
                num_samples * data.classes, data.classes, device=x.device
            )
            for cl in range(data.classes):
                class_indicator[
                    (cl * num_samples) : (cl * num_samples + num_samples), cl
                ] = 1
            class_indicator.requires_grad_(False)
            self.class_indicator = class_indicator

        # create repeated node features
        x = x.t()
        original_size = x.size()
        x = x.reshape(-1, 1)
        concating = [x, self.class_indicator]
        if self.reuse_feats:
            concating += [data.x.repeat(data.classes, 1)]
        x = torch.concat(concating, dim=1)

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
        # x = self.equiv1*x + (1-self.equiv1)*x.max(dim=0).values.repeat(x.shape[0], 1)

        return x
