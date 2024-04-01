import torch
import torch.nn.functional as F
from ugnn.utils import GraphConv


class DataPass:
    def __init__(self, x, edges, classes):
        self.x = x
        self.edges = edges
        self.classes = classes


class UniversalP(torch.nn.Module):
    def __init__(
        self,
        feats,
        classes,
        hidden=64,
        depth=10,
        cached=True,
    ):
        super().__init__()
        self.linear1 = torch.nn.Linear(feats, classes)
        # self.linear2 = torch.nn.Linear(hidden, classes)
        # hidden = 3 + classes + feats
        self.adjust1 = torch.nn.Linear(1 + classes + feats, hidden)
        self.adjust2 = torch.nn.Linear(hidden, 1)
        conv = GraphConv(cached=cached)
        self.convs = [conv for _ in range(depth)]
        self.diffusion = [0.9 for _ in range(depth)]

    def forward(self, data):
        x, edges = data.x, data.edges
        # predict
        x = F.dropout(x, training=self.training and x.shape[1] > 1)
        datax = x
        # x = F.relu(self.linear1(x))
        # x = self.linear2(x)
        x = self.linear1(x)
        # propagate
        h0 = x
        for conv, diffusion in zip(self.convs, self.diffusion):
            x = conv(x, edges) * diffusion + (1.0 - diffusion) * h0

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
        x = torch.concat(
            [x, self.class_indicator, datax.repeat(data.classes, 1)], dim=1
        )

        # transform and get back to original shape
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.adjust1(x))
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.adjust2(x))
        x = x.reshape(original_size).t()

        # propagate again
        self.training, training = False, self.training
        h0 = x
        for conv, diffusion in zip(self.convs, self.diffusion):
            x = conv(x, edges) * diffusion + (1.0 - diffusion) * h0
        self.training = training

        # x = self.equiv1*x + (1-self.equiv1)*x.max(dim=0).values.repeat(x.shape[0], 1)

        return x
