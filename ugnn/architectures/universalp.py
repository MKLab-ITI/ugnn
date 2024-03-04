import torch
import torch.nn.functional as F
from ugnn.utils import GraphConv


class UniversalP(torch.nn.Module):
    def __init__(self, feats, classes, hidden=64, depth=[0.9] * 2, layers=2):
        super().__init__()
        self.linear1 = torch.nn.Linear(feats, hidden)
        self.linear2 = torch.nn.Linear(hidden, classes)
        # hidden = 1 + classes + feats + 2
        hidden = 3 + classes + hidden
        self.adjust1 = torch.nn.Linear(1 + classes + feats, hidden)
        self.adjust2 = torch.nn.Linear(hidden, 1)
        self.additional_layers = torch.nn.ModuleList()
        for _ in range(layers - 2):
            self.additional_layers.append(torch.nn.Linear(hidden, hidden))

        self.conv = GraphConv()
        self.class_indicator = None
        # self.diffusion = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.9)) for _ in range(10)])
        self.diffusion = (
            depth if isinstance(depth, list) else [0.9 for _ in range(depth)]
        )

    def forward(self, data):
        x, edges = data.x, data.edges
        # predict
        x = F.dropout(x, training=self.training)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, training=self.training)
        x = self.linear2(x)

        # propagate
        h0 = x
        for diffusion in self.diffusion:
            x = self.conv(x, edges) * diffusion + (1.0 - diffusion) * h0

        # create class indicator if not existing
        if self.class_indicator is None:
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
            [x, self.class_indicator, data.x.repeat(data.classes, 1)], dim=1
        )

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
        return x
