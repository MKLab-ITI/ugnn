import torch.nn.functional as F
import torch


class ClassificationTask:
    def __init__(
        self,
        x: torch.Tensor,
        edges: torch.Tensor,
        labels: torch.Tensor,
        mask=None,
        l1: float = 0,
        classes=None,
    ):
        self.x = x
        self.edges = edges
        self.labels = labels
        self.mask = mask
        self.l1 = l1
        self.classes = int(labels.max().detach()) + 1 if classes is None else classes
        self.feats = x.shape[1]
        # just in case, disable any gradient computations
        self.x.requires_grad_(False)
        self.edges.requires_grad_(False)
        if self.labels is not None:
            self.labels.requires_grad_(False)

    def range(self, start: float, end: float):
        # end is non-inclusive
        num_nodes = self.labels.shape[0]
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.labels.device)
        mask[torch.arange(int(num_nodes * start), int(num_nodes * end))] = 1
        return self.on(mask)

    def to(self, device: torch.device):
        return ClassificationTask(
            self.x.to(device),
            self.edges.to(device),
            self.labels.to(device),
            None if self.mask is None else self.mask.to(device),
            l1=self.l1,
            classes=self.classes,
        )

    def on(self, mask: torch.Tensor):
        return ClassificationTask(
            self.x, self.edges, self.labels, mask, l1=self.l1, classes=self.classes
        )

    def out(self, model: torch.nn.Module):
        x = model(self)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, model: torch.nn.Module):
        out = self.out(model)
        loss = F.nll_loss(out[self.mask, :], self.labels[self.mask])
        if self.l1 != 0:
            loss = loss + self.l1 * torch.mean(out.abs())
        return loss

    def evaluate(self, model: torch.nn.Module):
        out = self.out(model)
        pred = out.argmax(dim=1)
        correct = (pred[self.mask] == self.labels[self.mask]).sum()
        return int(correct) / int(self.mask.sum())

    def split(self, training=0.5, validation=0.25):
        assert training + validation < 1
        assert training > 0
        assert validation > 0
        return {
            "train": self.range(0, training),
            "valid": self.range(training, training + validation),
            "test": self.range(training + validation, 1),
        }

    def overtrain(self):
        return {
            "train": self.range(0, 1),
            "valid": self.range(0, 1),
            "test": self.range(0, 1),
        }
