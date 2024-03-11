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
        deviation=None,
        mask_mask=torch.tensor(1, dtype=torch.bool),
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
        self.deviation = (lambda x, y: (x==y).sum()) if deviation is None else deviation
        self.mask_mask = mask_mask

    def range(self, start: float, end: float):
        # end is non-inclusive
        num_nodes = self.labels.shape[0]
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.labels.device)
        mask[torch.arange(int(num_nodes * start), int(num_nodes * end))] = 1
        return self.on(mask*self.mask_mask)

    def to(self, device: torch.device):
        return ClassificationTask(
            self.x.to(device),
            self.edges.to(device),
            self.labels.to(device),
            None if self.mask is None else self.mask.to(device),
            l1=self.l1,
            classes=self.classes,
            deviation=self.deviation,
            mask_mask=self.mask_mask.to(device)
        )

    def on(self, mask: torch.Tensor):
        return ClassificationTask(
            self.x, self.edges, self.labels, mask, l1=self.l1, classes=self.classes, deviation=self.deviation
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
        correct = self.deviation(pred[self.mask], self.labels[self.mask])
        return float(correct) / int(self.mask.sum())

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
