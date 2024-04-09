import torch


class HashedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def train(self, train=True):
        if self.training == train:
            return
        super().train(train)
        self.hashed = None

    def eval(self):
        if not self.training:
            return
        super().eval()
        self.hashed = None

    def forward(self, data):
        if self.hashed is None:
            self.hashed = self._forward(data)
        return self.hashed
