from ugnn.tasks.task import ClassificationTask
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


class PlanetoidTask(ClassificationTask):
    def __init__(self, name, device, **kwargs):
        dataset = Planetoid(root=f'/tmp/{name}',
                            name=name,
                            transform=T.Compose([T.NormalizeFeatures()]),
                            **kwargs)
        data = dataset[0]
        self.data = data
        data = data.to(device)
        super().__init__(data.x, data.edge_index, data.y, dataset.num_classes)

    def to(self, device):
        if self.labels.device is not device:
            raise Exception("PlanetoidTask device can change only during its construction")
        return self

    def split(self):
        return {
            "train": self.on(self.data.train_mask),
            "valid": self.on(self.data.val_mask),
            "test": self.on(self.data.test_mask),
        }