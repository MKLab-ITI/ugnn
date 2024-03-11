from ugnn.tasks.base import ClassificationTask
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import torch


class PlanetoidTask(ClassificationTask):
    def __init__(self, name, device, **kwargs):
        dataset = Planetoid(
            root=f"/tmp/{name}",
            name=name,
            transform=T.Compose([T.NormalizeFeatures()]),
            **kwargs,
        )
        data = dataset[0]
        self.data = data
        data = data.to(device)
        super().__init__(data.x, data.edge_index, data.y, dataset.num_classes)

    def to(self, device):
        if self.labels.device is not device:
            raise Exception(
                "PlanetoidTask device can change only during its construction"
            )
        return self

    def _create_masks(self, ntrain, nvalid, ntest):
        labels = self.labels
        unique_labels = torch.unique(labels)

        # Initialize masks
        train_mask = torch.zeros(len(labels), dtype=torch.bool)
        valid_mask = torch.zeros(len(labels), dtype=torch.bool)
        test_mask = torch.zeros(len(labels), dtype=torch.bool)

        for label in unique_labels:
            # Find indices of the current class
            indices = torch.where(labels == label)[0]

            # Shuffle indices to ensure random selection
            indices = indices[torch.randperm(len(indices))]

            # Assign indices to each split
            train_indices = indices[:ntrain]
            valid_indices = indices[ntrain : ntrain + nvalid]
            test_indices = indices[ntrain + nvalid : ntrain + nvalid + ntest]

            # Update masks
            train_mask[train_indices] = True
            valid_mask[valid_indices] = True
            test_mask[test_indices] = True

        return train_mask, valid_mask, test_mask

    def split(self):
        train_mask, valid_mask, test_mask = self._create_masks(20, 20, 200)
        return {
            "train": self.on(train_mask),
            "valid": self.on(valid_mask),
            "test": self.on(test_mask),
        }
        # return {
        #    "train": self.on(self.data.val_mask),
        #    "valid": self.on(self.data.test_mask),
        #    "test": self.on(self.data.train_mask),
        # }
