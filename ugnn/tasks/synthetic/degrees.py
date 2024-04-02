import torch
from ugnn.tasks.synthetic.utils import RandomGraphTask


def _count_neighbors(edge_index, num_nodes):
    return torch.bincount(edge_index.view(-1), minlength=num_nodes) // 2


class DegreeTask(RandomGraphTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         replicate=_count_neighbors,
                         #deviation=lambda x, y: torch.mean((x-y).abs())
        )
