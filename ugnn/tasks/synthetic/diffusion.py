import torch
import random
from ugnn.tasks.base import RegressionTask, ClassificationTask
from ugnn.tasks.synthetic.utils import graph_generator


class DiffusionTask(RegressionTask):
    # features -> scores
    def __init__(
        self,
        nodes: int = 20,
        max_density: float = 0.1,
        graphs: int = 100,
        alpha: float = 0.1,
        feats: int = 16,
        classes: int = 4,
        **kwargs
    ):
        generated = [
            graph_generator(nodes, random.uniform(0, max_density))
            for _ in range(graphs)
        ]
        edges = torch.cat(
            [
                graph * nodes + torch.tensor(generated[graph][0])
                for graph in range(graphs)
            ],
            dim=1,
        )
        mask_mask = torch.zeros(nodes * graphs, dtype=torch.bool)
        for graph in range(graphs):
            for node in range(generated[graph][1]):
                mask_mask[graph * nodes + node] = 1

        from ugnn.architectures.appnp import APPNP

        x = torch.rand(graphs * nodes, feats)
        model = APPNP(feats, classes, alpha=alpha, hidden=feats)
        model.eval()
        out = model.forward(ClassificationTask(x, edges, None, classes=classes))
        super().__init__(x, edges, out, classes=classes, mask_mask=mask_mask, **kwargs)
