import torch
import random
from ugnn.tasks.base import ClassificationTask, RegressionTask


def _set_generator(num_nodes):
    num_nodes = random.randint(num_nodes // 4, num_nodes)
    nodes1 = list()
    nodes2 = list()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            nodes1.append(i)
            nodes2.append(j)
            nodes2.append(i)
            nodes1.append(j)
    return [nodes1, nodes2], num_nodes


class EntropyTask(RegressionTask):
    def __init__(self, nodes: int = 20, graphs: int = 100, **kwargs):
        def replicate(x):
            x = x / torch.sum(x)
            ret = -torch.mean(x * torch.log(x))
            # print(ret)
            return ret

        generated = [_set_generator(nodes) for _ in range(graphs)]
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
        x = torch.rand(graphs * nodes, 1)

        def rotate(x):
            x = torch.relu(x - random.random()) + 0.001
            return x

        labels = torch.zeros(nodes * graphs, 1)
        for graph in range(graphs):
            x[graph * nodes : (graph * nodes + generated[graph][1]), :] = rotate(
                x[graph * nodes : (graph * nodes + generated[graph][1]), :]
            )
            labels[graph * nodes : (graph * nodes + generated[graph][1])] = replicate(
                x[graph * nodes : (graph * nodes + generated[graph][1]), :]
            )
        super().__init__(x, edges, labels, **kwargs, mask_mask=mask_mask)
