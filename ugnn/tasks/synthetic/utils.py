import random
import torch
from ugnn.tasks.base import ClassificationTask


def graph_generator(num_nodes, density):
    num_nodes = random.randint(num_nodes // 4, num_nodes)
    nodes1 = list()
    nodes2 = list()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (
                j == i + 1 or random.uniform(0, 1) <= density
            ):  # for density=1, we get a fully connected graph
                nodes1.append(i)
                nodes2.append(j)
                nodes2.append(i)
                nodes1.append(j)
    return [nodes1, nodes2], num_nodes


class RandomGraphTask(ClassificationTask):
    def __init__(
        self,
        nodes: int = 20,
        max_density: float = 0.1,
        graphs: int = 100,
        replicate=None,
        graph_ids=False,
        index_smoothing_depth=0,
        **kwargs
    ):
        if replicate is None:
            raise Exception("Must provide a method to replicate")
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
        x = torch.eye(nodes, nodes).repeat(graphs, 1)

        if index_smoothing_depth > 0:
            from ugnn.utils import GraphConv

            gc = GraphConv()
            with torch.no_grad():
                for _ in range(index_smoothing_depth):
                    x = gc(x, edges)

        if graph_ids:
            graph_embeddings = torch.zeros(nodes * graphs, graphs)
            for graph in range(graphs):
                graph_embeddings[graph * nodes : (graph * nodes + nodes), graph] = 1
            x = torch.cat([x, graph_embeddings], dim=1)

        labels = replicate(edges, nodes * graphs)
        super().__init__(x, edges, labels, **kwargs, mask_mask=mask_mask)
