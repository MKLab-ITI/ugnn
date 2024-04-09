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
        index_smoothing_depth=0,  # 1 for adjacency (4 for de=True), None for only rni, 0 to deactivate
        rni=0, # 7 for 100 nodes
        de=False, # distance encoding instead of rni
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
        if de:
            from ugnn.utils import GraphConv
            gc = GraphConv(normalize=True, add_self_loops=False)
            x = torch.zeros(nodes, nodes).repeat(graphs, 1)
            for graph in range(graphs):
                for node1, node2 in zip(generated[graph][0][1], generated[graph][0][1]):
                    x[graph * nodes + node1][node2] = 1
            gamma = 1
            accum = x
            for i in range(index_smoothing_depth):
                gamma *= 0.9
                accum = accum + gamma*x
                x = gc(x, edges)
            x = accum
        elif index_smoothing_depth is None:
            pass
        else:
            x = torch.eye(nodes, nodes).repeat(graphs, 1)
            if index_smoothing_depth > 0:
                from ugnn.utils import GraphConv
                gc = GraphConv(normalize=index_smoothing_depth>1, add_self_loops=index_smoothing_depth>1)
                with torch.no_grad():
                    for _ in range(index_smoothing_depth):
                        x = gc(x, edges)

        if rni > 0:
            xnew = torch.ceil(torch.rand(nodes * graphs, rni)/0.01)*0.01
            if index_smoothing_depth is None:
                x = xnew
            else:
                x = torch.cat([x, xnew], dim=1)

        if graph_ids:
            graph_embeddings = torch.zeros(nodes * graphs, graphs)
            for graph in range(graphs):
                graph_embeddings[graph * nodes : (graph * nodes + nodes), graph] = 1
            x = torch.cat([x, graph_embeddings], dim=1)

        labels = replicate(edges, nodes * graphs)
        super().__init__(x, edges, labels, **kwargs, mask_mask=mask_mask)
