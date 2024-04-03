import torch
from ugnn.tasks.synthetic.utils import RandomGraphTask
import networkx as nx


def _diameters(edge_index, num_nodes):
    graph = nx.Graph()
    for u, v in zip(edge_index[0], edge_index[1]):
        graph.add_edge(int(u), int(v))

    ret = torch.zeros((num_nodes), dtype=torch.long)
    for i in range(num_nodes):
        if i in graph:
            ret[i] = max(nx.single_source_bellman_ford(graph, i)[0].values())
    return ret


class DiameterTask(RandomGraphTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         replicate=_diameters,
                         #deviation=lambda x, y: torch.mean((x-y).abs())
        )
