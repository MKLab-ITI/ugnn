import torch
from ugnn.tasks.synthetic.utils import RandomGraphTask


def _count_triangles(edge_index, num_nodes, graphs):
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj_matrix[edge_index[0], edge_index[1]] = 1
    adj_matrix[edge_index[1], edge_index[0]] = 1

    n = num_nodes // graphs
    counts = torch.zeros((num_nodes), dtype=torch.long)
    for graph in range(graphs):
        for i in range(n):
            ni = graph * n + i
            for j in range(i + 1, n):
                nj = graph * n + j
                if adj_matrix[ni, nj] == 0:
                    continue
                for k in range(j + 1, n):
                    nk = graph * n + k
                    if adj_matrix[ni, nk] == 0 or adj_matrix[nj, nk] == 0:
                        continue
                    counts[ni] += 1
                    counts[nj] += 1
                    counts[nk] += 1
    return counts.type(torch.LongTensor) // 3


class TrianglesTask(RandomGraphTask):
    def __init__(self, *args, graphs=100, **kwargs):
        super().__init__(
            *args,
            graphs=graphs,
            **kwargs,
            replicate=lambda edges, nodes: _count_triangles(edges, nodes, graphs)
        )
