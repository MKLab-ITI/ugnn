import torch
import random
from ugnn.tasks.task import ClassificationTask


def _graph_generator(num_nodes, density):
    nodes1 = list()
    nodes2 = list()
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if j==i+1 or random.uniform(0, 1) < density:
                nodes1.append(i)
                nodes2.append(j)
                nodes2.append(i)
                nodes1.append(j)
    return [nodes1, nodes2]


def _count_neighbors(edge_index, num_nodes):
    return torch.bincount(edge_index.view(-1), minlength=num_nodes) // 2


def _count_triangles(edge_index, num_nodes):
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj_matrix[edge_index[0], edge_index[1]] = 1
    adj_cube = torch.matrix_power(adj_matrix, 3)
    triangle_counts = adj_cube.diagonal() // 2
    return triangle_counts.type(torch.LongTensor)


class RandomGraphTask(ClassificationTask):
    def __init__(self,
                 nodes: int = 20,
                 max_density: float = 0.1,
                 graphs: int = 100,
                 replicate=None,
                 **kwargs
                 ):
        if replicate is None:
            raise Exception("Must provide a method to replicate")
        edges = torch.cat(
            [graph * nodes + torch.tensor(_graph_generator(nodes, random.uniform(0, max_density)))
             for graph in range(graphs)],
            dim=1)
        x = torch.eye(nodes, nodes).repeat(graphs, 1)
        labels = replicate(edges, nodes * graphs)
        super().__init__(x, edges, labels, **kwargs)


class DegreeTask(RandomGraphTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, replicate=_count_neighbors)


class TrianglesTask(RandomGraphTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, replicate=_count_triangles)
