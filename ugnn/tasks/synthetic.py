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


def _count_square_cliques(edge_index, num_nodes, graphs):
    # https://cs.stackexchange.com/questions/64777/counting-the-number-of-squares-in-a-graph
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj_matrix[edge_index[0], edge_index[1]] = 1
    adj_matrix[edge_index[1], edge_index[0]] = 1

    n = num_nodes//graphs
    counts = torch.zeros((num_nodes), dtype=torch.long)
    for graph in range(graphs):
        for i in range(n):
            ni = graph*n+i
            for j in range(i+1, n):
                nj = graph*n+j
                if adj_matrix[ni, nj] == 0:
                    continue
                for k in range(j+1, n):
                    nk = graph*n+k
                    if adj_matrix[ni, nk] == 0 or adj_matrix[nj, nk] == 0:
                        continue
                    for m in range(k+1, n):
                        nm = graph*n+m
                        if adj_matrix[ni, nm] == 0 or adj_matrix[nj, nm] == 0 or adj_matrix[nk, nm] == 0:
                            continue
                        counts[ni] += 1
                        counts[nj] += 1
                        counts[nk] += 1
                        counts[nm] += 1
    return (counts>0).type(torch.LongTensor)


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


class SquareCliqueTask(RandomGraphTask):
    def __init__(self, *args, graphs=100, **kwargs):
        super().__init__(*args, **kwargs, graphs=graphs, replicate=lambda edges, nodes: _count_square_cliques(edges, nodes, graphs))


class TrianglesTask(RandomGraphTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, replicate=_count_triangles)


class DiffusionTask(ClassificationTask):
    def __init__(self,
                 nodes: int = 20,
                 max_density: float = 0.1,
                 graphs: int = 100,
                 alpha: float = 0.1,
                 feats: int = 16,
                 classes: int = 4,
                 **kwargs
                 ):
        edges = torch.cat(
            [graph * nodes + torch.tensor(_graph_generator(nodes, random.uniform(0, max_density)))
             for graph in range(graphs)],
            dim=1)
        from ugnn.architectures.appnp import APPNP
        x = torch.randn(graphs*nodes, feats)
        model = APPNP(feats, classes, alpha=alpha, hidden=feats)
        model.eval()
        out = model.forward(ClassificationTask(x, edges, None, classes=classes))
        out = out / out.mean(0, keepdim=True)[0]
        labels = out.argmax(dim=1)
        super().__init__(x, edges, labels, classes=classes, **kwargs)
