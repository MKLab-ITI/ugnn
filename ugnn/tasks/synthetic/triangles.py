import torch
from ugnn.tasks.synthetic.utils import RandomGraphTask


def _count_triangles(edge_index, num_nodes):
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj_matrix[edge_index[0], edge_index[1]] = 1
    adj_cube = torch.matrix_power(adj_matrix, 3)
    triangle_counts = adj_cube.diagonal() // 2
    return triangle_counts.type(torch.LongTensor)



class TrianglesTask(RandomGraphTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, replicate=_count_triangles)
