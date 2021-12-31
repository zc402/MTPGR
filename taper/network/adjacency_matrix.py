import numpy as np
import torch

# from network.kinematic import edges, heights, p2pat


class AdjacencyMatrix:
    # Adj matrix according to height layering partitioning strategy

    def __init__(self, edges, heights):
        """

        :param edges: spatially connected parts. 2d array of shape (num_edge, 2)
        :param heights: dict of height values. dict[part] = height
        """
        num_nodes = len(heights.keys())

        adjacency = np.zeros((num_nodes, num_nodes))  # 邻接矩阵

        for i, j in edges:  # Spatially connected edges
            adjacency[j, i] = 1
            adjacency[i, j] = 1

        self.adjacency = adjacency

        # A将邻接矩阵拆成3组，分别对应3个高度差标签。3组加在一起后等于邻接矩阵。
        # A.shape: (labels, target_vertices, neighbors)
        A = np.zeros((3, num_nodes, num_nodes))  # 3: number of labels (lower, equal, higher)
        for root in range(num_nodes):
            for j in range(num_nodes):
                if adjacency[root, j] == 1:
                    # ij相邻
                    hr = heights[root]  # 高度
                    hj = heights[j]
                    if hj - hr > 0:  # 邻接点在root之上
                        A[2, root, j] = 1
                    elif hj - hr < 0:
                        A[0, root, j] = 1
                    else:  # 高度一致。例如左右胯部。
                        A[1, root, j] = 1
        # 每个关键点与自身高度一致，添加到自身的连接
        A[1] = A[1] + np.eye(num_nodes)
        assert 0 <= A.any() <= 1
        assert 0 <= A.sum(axis=0).any() <= 1
        self.A = torch.tensor(A, dtype=torch.float32, requires_grad=False)

    def get_adjacency(self):
        """返回邻接矩阵"""
        return self.adjacency

    def get_height_config_adjacency(self):
        """返回以关键点高度进行配置的邻接矩阵，比邻接矩阵多一个label维度"""
        return self.A
