import numpy as np
import torch
from mtpgr.kinematic.parts import Parts
# from network.kinematic import edges, heights, p2pat


class AdjacencyMatrix:
    # Adj matrix according to height layering partitioning strategy

    def __init__(self, part_names, heights, edges):
        """

        :param edges: spatially connected parts. 2d array of shape (num_edge, 2)
        :param heights: dict of height values. dict[part] = height
        """
        num_nodes = len(part_names)

        adjacency = np.zeros((num_nodes, num_nodes))  # Adjacency matrix, 

        for i, j in edges:  # Spatially connected edges, bi-direction.
            adjacency[j, i] = 1
            adjacency[i, j] = 1
        adjacency = adjacency + np.eye(num_nodes)  # loop edge.
        
        normalizing_term = self.normalize_digraph(adjacency)

        # A is stacked by 3 adjacency matrix, corresponding to 3 height labels - lower, equal, higher
        # A.shape: (labels, target_vertices, neighbors)
        A = np.zeros((3, num_nodes, num_nodes))
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
        assert 0 <= A.any() <= 1
        assert 0 <= A.sum(axis=0).any() <= 1

        A = A * normalizing_term
        self.A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
    
    def normalize_digraph(self, A):
        # If a vertex is connected to 2 vertices and itself, 
        # then each of the vertex's contribution is reduced to 1/3
        # Args:
        #     A: adjacency matrix, shape: (num_node, num_node).
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD

    def get_height_config_adjacency(self):
        """返回以关键点高度进行配置的邻接矩阵，比邻接矩阵多一个label维度"""
        return self.A

    @classmethod
    def from_config(cls, cfg):
        parts = Parts.from_config(cfg)
        part_names = parts.get_parts()
        heights = parts.get_heights()
        edges = parts.get_edge_indices()
        inst = AdjacencyMatrix(part_names=part_names, heights=heights, edges=edges)
        return inst


# class AdjacencyMatrix:
#     # Adj matrix according to height layering partitioning strategy

#     def __init__(self, part_names, heights, edges):
#         """

#         :param edges: spatially connected parts. 2d array of shape (num_edge, 2)
#         :param heights: dict of height values. dict[part] = height
#         """
#         num_nodes = len(part_names)

#         adjacency = np.zeros((num_nodes, num_nodes))  # Adjacency matrix, 

#         for i, j in edges:  # Spatially connected edges, bi-direction.
#             adjacency[j, i] = 1
#             adjacency[i, j] = 1
#         adjacency = adjacency + np.eye(num_nodes)  # loop edge.
        
#         normalizing_term = self.normalize_digraph(adjacency)

#         # A.shape: (labels, target_vertices, neighbors)
#         A = np.zeros((4, num_nodes, num_nodes))
#         for root in range(num_nodes):
#             for j in range(num_nodes):
#                 if adjacency[root, j] == 1:
#                     # ij相邻
#                     hr = heights[root]  # 高度
#                     hj = heights[j]
#                     if hj - hr == 1:  # Higher
#                         A[2, root, j] = 1
#                     elif hj - hr == -1:  # Lower
#                         A[0, root, j] = 1
#                     elif hj - hr == 0:  # Same hight
#                         A[1, root, j] = 1
#                     else:  # Height is much higher or much lower
#                         A[3, root, j] = 1
                        
#         assert 0 <= A.any() <= 1
#         assert 0 <= A.sum(axis=0).any() <= 1

#         A = A * normalizing_term
#         self.A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
    
#     def normalize_digraph(self, A):
#         # If a vertex is connected to 2 vertices and itself, 
#         # then each of the vertex's contribution is reduced to 1/3
#         # Args:
#         #     A: adjacency matrix, shape: (num_node, num_node).
#         Dl = np.sum(A, 0)
#         num_node = A.shape[0]
#         Dn = np.zeros((num_node, num_node))
#         for i in range(num_node):
#             if Dl[i] > 0:
#                 Dn[i, i] = Dl[i]**(-1)
#         AD = np.dot(A, Dn)
#         return AD

#     def get_height_config_adjacency(self):
#         """返回以关键点高度进行配置的邻接矩阵，比邻接矩阵多一个label维度"""
#         return self.A

#     @classmethod
#     def from_config(cls, cfg):
#         parts = Parts.from_config(cfg)
#         part_names = parts.get_parts()
#         heights = parts.get_heights()
#         edges = parts.get_edge_indices()
#         inst = AdjacencyMatrix(part_names=part_names, heights=heights, edges=edges)
#         return inst