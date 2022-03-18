"""
The overall architecture of MTPGR
"""

from torch import nn

from .adjacency_matrix import AdjacencyMatrix
from .subnet import BoneNetwork, SpatialMean


class MTPGR(nn.Module):
    def __init__(self, edges, heights):
        """
        Traffic gesture recognizer
        :param edges: id array of shape (num_edges, 2)
        :param heights: dict {id: height_value}
        """
        super().__init__()
        in_channels = 3  # TODO: change to 6, 3 for position params, 3 for rotation params ?
        out_channels = 9  # TODO: 9 or 8*4=32 ?

        # Build adj matrix with height layering partitioning strategy
        A = AdjacencyMatrix(edges, heights).get_height_config_adjacency()
        self.bone = BoneNetwork(in_channels, 256, A)
        self.sml = SpatialMean(256, out_channels)

    def forward(self, x):
        x = self.bone(x)
        x = self.sml(x)
        return x

    @classmethod
    def from_config(cls, cfg):
        from mtpgr.kinematic import SparseToDense
        s2d = SparseToDense.from_config(cfg)
        heights = s2d.get_dense_id_height_map()
        edges = s2d.get_dense_edges()
        instance = MTPGR(edges, heights)
        return instance
