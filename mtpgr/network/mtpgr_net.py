"""
The overall architecture of MTPGR
"""

from torch import nn

from .adjacency_matrix import AdjacencyMatrix
from .subnet import BoneNetwork, SpatialMean


class MTPGR(nn.Module):
    def __init__(self, edges, heights, num_classes, bone_net):
        """
        Traffic gesture recognizer
        :param edges: id array of shape (num_edges, 2)
        :param heights: dict {id: height_value}
        """
        super().__init__()
        in_channels = 3  # Number of channels, 2 for 2D coords, 3 for 3D coords
        out_channels = num_classes  # Number of gesture classes

        # Build adj matrix with height layering partitioning strategy
        A = AdjacencyMatrix(edges, heights).get_height_config_adjacency()
        self.bone = bone_net(in_channels, 256, A)
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
        bone_net = BoneNetwork.from_config(cfg)
        instance = MTPGR(edges, heights, num_classes=cfg.DATASET.NUM_CLASSES, bone_net=bone_net)
        return instance
