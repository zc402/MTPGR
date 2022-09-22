"""
The overall architecture of MTPGR
"""

from torch import nn

from .adjacency_matrix import AdjacencyMatrix
from mtpgr.network.subnet.bone_network import BoneNetwork
from mtpgr.network.subnet.fuse_layer import SpatialMean, SparseConnect


class MTPGR(nn.Module):
    def __init__(self, adjacency_matrix, num_classes, bone_net, fuse="mean"):
        """
        Traffic gesture recognizer
        :param edges: id array of shape (num_edges, 2)
        :param heights: dict {id: height_value}
        """
        super().__init__()
        in_channels = 3  # Number of channels, 2 for 2D coords, 3 for 3D coords
        out_channels = num_classes  # Number of gesture classes

        # Build adj matrix with height layering partitioning strategy
        A = adjacency_matrix.get_adjacency()
        
        self.bone = bone_net(in_channels, 256, A)
        if fuse == "mean":
            self.sml = SpatialMean(256, out_channels)
        elif fuse == "sparse":
            self.sml = SparseConnect(256, out_channels)

    def forward(self, x):
        x = self.bone(x)
        x = self.sml(x)
        return x

    @classmethod
    def from_config(cls, cfg):
        # from mtpgr.kinematic import SparseToDense
        # s2d = SparseToDense.from_config(cfg)
        # heights = s2d.get_dense_id_height_map()
        # edges = s2d.get_dense_edges()
        bone_net = BoneNetwork.from_config(cfg)
        adjacency_mat = AdjacencyMatrix.from_config(cfg)
        fuse_strategy = cfg.MODEL.FUSE
        instance = MTPGR(adjacency_mat, num_classes=cfg.DATASET.NUM_CLASSES, bone_net=bone_net, fuse=fuse_strategy)
        
        return instance
