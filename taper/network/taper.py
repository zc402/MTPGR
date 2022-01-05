"""
The overall architecture of TAPER
"""

from torch import nn

from .adjacency_matrix import AdjacencyMatrix
from .subnet import BoneNetwork, SpatialMean
from taper.kinematic import edges_dense, heights_dense

class TAPER(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 3  # TODO: change to 6, 3 for position params, 3 for rotation params ?
        out_channels = 8  # TODO: 8 or 8*4=32 ?

        # Build adj matrix with height layering partitioning strategy
        A = AdjacencyMatrix(edges_dense, heights_dense).get_height_config_adjacency()
        self.bone = BoneNetwork(in_channels, 256, A)
        self.sml = SpatialMean(256, out_channels)

    def forward(self, x):
        x = self.bone(x)
        x = self.sml(x)
        return x