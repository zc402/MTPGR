"""
The overall architecture of TAPER
"""

from torch import nn

from .adjacency_matrix import AdjacencyMatrix
from .subnet import BoneNetwork, SpatialMean

class TAPER(nn.Module):
    def __init__(self, edges_dense, heights_dense):
        super().__init__()
        in_channels = 3  # TODO: change to 6, 3 for position params, 3 for rotation params ?
        out_channels = 9  # TODO: 9 or 8*4=32 ?

        # Build adj matrix with height layering partitioning strategy
        A = AdjacencyMatrix(edges_dense, heights_dense).get_height_config_adjacency()
        self.bone = BoneNetwork(in_channels, 256, A)
        self.sml = SpatialMean(256, out_channels)

    def forward(self, x):
        x = self.bone(x)
        x = self.sml(x)
        return x