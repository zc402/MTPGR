import torch

from mtpgr.network.subnet.layers.st_layer import STLayer
from torch import nn
from mtpgr.network.adjacency_matrix import AdjacencyMatrix

class BoneNetwork(nn.Module):
    """The STGCN bone network, contains mutliple STLayers. output shape NCTV"""

    def __init__(self, in_channels, out_channels, A):
        """
        :param in_channels: The channel 'C' from input NCTV, num of features in a vertex.
        :param out_channels: defaults to 256
        :param A:
        """
        super().__init__()
        edge_importance_weighting = True
        self.register_buffer('A', A)
        num_spatial_labels = A.size(0)

        self.st_layers = nn.ModuleList((
            STLayer(in_channels, 64, num_spatial_labels),
            STLayer(64, 64,   num_spatial_labels),
            STLayer(64, 64,   num_spatial_labels),
            STLayer(64, 64,   num_spatial_labels),
            STLayer(64, 128,  num_spatial_labels),
            STLayer(128, 128, num_spatial_labels),
            STLayer(128, 128, num_spatial_labels),
            STLayer(128, 256, num_spatial_labels),
            STLayer(256, 256, num_spatial_labels),
            STLayer(256, out_channels, num_spatial_labels),
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()), requires_grad=True)
                for i in self.st_layers
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)


    def forward(self, x):
        # x shape: N,C,T,V. T: Temporal features; V: Spatial features
        N, C, T, V = x.size()

        for gcn, importance in zip(self.st_layers, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        return x
