import torch

from mtpgr.network.subnet.layers.st_layer import STLayer
from torch import nn
from mtpgr.utils.log import log


class BoneNetwork(nn.Module):
    """The STGCN bone network, contains mutliple STLayers. output shape NCTV"""

    def __init__(self, in_channels:int, out_channels:int, A, use_attention=True, depth=10):
        """
        Args:
            in_channels: The channel 'C' from input NCTV, num of features in a vertex.
            out_channels: Output channels. Default: 256
            A: Adjacency matrix of shape (num_spatial_labels, V, V)
        """
        super().__init__()
        self.register_buffer('A', A)
        num_spatial_labels = A.size(0)

        if depth == 10:
            log.debug("GCN depth: 10")
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
        elif depth == 4:
            log.debug("GCN depth: 4")
            self.st_layers = nn.ModuleList((
                STLayer(in_channels, 64, num_spatial_labels),
                STLayer(64, 128,  num_spatial_labels),
                STLayer(128, 256, num_spatial_labels),
                STLayer(256, out_channels, num_spatial_labels),
            ))
        else:
            raise NotImplementedError()

        if use_attention:
            log.debug("Attention enabled.")
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()), requires_grad=True)
                for i in self.st_layers
            ])
        else:
            log.debug("Attention disabled.")
            self.edge_importance = [1] * len(self.st_layers)


    def forward(self, x):
        # x shape: N,C,T,V. T: Temporal features; V: Spatial features
        N, C, T, V = x.size()

        for gcn, importance in zip(self.st_layers, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        return x

    @classmethod
    def from_config(cls, cfg):
        def initializer(in_channels:int, out_channels:int, A):
            return BoneNetwork(
                in_channels, 
                out_channels, 
                A, 
                use_attention=cfg.MODEL.ATTENTION,
                depth=cfg.MODEL.GCN_DEPTH)
        return initializer
