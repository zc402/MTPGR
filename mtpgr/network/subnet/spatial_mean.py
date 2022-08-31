import torch

from torch import nn
from .bone_network import BoneNetwork

class SpatialMean(nn.Module):
    """ STGCN, output num_class scores
    Input N,C,T,V
    N: batch size
    C: coordinate axes, (x,y,z) for example
    T: time
    V: num of parts
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.fcn = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # x shape: N,C,T,V. T: Temporal features; V: Spatial features
        N, C, T, V = x.size()

        # 把V平均、C连接dense
        x = x.mean(dim=3)  # NCT
        x = x.permute(0, 2, 1)  # NTC
        x = x.reshape([N*T, 256])  # N*T, C
        x = self.fcn(x)

        # x = x.view(N, T, C)
        return x
