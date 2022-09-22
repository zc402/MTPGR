import torch

from torch import nn

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

        self.fcn = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_channels, out_channels),
            # nn.ReLU(inplace=True),  # new
            # nn.Linear(128, out_channels),  # new
        )

    def forward(self, x):
        # x shape: N,C,T,V. T: Temporal features; V: Spatial features
        N, C, T, V = x.size()

        # 把V平均、C连接dense
        x = x.mean(dim=3)  # NCT
        x = x.permute(0, 2, 1)  # NTC
        x = x.reshape([N*T, C])  # N*T, C
        x = self.fcn(x)

        # x = x.view(N, T, C)
        return x

class SparseConnect(nn.Module):
    """one weight per V x label
    Idea from "focus on the hands"
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels

        self.fcn = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_channels, out_channels),
        )

    def forward(self, x):
        # x shape: N,C,T,V. T: Temporal features; V: Spatial features
        N, C, T, V = x.size()

        x = x.permute(0, 2, 3, 1)  # N T V C
        x = x.reshape([N*T*V, C])  # N*T, C
        x = self.fcn(x)
        x = x.view(N*T, V, self.out_channels)
        x = x.sum(1)  # N*T, C
        return x