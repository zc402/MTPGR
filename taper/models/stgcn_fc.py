"""
ST-GCN
输入: N,C,T,V
输出：Class Score
"""
from taper.network.subnet.spatial_mean import SpatialMean
from taper.models import BaseModel


class STGCN_FC(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_pose = SpatialMean(2, 9)

        self._to_device()

    def _get_model_name(self) -> str:
        return "GCN-FC"

    def forward(self, x):
        return self.model_pose(x)