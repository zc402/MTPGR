"""
ST-GCN
输入: N,C,T,V
输出：Class Score
"""
from pathlib import Path
from taper.stgcn.st_gcn_fc import StgcnFc
from taper.models import BaseModel
import torch

class STGCN_FC(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_pose = StgcnFc(2, 9)

        self._to_device()

    def _get_model_name(self) -> str:
        return "GCN-FC"

    def forward(self, x):
        return self.model_pose(x)