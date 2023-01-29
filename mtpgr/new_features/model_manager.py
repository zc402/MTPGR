from pathlib import Path
import torch
import numpy as np

from mtpgr.network.mtpgr_net import MTPGR
from mtpgr.new_features.adjacency_matrix_v2 import AdjacencyMatrixV2
from mtpgr.new_features.isolated_model import IsolatedModel
from mtpgr.new_features.parts_v2 import PartsV2

class ModelManager:

    def __init__(self, cfg) -> None:

        model_name = f"{cfg.MODEL.GRAPH}_{cfg.MODEL.STRATEGY}"
        
        self.ckpt = Path(cfg.DATA_ROOT, cfg.MODEL.CKPT_DIR, model_name)
        self.partsV2: PartsV2 = PartsV2.from_config(cfg)
        self.adj_v2: AdjacencyMatrixV2 = AdjacencyMatrixV2.from_config(cfg)

        model_init = IsolatedModel.from_config(cfg)
        self.model = model_init(3, cfg.DATASET.NUM_CLASSES, self.adj_v2.get_adjacency())
        self.model.cuda()

    def load_ckpt(self):

        if self.ckpt.is_file():
            print("Checkpoint found. Resume from previous ckeckpoint")
            ckpt_data = torch.load(self.ckpt)
            self.model.load_state_dict(ckpt_data)
        else:
            print("Checkpoint not found. Initialize random model parameters.")

    def save_ckpt(self):
        self.ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.ckpt)

    def prepare_data(self, batch):
        x = self.partsV2.aggregate_features(batch['Vp'], batch['Vr'], batch['Vc'], batch['Vb'])
        y_true = batch['combine'][:, 0]
        # x: NTVC -> NCTV
        x = x.permute((0, 3, 1, 2))
        x = x.cuda()
        y_true = y_true.cuda()
        return x, y_true
