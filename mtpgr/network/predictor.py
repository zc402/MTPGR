from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import logging

from torch import optim
from mtpgr.dataset import ConcatVideo
from mtpgr.config import get_cfg_defaults
from mtpgr.network import MTPGR

# joint xy coords -> gcn -> fcn
class Predictor:
    def __init__(self, dataloader, model, ckpt, device):

        self.data_loader = dataloader
        self.model = self._load_ckpt(model, ckpt, device)
        self.ckpt = ckpt
        self.device = device
        self.logger = logging.getLogger(__name__)

    def post_step(self, class_TC, label_T):
        raise NotImplementedError()

    def run_epoch(self):
        for train_data in self.data_loader:
            tensor_NCTV = train_data['tensor_ctv'].to(self.device)  # Batch is N dim
            label_NT = train_data['label_t'].to(self.device)  # N,T

            class_TC = self.model(tensor_NCTV)  # Out: N*T,C
            label_T = label_NT.reshape([-1])  # N*T

            self.post_step(class_TC, label_T)

    @staticmethod
    def _load_ckpt(model, ckpt, device):

        if ckpt.is_file():
            print("Resume from previous ckeckpoint")
            ckpt = torch.load(ckpt)
            model.load_state_dict(ckpt)
        else:
            print("Initialize random model parameters.")
            ckpt.parent.mkdir(exist_ok=True)
        model.to(device)
        return model

    def save_ckpt(self):
        torch.save(self.model.state_dict(), self.ckpt)
        print('Model save')

    @classmethod
    def from_config(cls, cfg, data_loader):
        model = MTPGR.from_config(cfg)
        device = torch.device(cfg.MODEL.DEVICE)
        ckpt = Path(cfg.DATA_ROOT) / cfg.MODEL.CKPT_DIR / cfg.MODEL.MTPGR_CKPT
        instance = Predictor(data_loader, model, ckpt, device)
        return instance
