from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch import optim
from mtpgr.config import get_cfg_defaults
from mtpgr.network import MTPGR
from mtpgr.utils.log import log

# joint xy coords -> gcn -> fcn
class Predictor:
    def __init__(self, dataloader, model, ckpt, device):

        self.data_loader = dataloader
        self.model = self._load_ckpt(model, ckpt, device)
        self.ckpt = ckpt
        self.device = device

    def post_step(self, pred, label):
        """
        Shapes:
            pred: (N*T, C)
            label: (N*T,)
        """
        pass

    def post_epoch(self):
        pass

    def run_epoch(self):
        for batch_data in self.data_loader:

            tensor_input = batch_data["kp"].to(self.device)  # shape: (N,T,V,C), network input expect: (N,C,T,V)
            tensor_input = torch.permute(tensor_input, (0, 3, 1, 2))
            tensor_label = batch_data["combine"].to(self.device)  # label shape: (N,T)

            tensor_pred = self.model(tensor_input)  # pred shape: (N*T,C)
            tensor_label = tensor_label.reshape([-1])  # label shape: (N*T,)

            self.post_step(tensor_pred, tensor_label)
        self.post_epoch()

    @staticmethod
    def _load_ckpt(model, ckpt, device):

        if ckpt.is_file():
            log.info("Checkpoint found. Resume from previous ckeckpoint")
            ckpt = torch.load(ckpt)
            model.load_state_dict(ckpt)
        else:
            log.warn("Checkpoint not found. Initialize random model parameters.")
            ckpt.parent.mkdir(exist_ok=True)
        model.to(device)
        return model

    def save_ckpt(self):
        torch.save(self.model.state_dict(), self.ckpt)
        log.info('Model saved')

    @classmethod
    def from_config(cls, cfg, data_loader):
        model = MTPGR.from_config(cfg)
        device = torch.device(cfg.MODEL.DEVICE)
        ckpt = Path(cfg.DATA_ROOT) / cfg.MODEL.CKPT_DIR / cfg.MODEL.MTPGR_CKPT
        instance = Predictor(data_loader, model, ckpt, device)
        return instance
