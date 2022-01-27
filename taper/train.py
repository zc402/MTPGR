from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import logging

from torch import optim
from taper.dataset import ConcatVideo
from taper.config import get_cfg_defaults
from taper.network import TAPER

# joint xy coords -> gcn -> fcn
class Trainer:
    def __init__(self):
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file(Path('configs', 'train.yaml'))
        self.data_loader = self.train_data_loader(self.cfg)
        self.model = self.train_model(self.cfg)
        self.loss = CrossEntropyLoss()  # The input is expected to contain raw, unnormalized scores for each class.
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)
        self.logger = logging.getLogger(__name__)

    def train(self):
        device = torch.device(self.cfg.MODEL.DEVICE)
        step = 1
        for epoch in range(100):
            for train_data in self.data_loader:
                tensor_NCTV = train_data['tensor_ctv'].to(device)  # Batch is N dim
                label_NT = train_data['label_t'].to(device)  # N,T

                class_out = self.model(tensor_NCTV)  # Out: N*T,C
                label_NT = label_NT.reshape([-1])  # N*T

                # Cross Entropy, Input: (N, C), Target: (N).
                loss_tensor = self.loss(class_out, label_NT)
                self.opt.zero_grad()
                loss_tensor.backward()
                self.opt.step()

                if step % 100 == 0:
                    print("Step: %d, Loss: %f" % (step, loss_tensor.item()))
                    acc = self.acc(class_out, label_NT)
                    print("Accuracy: {:.2f}".format(acc))

                if step % 2000 == 0:
                    self.save_ckpt(self.cfg, self.model)
                    print('Model save')
                step = step + 1

    @staticmethod
    def train_data_loader(cfg):
        concat_dataset = ConcatVideo.from_config(cfg)
        train_loader = DataLoader(concat_dataset, batch_size=cfg.MODEL.BATCH_SIZE, shuffle=True, drop_last=True)
        return train_loader

    @staticmethod
    def train_model(cfg):

        model = TAPER.from_config(cfg)
        ckpt_path = Path(cfg.DATA_ROOT) / cfg.MODEL.CKPT_DIR / cfg.MODEL.TAPER_CKPT
        if ckpt_path.is_file():
            print("Resume from previous ckeckpoint")
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt)
        else:
            print("Previous checkpoint not found.")
            print("Start the training from scratch!")
            ckpt_path.parent.mkdir(exist_ok=True)
        model.train()
        device = torch.device(cfg.MODEL.DEVICE)
        model.to(device)
        return model

    @staticmethod
    def save_ckpt(cfg, model):
        save_path = Path(cfg.DATA_ROOT) / cfg.MODEL.CKPT_DIR / cfg.MODEL.TAPER_CKPT
        torch.save(model.state_dict(), save_path)

    @staticmethod
    def acc(input, target):
        class_N = torch.argmax(input, dim=1)
        acc = (class_N == target).float().mean().item()
        return acc

if __name__ == '__main__':
    Trainer().train()