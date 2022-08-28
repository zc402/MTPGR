from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import logging

from mtpgr.network.predictor import Predictor
from torch import optim
from mtpgr.dataset import ConcatDataset
from mtpgr.config import get_cfg_defaults
from mtpgr.network import MTPGR

# joint xy coords -> gcn -> fcn
class Trainer:
    def __init__(self, predictor):
        self.predictor = predictor
        self.predictor.post_step = self.post_step

        self.model = predictor.model
        self.loss = CrossEntropyLoss()  # The input is expected to contain raw, unnormalized scores for each class.
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)
        # self.logger = self._logger_setup()

        self.step = 1

    def post_step(self, class_TC, label_T):
        # Cross Entropy, Input: (N*T, C), Target: (N*T).
        loss_tensor = self.loss(class_TC, label_T)
        self.opt.zero_grad()
        loss_tensor.backward()
        self.opt.step()

        if self.step % 100 == 0:
            print("Step: %d, Loss: %f" % (self.step, loss_tensor.item()))
            acc = self.acc(class_TC, label_T)
            print("Accuracy: {:.2f}".format(acc))

        if self.step % 2000 == 0:
            self.predictor.save_ckpt()

        self.step = self.step + 1

    def train(self):
        print("Training ...")
        for epoch in range(100):
            self.predictor.run_epoch()

    @staticmethod
    def acc(input, target):
        class_N = torch.argmax(input, dim=1)
        acc = (class_N == target).float().mean().item()
        return acc

    @classmethod
    def _data_loader(cls, cfg):  # Dataloader for training
        concat_dataset = ConcatDataset.from_config(cfg)
        train_loader = DataLoader(concat_dataset, batch_size=cfg.MODEL.BATCH_SIZE, shuffle=True, drop_last=True)
        return train_loader

    @classmethod
    def from_config(cls, cfg):
        predictor = Predictor.from_config(cfg, cls._data_loader(cfg))
        instance = Trainer(predictor)
        return instance

    # def _logger_setup(self):
    #     logger = logging.getLogger(__name__)
    #     console = logging.StreamHandler()
    #     # add the handler to the root logger
    #     logger.addHandler(console)
    #     logger.setLevel(logging.INFO)
    #     return logger

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    train_cfg = get_cfg_defaults()
    train_cfg.merge_from_file(Path('configs', 'default_model.yaml'))
    # train_cfg.merge_from_file(Path('configs', 'debug.yaml'))
    # train_cfg.merge_from_file(Path('configs', 'train_no_spatial_edges.yaml'))

    Trainer.from_config(train_cfg).train()
