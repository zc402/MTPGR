from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import logging
from tqdm import tqdm
from mtpgr.network.predictor import Predictor
from torch import optim
from mtpgr.dataset.pgv2_dataset import PGv2TrainDataset
from mtpgr.config import get_cfg_defaults
from mtpgr.network import MTPGR
from mtpgr.analysis.chalearn_jaccard import ChaLearnJaccard

# joint xy coords -> gcn -> fcn
class Trainer:
    def __init__(self, predictor, num_classes):
        self.predictor = predictor
        self.predictor.post_step = self.post_step
        self.num_classes = num_classes

        self.model = predictor.model
        self.loss = CrossEntropyLoss()  # The input is expected to contain raw, unnormalized scores for each class.
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)
        # self.logger = self._logger_setup()

        self.step = 0

    def post_step(self, pred, label):
        """
        Shapes:
            pred: (N*T, C)
            label: (N*T,)
        """
        # Cross Entropy. Args - input: (N*T, C), target: (N*T).
        loss_tensor = self.loss(pred, label)
        self.opt.zero_grad()
        loss_tensor.backward()
        self.opt.step()

        if self.step % 100 == 0:
            print("Step: %d, Loss: %f" % (self.step, loss_tensor.item()))
            acc = self.acc(pred, label)
            print("Accuracy: {:.2f}".format(acc))
            self._jaccard(pred.cpu().detach().numpy(), label.cpu().numpy(), self.num_classes)

        if self.step % 1000 == 0:
            self.predictor.save_ckpt()

        self.step = self.step + 1

    def train(self):
        print("Training...")
        for epoch in tqdm(range(200)):
            self.predictor.run_epoch()

    @staticmethod
    def acc(input, target):
        class_N = torch.argmax(input, dim=1)
        acc = (class_N == target).float().mean().item()
        return acc
    
    @staticmethod
    def _jaccard(pred, gt, num_classes):
        # Convert to list([gt][pred])
        gt_pred_list = [(np.argmax(pred, axis=-1), gt)]
        J = ChaLearnJaccard(num_classes).mean_jaccard_index(gt_pred_list)
        print(f"Jaccard: {J}")

    @classmethod
    def _data_loader(cls, cfg):  # Dataloader for training
        concat_dataset = PGv2TrainDataset.from_config(cfg)
        train_loader = DataLoader(concat_dataset, batch_size=cfg.MODEL.BATCH_SIZE, shuffle=True, drop_last=True)
        return train_loader

    @classmethod
    def from_config(cls, cfg):
        predictor = Predictor.from_config(cfg, cls._data_loader(cfg))
        instance = Trainer(predictor, cfg.DATASET.NUM_CLASSES)
        return instance

    # def _logger_setup(self):
    #     logger = logging.getLogger(__name__)
    #     console = logging.StreamHandler()
    #     # add the handler to the root logger
    #     logger.addHandler(console)
    #     logger.setLevel(logging.INFO)
    #     return logger

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train_cfg = get_cfg_defaults()
    train_cfg.merge_from_file(Path('configs', 'default_model.yaml'))
    # train_cfg.merge_from_file(Path('configs', 'debug.yaml'))
    # train_cfg.merge_from_file(Path('configs', 'train_no_spatial_edges.yaml'))

    Trainer.from_config(train_cfg).train()
