import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from mtpgr.network.predictor import Predictor
from torch import optim
from mtpgr.dataset.pgv2_dataset import PGv2TrainDataset
from mtpgr.config import get_cfg_defaults
from mtpgr.analysis.chalearn_jaccard import ChaLearnJaccard
from mtpgr.utils.log import log
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

    def post_step(self, pred, label, **kwargs):
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
            log.info("Step: %d, Loss: %f" % (self.step, loss_tensor.item()))
            self._report(pred, label)

        self.step = self.step + 1

    def train(self):
        log.info("Training...")
        for epoch in tqdm(range(200)):
            self.predictor.run_epoch()
            if epoch % 50 == 0:
                self.predictor.save_ckpt()
        self.predictor.save_ckpt()

    def _report(self, pred, label):
        acc = self.acc(pred, label)
        log.info("Accuracy: {:.2f}".format(acc))
        self._jaccard(pred.cpu().detach().numpy(), label.cpu().numpy(), self.num_classes)

    @staticmethod
    def acc(input, target):
        class_N = torch.argmax(input, dim=1)
        acc = (class_N == target).float().mean().item()
        return acc
    
    @staticmethod
    def _jaccard(pred, gt, num_classes):
        # Convert to list([gt][pred])
        gt_pred_list = [(np.argmax(pred, axis=-1), gt)]
        J, _, _ = ChaLearnJaccard(num_classes).mean_jaccard_index(gt_pred_list)
        log.info(f"Jaccard: {J}")

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
    parser = argparse.ArgumentParser(description='Trainer for monocular traffic police gesture recognizer')
    parser.add_argument('-c', '--config', type=str, default="default_model.yaml", help='Enter the file name of a configuration from configs folder')
    args = parser.parse_args()
    config_path:Path = Path('configs', args.config)
    if not config_path.is_file():
        log.error(f"No such config file: {config_path}")

    train_cfg = get_cfg_defaults()
    train_cfg.merge_from_file(config_path)

    Trainer.from_config(train_cfg).train()
