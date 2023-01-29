import argparse
from pathlib import Path
import pickle
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from mtpgr.network.predictor import Predictor
from torch import optim
from mtpgr.new_features.isolated_dataset import IsolatedDataset
from mtpgr.config.defaults import get_cfg_defaults
from mtpgr.new_features.model_manager import ModelManager
# from mtpgr.analysis.chalearn_jaccard import ChaLearnJaccard
from mtpgr.utils.log import log
# joint xy coords -> gcn -> fcn
class Trainer:
    def __init__(self, cfg, ):

        self.debug = cfg.MODEL.DEBUG

        if self.debug == True:
            self.num_workers = 0
            self.save_debug_img = True
            
        elif self.debug == False:
            self.num_workers = min(cfg.NUM_CPU, 10)
            self.save_debug_img = False
        
        self.batch_size = cfg.MODEL.BATCH_SIZE
        self.max_epochs = cfg.DATASET.EPOCHS

        self.train_dataset = IsolatedDataset(cfg, 'train', 'random', do_augment=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=cfg.MODEL.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=self.num_workers,)

        self.test_dataset = IsolatedDataset(cfg, 'test', 'global', do_augment=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=self.num_workers, )

        self.mm = ModelManager(cfg)
        self.model = self.mm.model
        self.loss = CrossEntropyLoss()  # The input is expected to contain raw, unnormalized scores for each class.

        self.num_step = 0
        self.max_historical_acc = 0.

        self.mm.load_ckpt()
        self.optim = optim.Adam(self.mm.model.parameters(), lr=cfg.MODEL.LEARNING_RATE)

    def train_batches(self):

        loss_list = []
        pred_class_list = []
        true_class_list = []
        for step, batch_data in enumerate(self.train_loader):
            x, y_true = self.mm.prepare_data(batch_data)
            self.model.train()
            y = self.model(x)
            y_pred_class = torch.argmax(y, dim=1)

            loss_tensor = self.loss(y, y_true)
            self.optim.zero_grad()
            loss_tensor.backward()
            self.optim.step()

            loss_list.append(loss_tensor.item())
            pred_class_list.append(y_pred_class)
            true_class_list.append(y_true)
        
        acc = (torch.concat(pred_class_list) == torch.concat(true_class_list)).float().mean().item()
        log.info("Train Acc: %f, Loss: %f" % (acc, np.mean(loss_list)))

    def train_epochs(self):
        log.info("Training...")
        max_acc = 0.0
        for epoch in range(self.max_epochs):
            self.train_batches()

            if epoch % 2 == 0:
                acc, _ = self.eval()
                if acc > max_acc:
                    self.mm.save_ckpt()
                    print('New best acc %.3f, Checkpoint saved' % acc)
                    max_acc = acc
                else:
                    print('Not save, current acc: %.3f, best acc: %.3f' % (acc, max_acc))
                

    def eval(self):
        y_pred_list = []
        y_true_list = []
        for batch_data in tqdm(self.test_loader):
            # batch size = 1
            x, y_true = self.mm.prepare_data(batch_data)
            self.model.eval()
            with torch.no_grad():
                y = self.model(x)
            y_pred_class = torch.argmax(y, dim=1)

            y_pred_list.append(y_pred_class)
            y_true_list.append(y_true)

        acc = (torch.tensor(y_pred_list) == torch.tensor(y_true_list)).float().mean().item()
        results = {'y_pred': torch.tensor(y_pred_list), 'y_true': torch.tensor(y_true_list)}
        print(f'Test Accuracy: {acc}')
        return acc, results


if __name__ == '__main__':

    train_cfg = get_cfg_defaults()
    Trainer(train_cfg).train_epochs()

    acc, res = Trainer(train_cfg).eval()
    with Path('eval_result.pkl').open('wb') as f:
        pickle.dump(res, f)
