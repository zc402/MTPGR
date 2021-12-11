from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from taper.dataset.traffic import TrafficGesClips
from taper.models.stgcn_fc import STGCN_FC

from torch import optim


# joint xy coords -> gcn -> fcn
class GcnTrainer:
    def __init__(self):
        self.batch_size = 10
        self.clip_len = 150
        self.dataset = TrafficGesClips(self.clip_len)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.model = STGCN_FC()
        self.model.train()
        self.loss = CrossEntropyLoss()  # The input is expected to contain raw, unnormalized scores for each class.
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self):
        step = 1
        self.model.load_ckpt()
        for epoch in range(100):
            for ges_data in self.data_loader:
                tensor_NCTV = ges_data['tensor_ctv']
                # Expect: N,C,T,V
                # features = features.to(self.model.device, dtype=torch.float32)

                class_out = self.model(tensor_NCTV)  # Out: N*T,C
                label_NT = ges_data['label_t']  # N,T
                label_NT = label_NT.reshape([-1])  # N*T
                # target = target.to(self.model.device, dtype=torch.long)

                # Cross Entropy, Input: (N, C), Target: (N).
                loss_tensor = self.loss(class_out, label_NT)
                self.opt.zero_grad()
                loss_tensor.backward()
                self.opt.step()

                if step % 100 == 0:
                    print("Step: %d, Loss: %f" % (step, loss_tensor.item()))
                if step % 2000 == 0:
                    self.model.save_ckpt()
                step = step + 1
