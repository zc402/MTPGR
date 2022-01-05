from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from taper.models.stgcn_fc import STGCN_FC

from torch import optim
from taper.dataset import SingleVideo, ConcatVideo
from taper.kinematic import dense_indices
from taper.config import get_cfg_defaults
from taper.network import TAPER

# joint xy coords -> gcn -> fcn
class GcnTrainer:
    def __init__(self):
        self.cfg = get_cfg_defaults()
        self.data_loader = self.train_data_loader(self.cfg)
        self.model = self.train_model(self.cfg)
        self.loss = CrossEntropyLoss()  # The input is expected to contain raw, unnormalized scores for each class.
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self):
        step = 1
        for epoch in range(100):
            for train_data in self.data_loader:
                tensor_NCTV = train_data['tensor_ctv']  # Batch is N dim
                label_NT = train_data['label_t']  # N,T

                class_out = self.model(tensor_NCTV)  # Out: N*T,C
                label_NT = label_NT.reshape([-1])  # N*T

                # Cross Entropy, Input: (N, C), Target: (N).
                loss_tensor = self.loss(class_out, label_NT)
                self.opt.zero_grad()
                loss_tensor.backward()
                self.opt.step()

                if step % 100 == 0:
                    print("Step: %d, Loss: %f" % (step, loss_tensor.item()))
                if step % 2000 == 0:
                    self.save_ckpt(self.cfg, self.model)
                step = step + 1

    @staticmethod
    def train_data_loader(cfg):
        vibe_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.VIBE_DIR
        vibe_list = vibe_folder.glob('*.npy')
        label_list = [Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.LABEL_DIR / (name + '.json5') for name in vibe_list]

        video_dataset_list = [SingleVideo(v, l, dense_indices) for v, l in zip(vibe_list, label_list)]
        concat_dataset = ConcatVideo(video_dataset_list, cfg.TRAIN.CLIP_LEN)
        train_loader = DataLoader(concat_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True)
        return train_loader

    @staticmethod
    def train_model(cfg):
        model = TAPER()
        ckpt_path = Path(cfg.CKPT_DIR) / cfg.TAPER_CKPT
        if ckpt_path.is_file():
            print("Resume from previous ckeckpoint")
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt)
        else:
            print("Previous checkpoint not found.")
            print("Start the training from scratch!")
            Path(cfg.CKPT_DIR).mkdir(exist_ok=True)
        model.train()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        return model

    @staticmethod
    def save_ckpt(cfg, model):
        save_path = Path(cfg.CKPT_DIR) / cfg.TAPER_CKPT
        torch.save(model.state_dict(), save_path)
