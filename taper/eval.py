from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn import CrossEntropyLoss

from torch import optim
from taper.dataset import SingleVideo, ConcatVideo
from taper.kinematic import dense_indices
from taper.config import get_cfg_defaults
from taper.network import TAPER

# joint xy coords -> gcn -> fcn
class Evaluate():
    def __init__(self):
        self.cfg = get_cfg_defaults()
        self.data_loader = self.eval_data_loader(self.cfg)
        self.model = self.eval_model(self.cfg)

    @torch.no_grad()
    def eval(self):
        device = torch.device(self.cfg.MODEL.DEVICE)
        out_stream = []
        label_stream = []
        for num_frame, train_data in enumerate(self.data_loader):
            tensor_NCTV = train_data['tensor_ctv'].to(device)  # Batch is N dim
            label_NT = train_data['label_t'].to(device)  # N,T

            out_TC = self.model(tensor_NCTV)  # Out: N*T,C
            out_T = torch.argmax(out_TC, dim=1)  # N*T
            out_T = out_T.cpu()
            label_T = label_NT.reshape([-1])  # N*T
            label_T = label_T.cpu()

            # To simulate the sliding window, only the last one result counts. except the beginning.
            if len(out_stream) == 0:
                out_stream.extend(out_T.tolist())
            else:
                out_stream.append(out_T.tolist()[-1])

            if len(label_stream) == 0:
                label_stream.extend(label_T.tolist())
            else:
                label_stream.append(label_T.tolist()[-1])

            if num_frame % 1000 == 0:
                correct = np.array(out_stream) == np.array(label_stream)
                acc = correct.astype(np.float32).mean()
                print("Frame: {}, Accuracy: {:.2f}".format(num_frame+self.cfg.EVAL.CLIP_LEN, acc))
            # if step % 100 == 0:
            #     print("Step: %d, Loss: %f" % (step, loss_tensor.item()))
            #     acc = self.acc(class_out, label_NT)
            #     print("Accuracy: {:.2f}".format(acc))
            #
            # if step % 2000 == 0:
            #     self.save_ckpt(self.cfg, self.model)
            #     print('Model save')
            # step = step + 1

    @staticmethod
    def eval_data_loader(cfg):
        names = cfg.EVAL.SET
        vibe_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.VIBE_DIR
        vibe_list = [vibe_folder / (name + '.pkl') for name in names]
        label_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.LABEL_DIR
        label_list = [label_folder / (name + '.json5') for name in names]

        video_dataset_list = [SingleVideo(v, l, dense_indices) for v, l in zip(vibe_list, label_list)]
        concat_dataset = ConcatVideo(video_dataset_list, cfg.EVAL.CLIP_LEN)
        eval_loader = DataLoader(concat_dataset, batch_size=1, shuffle=False, drop_last=True)
        return eval_loader

    @staticmethod
    def eval_model(cfg):
        model = TAPER()
        ckpt_path = Path(cfg.DATA_ROOT) / cfg.MODEL.CKPT_DIR / cfg.MODEL.TAPER_CKPT
        if ckpt_path.is_file():
            print("Load ckeckpoint")
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt)
        else:
            raise FileNotFoundError(f'No checkpoint in {ckpt_path.absolute()}')
        model.eval()
        device = torch.device(cfg.MODEL.DEVICE)
        model.to(device)
        return model

    @staticmethod
    def acc(input, target):
        class_N = torch.argmax(input, dim=1)
        acc = (class_N == target).float().mean().item()
        return acc

if __name__ == '__main__':
    Evaluate().eval()