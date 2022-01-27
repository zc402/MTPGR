import pickle
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from taper.dataset import SingleVideo, ConcatVideo
from taper.kinematic import dense_indices
from taper.config import get_cfg_defaults
from taper.network import TAPER

# joint xy coords -> gcn -> fcn
class Val():
    def __init__(self):
        self.cfg = get_cfg_defaults()
        self.data_loader = self.eval_data_loader(self.cfg)
        self.model = self.eval_model(self.cfg)

    @torch.no_grad()
    def eval(self):
        device = torch.device(self.cfg.MODEL.DEVICE)
        pred_stream = []  # List of predicted classes: [0,0,0,1,1,1,0,0,2,2,...]
        label_stream = []  # List of ground truth classes
        for num_frame, train_data in enumerate(tqdm(self.data_loader)):
            tensor_NCTV = train_data['tensor_ctv'].to(device)  # Batch is N dim
            label_NT = train_data['label_t'].to(device)  # N,T

            pred_TC = self.model(tensor_NCTV)  # Out: N*T,C
            pred_T = torch.argmax(pred_TC, dim=1)  # N*T. In eval mode, N=1
            label_T = label_NT.reshape([-1])  # N*T

            self.sliding_add(self.cfg, pred_stream, label_stream, num_frame, pred_T, label_T)
        self.post_prediction(self.cfg, pred_stream, label_stream)

    @staticmethod
    def eval_data_loader(cfg):
        names = cfg.VAL.SET
        vibe_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.VIBE_DIR
        vibe_list = [vibe_folder / (name + '.pkl') for name in names]
        label_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.LABEL_DIR
        label_list = [label_folder / (name + '.json5') for name in names]

        video_dataset_list = [SingleVideo(v, l, dense_indices, cfg.MODEL.USE_CAM_POSE) for v, l in zip(vibe_list, label_list)]
        concat_dataset = ConcatVideo(video_dataset_list, cfg.VAL.CLIP_LEN)
        eval_loader = DataLoader(concat_dataset, batch_size=1, shuffle=False, drop_last=True)
        return eval_loader

    @staticmethod
    def val_model(cfg):
        model = TAPER()
        ckpt_path = Path(cfg.DATA_ROOT) / cfg.MODEL.CKPT_DIR / cfg.MODEL.TAPER_CKPT
        if ckpt_path.is_file():
            print("Load ckeckpoint")
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt)
        else:
            raise FileNotFoundError(f'No checkpoint in {ckpt_path.absolute()}')
        model.val()
        device = torch.device(cfg.MODEL.DEVICE)
        model.to(device)
        return model

    @staticmethod
    def sliding_add(cfg,
                    pred_stream,
                    label_stream,
                    num_frame,
                    pred_T,
                    label_T):

        pred_T = pred_T.cpu()
        label_T = label_T.cpu()

        # To simulate the sliding window, only the last one result counts. except the beginning.
        if len(pred_stream) == 0:
            pred_stream.extend(pred_T.tolist())
        else:
            pred_stream.append(pred_T.tolist()[-1])

        if len(label_stream) == 0:
            label_stream.extend(label_T.tolist())
        else:
            label_stream.append(label_T.tolist()[-1])

        plot = False  # Graph of predictions and labels
        if not plot:
            return
        if (num_frame + cfg.VAL.CLIP_LEN) % 10000 == 0:
            correct = np.array(pred_stream) == np.array(label_stream)
            acc = correct.astype(np.float32).mean()
            print("Frame: {}, Accuracy: {:.2f}".format(num_frame + cfg.VAL.CLIP_LEN, acc))
            plt.plot(label_stream)
            plt.plot(pred_stream)
            plt.show()

    @staticmethod
    def post_prediction(cfg, pred_stream, label_stream,):
        assert len(pred_stream) == len(label_stream)
        correct = np.array(pred_stream) == np.array(label_stream)
        acc = correct.astype(np.float32).mean()
        print("Frame: {}, Accuracy: {:.2f}".format(len(pred_stream) + cfg.VAL.CLIP_LEN, acc))

        save_path = Path('output') / 'j14_nocam_cls8' / 'result.pkl'
        save_path.parent.mkdir(exist_ok=True)

        with save_path.open('wb') as f:
            pickle.dump((pred_stream, label_stream), f)


if __name__ == '__main__':
    Val().eval()