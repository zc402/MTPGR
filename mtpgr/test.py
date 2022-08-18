import logging
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mtpgr.dataset import SingleVideo, ConcatVideo
from mtpgr.config import get_cfg_defaults

# joint xy coords -> gcn -> fcn
from mtpgr.network.predictor import Predictor


class Tester():
    def __init__(self, predictor, output_name):
        """
        Args:
            predictor: Includes dataloader and neural network
            output_name: Filename for saving prediction result
        """
        self.predictor = predictor
        self.output_name = output_name
        self.predictor.model.eval()
        self.predictor.post_step = self.post_step

        self.pred_stream = []
        self.pred_TC_stream = []
        self.label_stream = []

        self.step = 1

    @torch.no_grad()
    def val(self):
        self.predictor.run_epoch()
        self.post_epoch()

        # self.sliding_add(self.cfg, pred_stream, label_stream, num_frame, pred_T, label_T)
        # self.post_prediction(self.cfg, pred_stream, label_stream)

    def post_step(
        self, 
        class_TC,  # predicted score (float), shape:(time, confidence_score)
        label_T,  # true class (int)
        ):

        class_TC = class_TC.cpu()
        class_T = torch.argmax(class_TC, dim=-1)
        class_T = class_T.view([-1])
        label_T = label_T.cpu()
        label_T = label_T.view([-1])

        # To simulate the sliding window, only the last one result counts. except the beginning.
        if len(self.pred_stream) == 0:  # At the beginning, append (temporal_length) frames
            self.pred_stream.extend(class_T.tolist())
            self.pred_TC_stream.extend(class_TC.tolist())
            self.label_stream.extend(label_T.tolist())
        else:  # At the middle, only append last frame
            self.pred_stream.append(class_T.tolist()[-1])  # Temporally last result in GCN
            self.pred_TC_stream.append(class_TC.tolist()[-1])
            self.label_stream.append(label_T.tolist()[-1])
            
            

        plot = False  # Graph of predictions and labels

        if len(self.pred_stream) % 5000 == 0:
            correct = np.array(self.pred_stream) == np.array(self.label_stream)
            acc = correct.astype(np.float32).mean()
            print("Frame: {}, Accuracy: {:.2f}".format(len(self.pred_stream), acc))
            if plot:
                plt.plot(self.label_stream)
                plt.plot(self.pred_stream)
                plt.show()

        self.step = self.step + 1

    def post_epoch(self):
        # End of validation, dump results
        assert len(self.pred_stream) == len(self.label_stream)
        correct = np.array(self.pred_stream) == np.array(self.label_stream)
        acc = correct.astype(np.float32).mean()
        print("Total Frame: {}, Accuracy: {:.2f}".format(len(self.pred_stream), acc))

        save_path = Path('output') / self.output_name
        save_path.parent.mkdir(exist_ok=True)

        with save_path.open('wb') as f:
            pickle.dump({
                'pred_T':self.pred_stream,
                'pred_TC':self.pred_TC_stream,
                'label_T': self.label_stream
                }, f)

    @classmethod
    def _data_loader(cls, cfg):  # Dataloader for validate
        concat_dataset = ConcatVideo.from_config(cfg)
        eval_loader = DataLoader(concat_dataset, batch_size=1, shuffle=False, drop_last=False)
        return eval_loader

    @classmethod
    def from_config(cls, cfg):
        predictor = Predictor.from_config(cfg, cls._data_loader(cfg))
        instance = Tester(predictor, cfg.OUTPUT)
        return instance


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    val_cfg = get_cfg_defaults()
    val_cfg.merge_from_file(Path('configs', 'default_model.yaml'))
    val_cfg.DATASET.MODE = "TEST"
    Tester.from_config(val_cfg).val()
