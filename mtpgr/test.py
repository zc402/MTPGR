
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mtpgr.utils.log import log
from mtpgr.dataset.pgv2_dataset import PGv2TestDataset
from mtpgr.config import get_cfg_defaults
from mtpgr.analysis.chalearn_jaccard import ChaLearnJaccard
from mtpgr.analysis.confusion_matrix import compute_cm

# joint xy coords -> gcn -> fcn
from mtpgr.network.predictor import Predictor


class Tester():
    def __init__(self, predictor, num_classes, output_name):
        """
        Args:
            predictor: Includes dataloader and neural network
            output_name: Filename for saving prediction result
        """
        self.predictor = predictor
        self.num_classes = num_classes
        self.output_name = output_name

        self.predictor.model.eval()
        self.predictor.post_step = self.post_step
        # self.predictor.post_epoch = self.post_epoch

        self.pred_stream = []
        self.pred_TC_stream = []
        self.label_stream = []

        self.save_folder = Path('output') / self.output_name
        self.save_folder.parent.mkdir(exist_ok=True)
        self.save_folder.mkdir(exist_ok=True)

        self.result_list = []  # Test results. Shape: (seqs, {"pred", "label"})

    @torch.no_grad()
    def val(self):
        self.predictor.run_epoch()

        result_save_path = self.save_folder / "result.pkl"
        # Save to disk
        with result_save_path.open('wb') as f:
            pickle.dump(self.result_list, f)
        
        self._jaccard(self.result_list, self.num_classes)
        self._confusion_matrix(self.result_list)

    def post_step(self, pred, label):
        """
        In test, 1 step == 1 epoch. N (batch size) = 1.
        Shapes:
            pred: (N*T, C)
            label: (N*T,)
        """
        self.result_list.append({
                "pred": pred.cpu().numpy(), 
                "label": label.cpu().numpy()
            })

    @classmethod
    def _test_set_dataloader(cls, cfg):  # Dataloader for validate
        test_dataset = PGv2TestDataset.from_config(cfg)
        eval_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        return eval_loader

    @classmethod
    def from_config(cls, cfg):
        predictor = Predictor.from_config(cfg, cls._test_set_dataloader(cfg))
        instance = Tester(predictor, cfg.DATASET.NUM_CLASSES, cfg.OUTPUT)
        return instance

    @staticmethod
    def _jaccard(result_list, num_classes):
        # Convert to list([gt][pred])
        pred_T = [np.argmax(seq_res["pred"], axis=-1) for seq_res in result_list]  # Shape: (seqs, T)
        label_T = [seq_res["label"] for seq_res in result_list]
        gt_pred_list = [(gt, pred) for gt, pred in zip(label_T, pred_T)]
        J, Js = ChaLearnJaccard(num_classes).mean_jaccard_index(gt_pred_list)
        log.info(f"Jaccard score is {J}. Scores for each sequence are {Js}")

    def _confusion_matrix(self, result_list):
        pred_T = [np.argmax(seq_res["pred"], axis=-1) for seq_res in result_list]  # Shape: (seqs, T)
        label_T = [seq_res["label"] for seq_res in result_list]
        pred_T = np.concatenate(pred_T)
        label_T = np.concatenate(label_T)
        compute_cm(label_T, pred_T, save_folder=self.save_folder)


if __name__ == '__main__':
    val_cfg = get_cfg_defaults()
    val_cfg.merge_from_file(Path('configs', 'default_model.yaml'))
    Tester.from_config(val_cfg).val()
