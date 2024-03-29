
import argparse
import logging
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mtpgr.config.defaults import get_auto_name

from mtpgr.utils.log import log
from mtpgr.dataset.pgv2_dataset import PGv2TestDataset
from mtpgr.config.defaults import get_cfg_defaults
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
        self.predictor: Predictor = predictor
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
    def test(self):
        self.predictor.run_epoch()

        result_save_path = self.save_folder / "result.pkl"
        # Save to disk
        with result_save_path.open('wb') as f:
            pickle.dump(self.result_list, f)
        
        j_score = self._jaccard(self.result_list, self.num_classes)
        self._confusion_matrix(self.result_list)
        return j_score

    def post_step(self, pred, label, **kwargs):
        """
        In test, 1 step == 1 epoch. N (batch size) = 1.
        Shapes:
            pred: (N*T, C)
            label: (N*T,)
        """
        self.result_list.append({
                "pred": pred.cpu().numpy(), 
                "label": label.cpu().numpy(),
                "batch_data": kwargs["batch_data"]
            })

    @classmethod
    def _test_set_dataloader(cls, cfg):  # Dataloader for validate
        test_dataset = PGv2TestDataset.from_config(cfg)
        eval_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        return eval_loader

    @classmethod
    def from_config(cls, cfg):
        if cfg.MODEL.NAME == 'auto':
            model_name = get_auto_name(cfg)
        else:
            model_name = cfg.MODEL.NAME
        log.info(f"--- Testing: {model_name} ---")
        predictor = Predictor.from_config(cfg, cls._test_set_dataloader(cfg))
        instance = Tester(predictor, cfg.DATASET.NUM_CLASSES, model_name)
        return instance

    def _jaccard(self, result_list, num_classes):
        # Convert to list([gt][pred])
        pred_T = [np.argmax(seq_res["pred"], axis=-1) for seq_res in result_list]  # Shape: (seqs, T)
        label_T = [seq_res["label"] for seq_res in result_list]
        gt_pred_list = [(gt, pred) for gt, pred in zip(label_T, pred_T)]
        J, Js, j_mat = ChaLearnJaccard(num_classes).mean_jaccard_index(gt_pred_list)
        log.info(f"Jaccard score is {J}. Scores for each sequence are {Js}")
        np.savetxt(self.save_folder / "jaccard_matrix.txt", j_mat, fmt='%-.2f')
        np.savetxt(self.save_folder / "jaccard_each_seq.txt", Js, fmt='%-.2f')
        np.savetxt(self.save_folder / "jaccard_avg.txt", J[np.newaxis], fmt='%-.2f')
        return J

    def _confusion_matrix(self, result_list):
        pred_T = [np.argmax(seq_res["pred"], axis=-1) for seq_res in result_list]  # Shape: (seqs, T)
        label_T = [seq_res["label"] for seq_res in result_list]
        pred_T = np.concatenate(pred_T)
        label_T = np.concatenate(label_T)
        compute_cm(label_T, pred_T, save_folder=self.save_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tester for monocular traffic police gesture recognizer')
    parser.add_argument('-c', '--config', type=str, default="default_model.yaml", help='Enter the file name of a configuration from configs folder')
    args = parser.parse_args()
    config_path:Path = Path('configs', args.config)
    if not config_path.is_file():
        log.error(f"No such config file: {config_path}")

    val_cfg = get_cfg_defaults()
    val_cfg.merge_from_file(config_path)
    log.setLevel(logging.INFO)
    Tester.from_config(val_cfg).test()
