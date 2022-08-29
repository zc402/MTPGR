from typing import List, Dict
import math
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

from mtpgr.config.defaults import get_cfg_defaults
from mtpgr.dataset.pgv2_vibe_seq_dataset import PGv2VIBESeqDataset

class PGv2TrainDataset(Dataset):
    """
    Video or VIBE output sequences in PGv2 dataset are too long (~10 min) to be used in a single batch.
    Therefore when training, these sequences are concatenated and sliced into shorter sequences. 
    Return:
        kp, ges, ori, combine
    """
    def __init__(
            self, 
            seq_datasets: List[PGv2VIBESeqDataset], 
            truncate_len: int
            ):

        self.concat_dataset = torch.utils.data.ConcatDataset(seq_datasets)
        self.truncate_len = truncate_len
        self.seq_datasets = seq_datasets
        self.num_total_frames = len(self.concat_dataset)
        self.truncate_start_frame = 0
        # Number of truncated sequence. If the concat video ends before last sequence is finished then the last seq is filled from the beginning
        self.num_sequences = math.ceil(self.num_total_frames / self.truncate_len)  

    def __len__(self):
        """Total number of truncated short sequences"""
        return self.num_sequences
    
    def __getitem__(self, index) -> Dict:
        """Return a dict of tensors and labels
        args:
            index: The number of truncated sequence
        """
        start_frame = self.truncate_start_frame + (index * self.truncate_len)
        end_frame = start_frame + self.truncate_len  ## Right boundary, do not include self - [start, end)

        # truncated_seq: List[Dict[str: np.ndarray]] = [self.concat_dataset[x] for x in range(index, index+self.clip_len)]
        # Is this concat dataset bug still there?
        
        # Shape: [0:truncate_len - frames] [str - labels] [each label...]
        truncated_seq: np.ndarray = self._cycle(self.concat_dataset, start_frame, end_frame)
        assert truncated_seq.shape[0] == self.truncate_len

        truncated_seq_dict = {}  # {"kp": np.ndarray, "ges": np.ndarray, ...}
        result_dict_keys:List[str] = truncated_seq[0].keys()
        for key_name in result_dict_keys:
            truncated_seq_dict[key_name] = np.stack([truncated_seq[i][key_name] for i in range(self.truncate_len)])

        return truncated_seq_dict
    
    def _cycle(self, seq, ia, ib):
        """Read the list from beginning if the index is out of boundary.
        Slice (e.g. dataset[1:3]) not available. Pytorch concat dataset raises a error when visited with slice."""
        seq_len = len(seq)
        new_seq = [seq[i % seq_len] for i in range(ia, ib)]
        return np.array(new_seq)
    

    @classmethod
    def from_config(cls, cfg):
        video_names: List[str] = cfg.DATASET.TRAIN_VIDEOS  # Videos from training set
        vibe_datasets = [PGv2VIBESeqDataset.from_config(cfg)(video_name) for video_name in video_names]
        return PGv2TrainDataset(vibe_datasets, cfg.MODEL.CLIP_LEN) 

# PGv2TrainDataset.from_config(get_cfg_defaults())[1]

class PGv2TestDataset(Dataset):
    """
    The test dataset yeilds a full sequence (~10min) when called. Do not support batch_size > 1
    """
    def __init__(
        self, 
        seq_datasets: List[PGv2VIBESeqDataset], 
        ):
        self.seq_datasets = seq_datasets

    def __len__(self):
        return len(self.seq_datasets)

    def __getitem__(self, index):
        seq: np.ndarray = np.array([self.seq_datasets[index][frame] for frame in range(len(self.seq_datasets[index]))])

        seq_dict = {}  # {"kp": np.ndarray, "ges": np.ndarray, ...}
        result_dict_keys:List[str] = seq[0].keys()
        for key_name in result_dict_keys:
            seq_dict[key_name] = np.stack([seq[i][key_name] for i in range(seq.shape[0])])

        return seq_dict

        # kp = np.stack([seq[i, 0] for i in range(seq.shape[0])])
        # ges = np.stack([seq[i, 1] for i in range(seq.shape[0])])
        # ori = np.stack([seq[i, 2] for i in range(seq.shape[0])])
        # combine = np.stack([seq[i, 3] for i in range(seq.shape[0])])
        # return kp, ges, ori, combine
    
    @classmethod
    def from_config(cls, cfg):
        video_names: List[str] = cfg.DATASET.TEST_VIDEOS  # Videos from test set
        vibe_datasets = [PGv2VIBESeqDataset.from_config(cfg)(video_name) for video_name in video_names]
        return PGv2TestDataset(vibe_datasets) 

PGv2TestDataset.from_config(get_cfg_defaults())[1]
