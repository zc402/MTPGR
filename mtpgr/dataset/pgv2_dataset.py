from typing import List, Dict
import logging
import math
import torch.utils.data
from torch.utils.data import Dataset
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
        self.num_sequences
    
    def __getitem__(self, index) -> Dict:
        """Return a dict of tensors and labels
        args:
            index: The number of truncated sequence
        """
        start_frame = self.truncate_start_frame + (index * self.truncate_len)
        end_frame = start_frame + self.truncate_len  ## Right boundary, do not include self - [start, end)

        # truncated_seq: List[Dict[str: np.ndarray]] = [self.concat_dataset[x] for x in range(index, index+self.clip_len)]
        # Is this concat dataset bug still there?
        
        # Shape: [0:truncate_len - frames] [0:4 - labels] [each label...]
        truncated_seq:np.ndarray = self._cycle(self.concat_dataset, start_frame, end_frame)

        # kp, ges, ori, combine = [truncated_seq[:, i] for i in range(4)]
        kp = np.stack([truncated_seq[i, 0] for i in range(truncated_seq.shape[0])])
        ges = np.stack([truncated_seq[i, 1] for i in range(truncated_seq.shape[0])])
        ori = np.stack([truncated_seq[i, 2] for i in range(truncated_seq.shape[0])])
        combine = np.stack([truncated_seq[i, 3] for i in range(truncated_seq.shape[0])])
        return kp, ges, ori, combine
    
    def _cycle(self, seq, ia, ib):
        """Read the list from beginning if the index is out of boundary.
        Slice (e.g. dataset[1:3]) not available. Pytorch concat dataset raises a error when visited with slice."""
        seq_len = len(seq)
        new_seq = [seq[i % seq_len] for i in range(ia, ib)]
        return np.array(new_seq)
    

    @classmethod
    def from_config(cls, cfg):
        video_names: List[str] = cfg.DATASET.TRAIN_VIDEOS  # Videos for training purpose
        vibe_datasets = [PGv2VIBESeqDataset.from_config(cfg)(video_name) for video_name in video_names]
        return PGv2TrainDataset(vibe_datasets, cfg.MODEL.CLIP_LEN) 

PGv2TrainDataset.from_config(get_cfg_defaults())[1]

# class PGv2TestDataset(Dataset):
#     """
#     The test dataset yeilds a full sequence (~10min) when called. Do not support batch_size > 1
#     """
#     def __init__(
#         self, 
#         seq_datasets: List[PGv2VIBESeqDataset], 
#         truncate_len: int
#         ):

# class ConcatDataset(Dataset):
#     """
#     Return Clips from concatenated gesture features in GCN input format: N,C,T,V
#     Should enable shuffle on training
#     """
#     def __init__(self, datasets, clip_len: int):
#         self.clip_len = clip_len
#         # concat_dataset: batch_size is the length of the clip.
#         self.concat_dataset = torch.utils.data.ConcatDataset(datasets)
#         self.num_frames = len(self.concat_dataset)

#     def __len__(self):

#         return self.num_frames - self.clip_len

#     def __getitem__(self, index):
#         # Extract a clip from concatenated videos
#         # Concat dataset do not support slice
#         clip: List[Dict[str: np.ndarray]] = [self.concat_dataset[x] for x in range(index, index+self.clip_len)]

#         tensor_TVC = [d['tensor_vc'] for d in clip]
#         tensor_CTV = np.transpose(tensor_TVC, (2, 0, 1))

#         label_T = [d['label'] for d in clip]
#         label_T = np.array(label_T)

#         return {'tensor_ctv': tensor_CTV,
#                 'label_t': label_T}  # N dimension is batch_size

#     @classmethod
#     def from_config(cls, cfg, mode="train"):

#         if mode == "train":
#             names = cfg.DATASET.TRAIN_VIDEOS
#             print("Loading dataset for TRAINING")
#         elif mode == "test":
#             names = cfg.DATASET.TEST_VIDEOS
#             print("Loading dataset for TESTING")
#         else:
#             raise NotImplementedError()
#         # Construct paths
#         vibe_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.VIBE_DIR
#         vibe_list = [vibe_folder / (name + '.pkl') for name in names]
#         # Choose ground truth 
#         if cfg.DATASET.GROUND_TRUTH == "33":
#             label_folder_name = cfg.GENDATA.COMBINE_LABEL_DIR
#         elif cfg.DATASET.GROUND_TRUTH == "9":
#             label_folder_name = cfg.GENDATA.GES_LABEL_DIR
#         else:
#             raise NotImplementedError(f"Unsupported ground truth config: {cfg.DATASET.GROUND_TRUTH}")
#         label_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / label_folder_name
#         label_list = [label_folder / (name + '.json') for name in names]

#         video_dataset_list = [SingleVIBESeqDataset.from_config(cfg)(vibe, label)
#                               for vibe, label in zip(vibe_list, label_list)]
#         instance = ConcatDataset(video_dataset_list, cfg.MODEL.CLIP_LEN)
#         return instance
