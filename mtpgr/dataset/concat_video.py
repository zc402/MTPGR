from typing import List, Dict
import logging
from torch.utils.data import Dataset, ConcatDataset
import numpy as np


class ConcatVideo(Dataset):
    """
    Return Clips from concatenated gesture features in GCN input format: N,C,T,V
    Should enable shuffle on training
    """
    def __init__(self, datasets, clip_len: int):
        self.clip_len = clip_len
        # concat_dataset: batch_size is the length of the clip.
        self.concat_dataset = ConcatDataset(datasets)
        self.num_frames = len(self.concat_dataset)

    def __len__(self):

        return self.num_frames - self.clip_len

    def __getitem__(self, index):
        # Extract a clip from concatenated videos
        # Concat dataset do not support slice
        clip: List[Dict[str: np.ndarray]] = [self.concat_dataset[x] for x in range(index, index+self.clip_len)]

        tensor_TVC = [d['tensor_vc'] for d in clip]
        tensor_CTV = np.transpose(tensor_TVC, (2, 0, 1))

        label_T = [d['label'] for d in clip]
        label_T = np.array(label_T)

        return {'tensor_ctv': tensor_CTV,
                'label_t': label_T}  # N dimension is batch_size

    @classmethod
    def from_config(cls, cfg):
        from pathlib import Path
        from .single_video import SingleVideo

        if cfg.DATASET.MODE == "TRAIN":
            names = cfg.DATASET.TRAIN_VIDEOS
            logging.info("Load dataset for TRAINING")
            
        elif cfg.DATASET.MODE == "TEST":
            names = cfg.DATASET.TEST_VIDEOS
            logging.info("Load dataset for TESTING")
        else:
            raise NotImplementedError()
        # Construct paths
        vibe_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.VIBE_DIR
        vibe_list = [vibe_folder / (name + '.pkl') for name in names]
        label_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.LABEL_DIR
        label_list = [label_folder / (name + '.json5') for name in names]

        video_dataset_list = [SingleVideo.from_config(cfg)(vibe, label)
                              for vibe, label in zip(vibe_list, label_list)]
        instance = ConcatVideo(video_dataset_list, cfg.MODEL.CLIP_LEN)
        return instance
