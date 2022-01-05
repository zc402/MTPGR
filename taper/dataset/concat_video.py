from typing import List, Dict

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
        # BUG of pytorch: concat dataset do not support slice
        clip: List[Dict[str: np.ndarray]] = [self.concat_dataset[x] for x in range(index, index+self.clip_len)]

        tensor_TVC = [d['tensor_vc'] for d in clip]
        tensor_CTV = np.transpose(tensor_TVC, (2, 0, 1))

        label_T = [d['label'] for d in clip]

        return {'tensor_ctv': tensor_CTV,
                'label_t': label_T}  # N dimension is batch_size

