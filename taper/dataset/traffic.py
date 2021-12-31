from typing import List, Dict

from torch.utils.data import Dataset, ConcatDataset, DataLoader
from pathlib import Path
from taper.dataset.path import to_vibe_params, to_gesture_label, train_videos
import json5
import pickle
from taper.kinematic.sparse_to_dense import part_indices
import numpy as np
from tqdm import tqdm

class SingleVideo(Dataset):
    """
    Dataset for single video, get clips.
    Continuous vibe params and corresponding gesture labels of shape: 8*4 + 1
    'shuffle' in dataloader should never be used.
    """
    def __init__(self, video: Path):
        vibe_path = to_vibe_params(video)
        with vibe_path.open('rb') as f:
            self.vibe = pickle.load(f)
        gesture_label_path = to_gesture_label(video)
        with gesture_label_path.open('r') as f:
            self.gesture = json5.load(f)
        # Note that 'vibe' is shorter than 'gesture' due to failed tracks caused by image occlusion.
        # Therefore, the 'frame' in 'vibe' is used as index for 'gesture'

    def __len__(self):
        return len(self.vibe)

    def __getitem__(self, index):
        vibe_params = self.vibe[index]  # vibe params for 1 frame
        vibe_params = vibe_params.get(1)  # person "1"
        frame_num = vibe_params['frame_ids'][0]  # frame_num is 0-based
        gesture = self.gesture[frame_num]

        tensor_VC = self._to_gcn_feature(vibe_params)
        return {'tensor_vc': tensor_VC,  # the batch_size in this dataset is the num_frames, or 'T'
                'label': gesture,  # a scalar
                }

    def _to_gcn_feature(self, vibe_params):
        """
        Convert vibe_params to STGCN input features of shape C,V.
        STGCN requires input features of shape N,C,T,V. (N:batch, C: num_features. T: num_frames. V: num_keypoints)
        :param vibe_params:
        :return:
        """
        pose = vibe_params['pose']  # pose params of shape 72,
        pose_VC = pose.reshape((-1, 3))  # (num_keypoints, rotation_3d)
        pose_VC_2 = pose_VC[part_indices, :]  # Only take useful parts, do not send unused parts into GCN.
        return pose_VC_2


class VideoConcat(ConcatDataset):
    """
    Concatenate dataset, return continuous frames. (batchsize = 10 means 10 continuous frames)
    """
    def __init__(self):
        video_datasets = [SingleVideo(v) for v in tqdm(train_videos)]
        super().__init__(datasets=video_datasets)


class TrafficGesClips(Dataset):
    """
    Return Clips from concatenated gesture features.
    return GCN input feature format: N,C,T,V
    """
    def __init__(self, clip_len: int):
        self.clip_len = clip_len
        self.frames = VideoConcat()

    def __len__(self):
        return len(self.frames) - self.clip_len

    def __getitem__(self, index):
        # BUG of pytorch: concat dataset do not support slice
        clip: List[Dict[str: np.ndarray]] = [self.frames[x] for x in range(index, index+self.clip_len)]


        tensor_TVC = [d['tensor_vc'] for d in clip]
        tensor_CTV = np.transpose(tensor_TVC, (2, 0, 1))

        label_T = [d['label'] for d in clip]

        return {'tensor_ctv': tensor_CTV,
                'label_t': label_T}  # batch_size is n

# ds = TrafficGesClips(15)
# dl = DataLoader(ds, batch_size=10, shuffle=True)
# a = next(iter(dl))
# pass