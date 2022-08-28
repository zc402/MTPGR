from pathlib import Path
import pickle
import json
import numpy as np
from torch.utils.data import Dataset


class SingleVideo(Dataset):
    """
    Load vibe results (SMPL parameters for each frame) by video name. (mp4 video not required)
    Return continuous vibe params and corresponding gesture labels of shape: 8*4 + 1
    Should not be shuffled
    If dense_indices are provided, only selected params are returned
    """

    def __init__(self, vibe_path: Path,
                 gesture_label_path: Path,
                 part_filter: list,
                 use_cam_pose: bool):
        """
        Args:
            vibe_path: Path of the vibe output (SMPL-X parameters)
            gesture_label_path: Path of the labels (json file)
            part_filter (list): Indices of interested parts. e.g. [0, 3, 4, 6,...]. Set None to use all parts from SMPL-X.
            use_cam_pose (bool): Concat camera pose to the last of V dim in tensor_VC
        """
        with vibe_path.open('rb') as f:
            self.vibe = pickle.load(f)
        with gesture_label_path.open('r') as f:
            self.label = json.load(f)  # [0,0,0,1,1,1,0,0,0,...]
        
        # Note that the length of 'vibe result' is shorter than 'label' due to untracked frames (occlusion etc.).
        # Therefore, the 'frame' key in vibe result is used to skip these untracked frames.
        self.part_filter = part_filter  # SMPL-X part filter
        self.use_cam_pose = use_cam_pose

    def __len__(self):
        return len(self.vibe)

    def __getitem__(self, index):
        vibe_params = self.vibe[index]  # vibe params for 1 frame
        vibe_params = vibe_params.get(1)  # person "1"
        frame_num = vibe_params['frame_ids'][0]  # frame_num is 0-based
        gesture = self.label[frame_num]

        tensor_VC = self._extract_pose_params(vibe_params)

        if self.use_cam_pose:
            # The cam pose should be concat to index 0 of all features, according to JOINT_MAP dense indices
            cam = vibe_params['pred_cam']
            cam = cam.reshape((-1, 3))
            tensor_VC = np.concatenate((cam, tensor_VC))

        if self.part_filter:
            tensor_VC = tensor_VC[self.part_filter]

        return {'tensor_vc': tensor_VC,  # the batch_size in this dataset is the num_frames, or 'T'
                'label': gesture,  # a scalar
                }

    def _extract_pose_params(self, vibe_params):
        """
        Convert vibe_params to STGCN input features of shape C,V.
        STGCN requires input features of shape N,C,T,V. (N:batch, C: num_features. T: num_frames. V: num_keypoints)
        :param vibe_params:
        :return:
        """
        pose = vibe_params['pose']  # 72 pose params,
        pose_VC = pose.reshape((-1, 3))  # (num_keypoints, rotation_3d)
        # pose_VC_2 = pose_VC[part_indices, :]  # Only take useful parts, do not send unused parts into GCN.
        return pose_VC

    @classmethod
    def from_config(cls, cfg):
        from mtpgr.kinematic import SparseToDense

        s2d = SparseToDense.from_config(cfg)
        filter = s2d.get_s2d_indices()  # Sparse indices of each joint. Used to extract dense coordinates.
        use_cam = cfg.MODEL.USE_CAMERA_POSE

        def new_initializer(vibe_path, ges_label_path):
            return SingleVideo(
                vibe_path,
                ges_label_path,
                filter,
                use_cam
            )

        return new_initializer
