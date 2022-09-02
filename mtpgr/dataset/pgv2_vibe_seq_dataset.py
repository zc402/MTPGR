from pathlib import Path
import pickle
import json
import numpy as np
from torch.utils.data import Dataset


class PGv2VIBESeqDataset(Dataset):
    """
    Load VIBE results (SMPL-X parameter sequence).
    Return continuous SMPL-X params and corresponding gesture labels of shape: (batch, class - 8*4 + 1)
    To load frame 30 ~ 45, use "vibe_seq = vibe_seq_dataset[30: 45]"
    Must keep order, don't shuffle.
    If dense_indices are provided, only selected params are returned
    """

    def __init__(self, vibe_path: Path,
                 gesture_label_path: Path,
                 ori_label_path: Path,
                 combine_label_path: Path,
                 part_filter: list=None,
                 use_cam_pose: bool=False):
        """
        Args:
            vibe_path (Path): Path of the vibe output (SMPL-X parameters)
            gesture_label_path (Path): Path of the 8 + 1 gesture labels (json file)
            ori_label_path (Path): Path of the 4 orientation labels
            combine_label_path (Path): Path of the 32 + 1 combined labels. 
            part_filter (list): Indices of interested parts. e.g. [0, 3, 4, 6,...]. Set None to use all parts from SMPL-X.
            use_cam_pose (bool): Concat camera pose to the last (TODO: First?) of V dim in tensor_VC
        
        """
        with vibe_path.open('rb') as f:
            self.vibe = pickle.load(f)
        with gesture_label_path.open('r') as f:
            self.ges_label = json.load(f)  # [0,0,0,1,1,1,0,0,0,...]
        with ori_label_path.open('r') as f:
            self.ori_label = json.load(f)
        self.ori_label = [ord(c) for c in self.ori_label]  # Convert F/L/B/R into ASCII because Pytorch dataloader do not accept char type
        with combine_label_path.open('r') as f:
            self.combine_label_path = json.load(f)
        
        self.part_filter = part_filter
        self.use_cam_pose = use_cam_pose
        self.file_name = vibe_path.stem

    def __len__(self):
        return len(self.vibe)

    def __getitem__(self, index):
        vibe_output = self.vibe[index]  # vibe output for 1 frame
        vibe_output = vibe_output.get(1)  # person 1 is the police, manually defined in preprocessing code
        vibe_output = {k:v[0] for k,v in vibe_output.items() if v is not None}  # Squeeze the frame dim. The original VIBE uses this dimision for frame_num. RT_VIBE keeps the structure, but leave this dim empty.
        """Note that the length of 'vibe result' is shorter than 'label' due to untracked frames (occlusion etc.).
        Therefore, the 'frame' key in vibe result is used to skip these untracked frames."""
        video_frame_num = vibe_output['frame_ids']  # Corresponding video frame. video_frame_num is 0-based
        gesture = self.ges_label[video_frame_num]
        orientation = self.ori_label[video_frame_num]
        combine = self.combine_label_path[video_frame_num]

        """vibe_output:
        {"pred_cam": ndarray - shape(3), "orig_cam": shape(4), "pose": shape(72), "betas": shape(10),
        "joints3d": shape(49, 3), "joints2d_img_coord": shape(49, 2), "bbox": shape(4), frame_ids: shape(,)}
        """
        keypoints = vibe_output['joints3d']
        pose = vibe_output['pose'].reshape((-1, 3))

        if self.use_cam_pose:
            cam = vibe_output['pred_cam']
            cam = cam.reshape((1, 3))
            keypoints = np.concatenate((keypoints, cam))

        if self.part_filter:
            keypoints = keypoints[self.part_filter]

        return {
            "kp": keypoints,  # Shape: (V - num_keypoints, C - 3D_coord)
            "ges": gesture,  # Shape: (,)
            "ori": orientation,  # Shape: (,)
            "combine": combine,  # Shape: (,)
            "pose": pose,  # Shape: (J - num_SMPL_joints, C - 3D_coord)
        }
    

    # def _get_vibe_pose_params(self, vibe_output):
    #     """
    #     Convert vibe_params to STGCN input features of shape C,V.
    #     STGCN requires input features of shape N,C,T,V. (N:batch, C: num_features. T: num_frames. V: num_keypoints)
    #     """
    #     pose = vibe_output['pose']  # 72 pose params,
    #     pose_VC = pose.reshape((-1, 3))  # (num_keypoints, rotation_3d)
    #     return pose_VC

    def get_name(self) -> str:
        return self.file_name

    @classmethod
    def from_config(cls, cfg):
        from mtpgr.kinematic import SparseToDense

        s2d = SparseToDense.from_config(cfg)
        part_filter = s2d.get_s2d_indices()  # Sparse indices of each joint. Used to extract dense coordinates.

        def new_initializer(video_name: str):
            dataset_path: Path = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR
            vibe_seq_path = dataset_path / cfg.GENDATA.VIBE_DIR / (video_name + ".pkl")
            gesture_label_path = dataset_path / cfg.GENDATA.GES_LABEL_DIR / (video_name + ".json")
            ori_label_path = dataset_path / cfg.GENDATA.ORI_LABEL_DIR / (video_name + ".json")
            combine_label_path = dataset_path / cfg.GENDATA.COMBINE_LABEL_DIR / (video_name + ".json")
            return PGv2VIBESeqDataset(
                vibe_seq_path,
                gesture_label_path,
                ori_label_path,
                combine_label_path,
                part_filter=part_filter,
                use_cam_pose=cfg.MODEL.USE_CAMERA_POSE
            )

        return new_initializer
