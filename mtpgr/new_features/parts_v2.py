from typing import List, Tuple
from vibe.models.smpl import JOINT_NAMES
from mtpgr.utils.log import log
import numpy as np
import torch

# Connection: pelvis, spine, spine

# This is the official SMPL joints. The pose (theta) parameters of SMPL model follows this order.
# VIBE joints extends the list, removes some parts (e.g. Head) and breaks the order. Their corresponding relations are in JOINT_MAP
SMPL_skeleton = {
    0: 'Pelvis',     3: 'Spine1',       6: 'Spine2',    9: 'Spine3',    12: 'Neck',     15: 'Head',
    1: 'L_Hip',      4: 'L_Knee',       7: 'L_Ankle',  10: 'L_Foot',
    2: 'R_Hip',      5: 'R_Knee',       8: 'R_Ankle',  11: 'R_Foot',
    13: 'L_Collar',  16: 'L_Shoulder',  18: 'L_Elbow',  20: 'L_Wrist',
    14: 'R_Collar',  17: 'R_Shoulder',  19: 'R_Elbow',  21: 'R_Wrist',
    22: 'L_Hand',
    23: 'R_Hand'
}

VIBE_3D_joints = JOINT_NAMES
""" The order of 3D keypoints produced by VIBE
[
    'OP Nose', 'OP Neck', 'OP RShoulder',
    'OP RElbow', 'OP RWrist', 'OP LShoulder',
    'OP LElbow', 'OP LWrist', 'OP MidHip',
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP REye', 'OP LEye', 'OP REar',
    'OP LEar', 'OP LBigToe', 'OP LSmallToe',
    'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    'Right Ankle', 'Right Knee', 'Right Hip',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Neck (LSP)', 'Top of Head (LSP)',
    'Pelvis (MPII)', 'Thorax (MPII)',
    'Spine (H36M)', 'Jaw (H36M)',
    'Head (H36M)', 'Nose', 'Left Eye',
    'Right Eye', 'Left Ear', 'Right Ear'
]"""

VIBE_J3D_NAME_TO_IDX = {name: idx for idx, name in enumerate(VIBE_3D_joints)}  # {part_name: part_idx}
SMPL_POSE_NAME_TO_IDX = {name: idx for idx, name in SMPL_skeleton.items()}

# Parts that are used by MTPGR model.
J3D_IN_USE = ['OP RAnkle', 'OP LAnkle', 'OP RKnee', 'OP LKnee', 'OP RHip', 'OP LHip', 'OP MidHip', 'Pelvis (MPII)', 'Spine (H36M)', 'Thorax (MPII)',
'OP RWrist', 'OP LWrist', 'OP RElbow', 'OP LElbow', 'OP RShoulder', 'OP LShoulder', 'OP Neck', 'OP Nose']

J3D_HEIGHTS = {
    'OP RAnkle': 0,
    'OP LAnkle': 0,
    'OP RKnee': 1,
    'OP LKnee': 1,
    'OP RHip': 2,
    'OP LHip': 2,
    'OP MidHip': 3, 'Pelvis (MPII)': 4, 'Spine (H36M)': 5, 'Thorax (MPII)': 6,
    'OP RWrist': 7,
    'OP LWrist': 7,
    'OP RElbow': 8,
    'OP LElbow': 8,
    'OP RShoulder': 9,
    'OP LShoulder': 9,
    'OP Neck': 10,
    'OP Nose': 11,
}

J3D_EDGES = [
        ('OP Neck', 'OP RShoulder'),  # 右肩
        ('OP Neck', 'OP LShoulder'),  # 左肩

        ('OP RShoulder', 'OP RElbow'),  # 右大臂
        ('OP LShoulder', 'OP LElbow'),  # 左大臂

        ('OP RElbow', 'OP RWrist'),  # 右小臂
        ('OP LElbow', 'OP LWrist'),  # 左小臂

        ('OP Nose', 'OP Neck'),  # 头
        ('OP Neck', 'Thorax (MPII)'),  # 躯干

        ('Spine (H36M)', 'Thorax (MPII)'),
        ('Pelvis (MPII)', 'Spine (H36M)'),
        ('OP MidHip', 'Pelvis (MPII)'),  # 为了演示SCPS的缺点，加入一些模拟SMPL脊椎点的点

        ('OP MidHip', 'OP RHip'),  # 右跨
        ('OP MidHip', 'OP LHip'),  # 左跨

        ('OP RHip', 'OP RKnee'),  # 大腿
        ('OP LHip', 'OP LKnee'),

        ('OP RKnee', 'OP RAnkle'),  # 小腿
        ('OP LKnee', 'OP LAnkle'),
    ]

# By the order of J3D_EDGES
BONE = ['B RShoulder', 'B LShoulder', 'B RElbow', 'B LElbow', 'B RWrist', 'B LWrist', 
        'B Nose', 'B Neck', 'B Spine (H36M)', 'B Pelvis (MPII)', 'B MidHip',
        'B RHip', 'B LHip', 'B RKnee', 'B LKnee', 'B RAnkle', 'B LAnkle']

BONE_HEIGHT = {
    'B RAnkle': 0,
    'B LAnkle': 0,
    'B RKnee': 1,
    'B LKnee': 1,
    'B RHip': 2,
    'B LHip': 2,
    'B MidHip': 3, 'B Pelvis (MPII)': 4, 'B Spine (H36M)': 5, 'B Thorax (MPII)': 6,
    'B RWrist': 7,
    'B LWrist': 7,
    'B RElbow': 8,
    'B LElbow': 8,
    'B RShoulder': 9,
    'B LShoulder': 9,
    'B Neck': 10,
    'B Nose': 11,
}

BONE_EDGES = [
        ('B Neck', 'B RShoulder'),  # 右肩
        ('B Neck', 'B LShoulder'),  # 左肩

        ('B RShoulder', 'B RElbow'),  # 右大臂
        ('B LShoulder', 'B LElbow'),  # 左大臂

        ('B RElbow', 'B RWrist'),  # 右小臂
        ('B LElbow', 'B LWrist'),  # 左小臂

        ('B Nose', 'B Neck'),  # 头

        ('B Pelvis (MPII)', 'B Spine (H36M)'),
        ('B MidHip', 'B Pelvis (MPII)'),  # 为了演示SCPS的缺点，加入一些模拟SMPL脊椎点的点

        ('B MidHip', 'B RHip'),  # 右跨
        ('B MidHip', 'B LHip'),  # 左跨

        ('B RHip', 'B RKnee'),  # 大腿
        ('B LHip', 'B LKnee'),

        ('B RKnee', 'B RAnkle'),  # 小腿
        ('B LKnee', 'B LAnkle'),
    ]

# Pose: theta, SMPL Joint rotations.
POSE_IN_USE = ['L_Ankle', 'R_Ankle', 
'L_Knee', 'R_Knee', 
'L_Hip', 'R_Hip', 
'L_Shoulder', 'R_Shoulder', 
'L_Elbow', 'R_Elbow', 
'L_Wrist', 'R_Wrist', 
'Neck', 'Head']  # No pelvis, as it is already used in CUSTOM set

POSE_HEIGHT = {
    'L_Ankle': 0, 'R_Ankle': 0,
    'L_Knee': 1, 'R_Knee': 1,
    'L_Hip': 2, 'R_Hip': 2,
    'L_Wrist': 4, 'R_Wrist': 4,
    'L_Elbow': 5, 'R_Elbow': 5,
    'L_Shoulder': 6, 'R_Shoulder':6,
    'Neck': 7, 'Head': 8
}  

# Edges between POSE vertices
POSE_INTERNAL_EDGES = [
    ('L_Ankle', 'L_Knee'), ('R_Ankle', 'R_Knee'),
    ('L_Knee', 'L_Hip'), ('R_Knee', 'R_Hip'),
    ('L_Hip', 'R_Hip'),
    ('L_Hip', 'L_Shoulder'), ('R_Hip', 'R_Shoulder'),
    ('L_Shoulder', 'L_Elbow'), ('R_Shoulder', 'R_Elbow'),
    ('L_Elbow', 'L_Wrist'), ('R_Elbow', 'R_Wrist'),
    ('L_Shoulder', 'R_Shoulder'),
    ('L_Shoulder', 'Neck'), ('R_Shoulder', 'Neck'),
    ('Neck', 'Head'),
]

# Edges between POSE and J3D
POSE_J3D_EDGES = [('L_Ankle', 'OP LAnkle'), ('R_Ankle', 'OP RAnkle'), 
('L_Knee', 'OP LKnee'), ('R_Knee', 'OP RKnee'), 
('L_Hip', 'OP LHip'), ('R_Hip', 'OP RHip'), 
('L_Shoulder', 'OP LShoulder'), ('R_Shoulder', 'OP RShoulder'),
('L_Elbow', 'OP LElbow'), ('R_Elbow', 'OP RElbow'),
('L_Wrist', 'OP LWrist'), ('R_Wrist', 'OP RWrist'),
('Neck', 'OP Neck')]

CUSTOM_IN_USE = ['Pelvis']  # From SMPL POSE

CUSTOM_HEIGHT = {'Pelvis': 100}  # Root rotation, connects to joints.

EDGES_P_C = [(joint, 'Pelvis') for joint in J3D_IN_USE]  # Connect each joints(J3D) to pelvis(theta)
EDGES_P_R = [('OP RAnkle','R_Ankle'), ('OP LAnkle', 'L_Ankle'), ('OP RKnee', 'R_Knee'), ('OP LKnee', 'L_Knee'), ('OP RHip', 'R_Hip'), ('OP LHip', 'L_Hip'),
('OP RWrist','R_Wrist'), ('OP LWrist','L_Wrist'), ('OP RElbow', 'R_Elbow'), ('OP LElbow', 'L_Elbow'),
('OP RShoulder', 'R_Shoulder'), ('OP LShoulder', 'L_Shoulder'), ('OP Neck', 'Neck'), ('OP Nose', 'Head')]
EDGES_P_B = [
    ('OP RShoulder', 'B RShoulder'), 
    ('OP LShoulder', 'B LShoulder'), 
    ('OP RElbow', 'B RElbow'), 
    ('OP LElbow', 'B LElbow'), 
    ('OP RWrist', 'B RWrist'), 
    ('OP LWrist', 'B LWrist'), 
    ('OP Nose', 'B Nose'), 
    ('OP Neck', 'B Neck'), 
    ('Spine (H36M)', 'B Spine (H36M)'), 
    ('Pelvis (MPII)', 'B Pelvis (MPII)'), 
    ('OP MidHip', 'B MidHip'),
    ('OP RHip', 'B RHip'), 
    ('OP LHip', 'B LHip'), 
    ('OP RKnee', 'B RKnee'), 
    ('OP LKnee', 'B LKnee'), 
    ('OP RAnkle', 'B RAnkle'), 
    ('OP LAnkle', 'B LAnkle')]


class PartsV2:
    def __init__(self, graph='CPR'):
        self.graph = graph
        self.part_names: List[str] = []  # All part names. Shape:(V,)
        self.heights: List[int] = []  # Heights. Same order with part_names. Shape:(V,)
        self.edges: List[Tuple(str, str)] = []  # Edges. Shape:(E, 2)

        if 'P' in graph:
            self.part_names.extend(J3D_IN_USE)
            self.heights.extend([J3D_HEIGHTS[name] for name in J3D_IN_USE])
            self.edges.extend(J3D_EDGES)

        if 'R' in graph:
            self.part_names.extend(POSE_IN_USE)
            self.heights.extend([POSE_HEIGHT[name] for name in POSE_IN_USE])
            self.edges.extend(POSE_INTERNAL_EDGES)
        
        if 'C' in graph:
            self.part_names.extend(CUSTOM_IN_USE)
            self.heights.extend([CUSTOM_HEIGHT[name] for name in CUSTOM_IN_USE])

        if 'B' in graph:
            self.part_names.extend(BONE)
            self.heights.extend([BONE_HEIGHT[name] for name in BONE])
            self.edges.extend(BONE_EDGES)
            pass

        if 'P' in graph and 'C' in graph:
            self.edges.extend(EDGES_P_C)
        
        if 'P' in graph and 'R' in graph:
            self.edges.extend(EDGES_P_R)
        
        if 'P' in graph and 'B' in graph:
            self.edges.extend(EDGES_P_B)

    def get_edge_indices(self) -> List[Tuple[int, int]]:
        edge_indices = []
        name2id = {name: idx for idx, name in enumerate(self.part_names)}
        for edge in self.edges:
            a,b = edge
            edge_indices.append((name2id[a], name2id[b]))
        return edge_indices

    def get_part_id(self, name) -> int:
        name2id = {name: idx for idx, name in enumerate(self.part_names)}
        return name2id[name]
    
    def get_parts(self) -> List[str]:
        # Return: list of part names
        return self.part_names
    
    def get_heights(self) -> List[int]:
        # Return: list of hight numbers
        return self.heights
    
    
    @classmethod
    def from_config(cls, cfg):
        return PartsV2(graph=cfg.MODEL.GRAPH)

    @staticmethod
    def filter_P(joints3d):
        # joints3d Shape: (V - num_keypoints, C - 3D_coord)
        vibe_j3D_indices = [VIBE_J3D_NAME_TO_IDX[name] for name in J3D_IN_USE]
        Vp = joints3d[vibe_j3D_indices]
        return Vp  # Shape: (V - num_keypoints_in_use, C - 3D_coord)

    @staticmethod
    def filter_R(pose3d):
        # pose3d Shape: (V - num_keypoints, C - 3D_coord)
        pose_in_use_indices = [SMPL_POSE_NAME_TO_IDX[name] for name in POSE_IN_USE]
        Vr = pose3d[pose_in_use_indices]
        return Vr
    
    @staticmethod
    def filter_C(pose3d):
        camera_pose_indices = [SMPL_POSE_NAME_TO_IDX[name] for name in CUSTOM_IN_USE]
        Vc = pose3d[camera_pose_indices]
        return Vc

    @staticmethod
    def filter_B(joints3d):
        vec_list = []
        for p1_name, p2_name in J3D_EDGES:
            p1_idx, p2_idx = [VIBE_J3D_NAME_TO_IDX[name] for name in (p1_name, p2_name)]
            p1_val, p2_val = joints3d[p1_idx], joints3d[p2_idx]
            vec = p2_val - p1_val
            vec_list.append(vec)
        vec_arr = np.stack(vec_list)

        return vec_arr

    def aggregate_features(self, Vp, Vr, Vc, Vb):
        # Vp shape: N,T,V,C
        features = []
        if 'P' in self.graph:
            features.append(Vp)
        if 'R' in self.graph:
            features.append(Vr)
        if 'C' in self.graph:
            features.append(Vc)
        if 'B' in self.graph:
            features.append(Vb)
        concat_V = torch.concat(features, dim=2)
        return concat_V