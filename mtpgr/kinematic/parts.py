from typing import List, Tuple
from vibe.models.smpl import JOINT_NAMES
from mtpgr.utils.log import log

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
J3D_IN_USE = ['OP RAnkle', 'OP LAnkle', 'OP RKnee', 'OP LKnee', 'OP RHip', 'OP LHip', 'OP MidHip', 
'OP RWrist', 'OP LWrist', 'OP RElbow', 'OP LElbow', 'OP RShoulder', 'OP LShoulder', 'OP Neck', 'OP Nose']

J3D_HEIGHTS = {
    'OP RAnkle': 0,
    'OP LAnkle': 0,
    'OP RKnee': 1,
    'OP LKnee': 1,
    'OP RHip': 2,
    'OP LHip': 2,
    'OP MidHip': 3,
    'OP RWrist': 4,
    'OP LWrist': 4,
    'OP RElbow': 5,
    'OP LElbow': 5,
    'OP RShoulder': 6,
    'OP LShoulder': 6,
    'OP Neck': 7,
    'OP Nose': 8,
}

J3D_EDGES = [
        ('OP Neck', 'OP RShoulder'),  # 右肩
        ('OP Neck', 'OP LShoulder'),  # 左肩

        ('OP RShoulder', 'OP RElbow'),  # 右大臂
        ('OP LShoulder', 'OP LElbow'),  # 左大臂

        ('OP RElbow', 'OP RWrist'),  # 右小臂
        ('OP LElbow', 'OP LWrist'),  # 左小臂

        ('OP Nose', 'OP Neck'),  # 头
        ('OP Neck', 'OP MidHip'),  # 躯干

        ('OP MidHip', 'OP RHip'),  # 右跨
        ('OP MidHip', 'OP LHip'),  # 左跨

        ('OP RHip', 'OP RKnee'),  # 大腿
        ('OP LHip', 'OP LKnee'),

        ('OP RKnee', 'OP RAnkle'),  # 小腿
        ('OP LKnee', 'OP LAnkle'),
    ]


POSE_HEIGHT = {}  # Pose: theta, SMPL Joint rotations.

CUSTOM_IN_USE = ['Pelvis']  # From SMPL POSE

CUSTOM_HEIGHT = {'Pelvis': 100}  # Root rotation, connects to joints.

CUSTOM_EDGES = [(joint, 'Pelvis') for joint in J3D_HEIGHTS.keys()]  # Connect each joints(J3D) to pelvis(theta)

class Parts:
    def __init__(self, use_cam_pose: bool):
        self.use_cam_pose = use_cam_pose
        self.part_names: List[str] = []  # All part names. Shape:(V,)
        self.heights: List[int] = []  # Heights. Same order with part_names. Shape:(V,)
        self.edges: List[Tuple(str, str)] = []  # Edges. Shape:(E, 2)

        if use_cam_pose:
            log.debug("use camera pose")
            self.part_names.extend(J3D_IN_USE)
            self.part_names.extend(CUSTOM_IN_USE)
            self.heights.extend([J3D_HEIGHTS[name] for name in J3D_IN_USE])
            self.heights.extend([CUSTOM_HEIGHT[name] for name in CUSTOM_IN_USE])
            self.edges.extend(J3D_EDGES)
            self.edges.extend(CUSTOM_EDGES)
            
        else:
            log.debug("no camera pose")
            self.part_names.extend(J3D_IN_USE)
            self.heights.extend([J3D_HEIGHTS[name] for name in J3D_IN_USE])
            self.edges.extend(J3D_EDGES)

    def get_edge_indices(self) -> List[Tuple[int, int]]:
        edge_indices = []
        name2id = {name: idx for idx, name in enumerate(self.part_names)}
        for edge in self.edges:
            a,b = edge
            edge_indices.append((name2id[a], name2id[b]))
        return edge_indices
    
    def get_parts(self) -> List[str]:
        # Return: list of part names
        return self.part_names
    
    def get_heights(self) -> List[int]:
        # Return: list of hight numbers
        return self.heights
    
    def get_features(self, VIBE_j3D=None, VIBE_j2D=None, SMPL_pose=None) -> List[float]:
        features = []
        vibe_j3D_indices = [VIBE_J3D_NAME_TO_IDX[name] for name in J3D_IN_USE]
        j3D_values = VIBE_j3D[vibe_j3D_indices]
        features.extend(j3D_values)

        if self.use_cam_pose:
            smpl_pose_indices = [SMPL_POSE_NAME_TO_IDX[name] for name in CUSTOM_IN_USE]
            custom_values = SMPL_pose[smpl_pose_indices]
            features.extend(custom_values)
        
        return features
    
    @classmethod
    def from_config(cls, cfg):
        return Parts(cfg.MODEL.USE_CAMERA_POSE)


# class Parts:
#     def __init__(self, use_cam_pose: bool):
#         """

#         """
#         self.jname_to_idx = {name: idx for idx, name in enumerate(VIBE_3D_joints)}  # {part_name: part_idx}
#         if use_cam_pose:
#             assert (48 in self.jname_to_idx.values()) and (49 not in self.jname_to_idx.values())
#             self.jname_to_idx.update({'PRED_CAM': 49})

#     def get_name_id_map(self):
#         return self.jname_to_idx

#     @classmethod
#     def from_config(cls, cfg):
#         return Parts(cfg.MODEL.USE_CAMERA_POSE)
