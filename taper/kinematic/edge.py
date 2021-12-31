"""
Spatially connected joints
"""
import numpy as np
from vibe.models.smpl import JOINT_MAP
from taper.kinematic.sparse_to_dense import part_after_take
_edges = [
    'OP Neck', 'OP RShoulder',  # 右肩
    'OP Neck', 'OP LShoulder',  # 左肩

    'OP RShoulder', 'OP RElbow',  # 右大臂
    'OP LShoulder', 'OP LElbow',  # 左大臂

    'OP RElbow', 'OP RWrist',  # 右小臂
    'OP LElbow', 'OP LWrist',  # 左小臂

    'OP Neck', 'OP MidHip',  # 躯干

    'OP MidHip', 'OP RHip',  # 右跨
    'OP MidHip', 'OP LHip',  # 左跨

    'OP RHip', 'OP RKnee',  # 大腿
    'OP LHip', 'OP LKnee',

    'OP RKnee', 'OP RAnkle',  # 小腿
    'OP LKnee', 'OP LAnkle',
]

edges = list(map(JOINT_MAP.get, _edges))  # edge_list[part_idx]
edges = np.array(edges).reshape((-1, 2))  # part_idx array of shape (edges, 2)

