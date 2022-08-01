"""
Spatially connected joints
"""
import numpy as np


class Edges:

    _edges_name = [
        'OP Neck', 'OP RShoulder',  # 右肩
        'OP Neck', 'OP LShoulder',  # 左肩

        'OP RShoulder', 'OP RElbow',  # 右大臂
        'OP LShoulder', 'OP LElbow',  # 左大臂

        'OP RElbow', 'OP RWrist',  # 右小臂
        'OP LElbow', 'OP LWrist',  # 左小臂

        'OP Nose', 'OP Neck',  # 头
        'OP Neck', 'OP MidHip',  # 躯干

        'OP MidHip', 'OP RHip',  # 右跨
        'OP MidHip', 'OP LHip',  # 左跨

        'OP RHip', 'OP RKnee',  # 大腿
        'OP LHip', 'OP LKnee',

        'OP RKnee', 'OP RAnkle',  # 小腿
        'OP LKnee', 'OP LAnkle',
    ]

    def __init__(self, pname_id_map: dict, use_cam_pose: bool):
        if use_cam_pose:
            self._edges_name.extend(['OP Neck', 'PRED_CAM'])

        self.edges = list(map(pname_id_map.get, self._edges_name))  # edge_list[part_idx]
        self.edges = np.array(self.edges).reshape((-1, 2))  # part_id array of shape (num_edges, 2)

    def get_edges(self):
        return self.edges

    @classmethod
    def from_config(cls, cfg):
        from .parts import Parts
        parts = Parts.from_config(cfg)
        return Edges(parts.get_name_id_map(), cfg.MODEL.USE_CAMERA_POSE)



