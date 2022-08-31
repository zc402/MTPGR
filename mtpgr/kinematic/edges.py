"""
Spatially connected joints
"""
from typing import List
import numpy as np
from mtpgr.utils.log import log

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

    def __init__(self, pname_id_map: dict, use_cam_pose: bool, no_spatial_edges: bool=False):
        if use_cam_pose:  # Use the camera pose as a vertice
            self._edges_name.extend(['OP Neck', 'PRED_CAM'])
            log.debug("Edges: Using camera features (default)")
        else:
            log.debug("Edges: Using NO camera mode")
        
        if no_spatial_edges:  # Disable spatial edges for ablation study
            self._edges_name = ['OP Nose', 'OP Neck']
            log.debug("Edges: Using NO spatial edge mode")
            assert not use_cam_pose, "'No spatial edge' option is conflict with 'camera pose' option"

        self.edges = list(map(pname_id_map.get, self._edges_name))  # edge_list[part_idx]
        self.edges = np.array(self.edges).reshape((-1, 2))  # part_id array of shape (num_edges, 2)

    def get_edges(self):
        return self.edges

    @classmethod
    def from_config(cls, cfg):
        from .parts import Parts
        parts = Parts.from_config(cfg)
        if cfg.MODEL.NO_SPATIAL_EDGES:
            return Edges(parts.get_name_id_map(), False, True)
        return Edges(parts.get_name_id_map(), cfg.MODEL.USE_CAMERA_POSE)



