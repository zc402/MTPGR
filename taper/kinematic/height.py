"""
Height value for each joint.
"""
from .part import Parts

class Heights:

    # _heights: list_height[list_parts[part_name]]
    _heights_name = [
        ['OP RAnkle', 'OP LAnkle'],
        ['OP RKnee', 'OP LKnee'],
        ['OP RHip', 'OP LHip'],
        ['OP MidHip'],
        ['OP RWrist', 'OP LWrist'],
        ['OP RElbow', 'OP LElbow'],
        ['OP RShoulder', 'OP LShoulder'],
        ['OP Neck']
    ]

    def __init__(self, use_cam_pose):
        self.use_cam_pose = use_cam_pose
        self.parts = Parts(use_cam_pose)

    def get(self):
        if self.use_cam_pose:
            self._heights_name[3] = ['OP MidHip', 'PRED_CAM']

        # dict: heights[part] = height_value
        heights = {}
        for h, parts in enumerate(self._heights_name):
            for part in parts:
                part_idx = self.parts.name_id_map()[part]
                heights[part_idx] = h

        return heights


