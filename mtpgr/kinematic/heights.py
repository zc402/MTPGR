"""
Height value for each joint.
"""


class Heights:

    def __init__(self, pname_id_map: dict, use_cam_pose: bool):
        """
        The relative height of each body part.
        Return heights[part_id] = height_value
        :param pname_id_map: dict {part_name: part_id}
        :param use_cam_pose: Use camera pose parameters
        """
        # _heights: list_height[list_parts[part_name]]
        self._heights_name = [
            ['OP RAnkle', 'OP LAnkle'],
            ['OP RKnee', 'OP LKnee'],
            ['OP RHip', 'OP LHip'],
            ['OP MidHip'],
            ['OP RWrist', 'OP LWrist'],
            ['OP RElbow', 'OP LElbow'],
            ['OP RShoulder', 'OP LShoulder'],
            ['OP Neck'],
            ['OP Nose']
        ]

        if use_cam_pose:
            self._heights_name[3] = ['OP MidHip', 'PRED_CAM']

        assert self._no_duplicated_names(self._heights_name)

        self.heights = {}  # dict: heights[part_id] = height_value
        for h, parts_same_height in enumerate(self._heights_name):
            for part_name in parts_same_height:
                part_id = pname_id_map[part_name]
                self.heights[part_id] = h

    @staticmethod
    def _no_duplicated_names(heights_name):
        # Ensure no duplicated height names
        used_names = []
        for line in heights_name:
            for name in line:
                if name in used_names:
                    return False
                else:
                    used_names.append(name)
        return True

    def get_heights(self):
        return self.heights

    @classmethod
    def from_config(cls, cfg):
        from .parts import Parts
        parts = Parts.from_config(cfg)
        return Heights(parts.get_name_id_map(), cfg.MODEL.USE_CAM_POSE)
