from vibe.models.smpl import JOINT_NAMES

class Parts:
    def __init__(self, use_cam_pose: bool):
        """
        SMPL 3D joints mapping uses the array index of JOINT_NAMES, not JOINT_MAP.
        """
        self.name_id_map = {name: idx for idx, name in enumerate(JOINT_NAMES)}  # {part_name: part_idx}
        if use_cam_pose:
            self.name_id_map.update({'PRED_CAM': -1})

    def get_name_id_map(self):
        return self.name_id_map

    @classmethod
    def from_config(cls, cfg):
        return Parts(cfg.MODEL.USE_CAM_POSE)
