from vibe.models.smpl import JOINT_MAP

class Parts:
    def __init__(self, use_cam_pose: bool):
        self.name_id_map = JOINT_MAP  # {part_name: part_idx}
        if use_cam_pose:
            self.name_id_map.update({'PRED_CAM': -1})

    def get_name_id_map(self):
        return self.name_id_map

    @classmethod
    def from_config(cls, cfg):
        return Parts(cfg.MODEL.USE_CAM_POSE)
