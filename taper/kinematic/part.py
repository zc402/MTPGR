from vibe.models.smpl import JOINT_MAP

class Parts:
    def __init__(self, use_cam_pose):
        self.use_cam_pose = use_cam_pose

    def name_id_map(self):
        # {part_name: part_idx}
        joint_map = JOINT_MAP
        if self.use_cam_pose:
            joint_map.update({'PRED_CAM': -1})
            return joint_map
        else:
            return joint_map