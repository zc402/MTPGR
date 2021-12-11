# 每个关键点所处的高度{关键点index: 高度}。
# Height Layers
from vibe.models.smpl import JOINT_MAP
from taper.kinematic.parts import p2pat

# _heights: list_height[list_parts[part_name]]
_heights = [
    ['OP RAnkle', 'OP LAnkle'],
    ['OP RKnee', 'OP LKnee'],
    ['OP RHip', 'OP LHip'],
    ['OP MidHip'],
    ['OP RWrist', 'OP LWrist'],
    ['OP RElbow', 'OP LElbow'],
    ['OP RShoulder', 'OP LShoulder'],
    ['OP Neck']
]
# dict[part] = height
heights = {}
for h, parts in enumerate(_heights):
    for part in parts:
        part_idx = JOINT_MAP[part]
        heights[part_idx] = h

heights_at = {}  # height idx after '.take()'
for k, v in heights.items():
    k_at = p2pat(k)
    heights_at[k_at] = v
