"""
Height value for each joint.
"""
from vibe.models.smpl import JOINT_MAP

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
# dict: heights[part] = height_value
heights = {}
for h, parts in enumerate(_heights):
    for part in parts:
        part_idx = JOINT_MAP[part]
        heights[part_idx] = h


