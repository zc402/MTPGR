import math
from pathlib import Path
import pickle
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, Rectangle
import torch
import cv2

from mtpgr.config.defaults import get_cfg_defaults
from mtpgr.kinematic.parts import Parts, VIBE_J3D_NAME_TO_IDX, J3D_IN_USE, J3D_EDGES


# The POSE is rotation angles, but visualization needs 3D locations, therefore each pose is translated to corresponding 3D joint
FAKE_POSE_IN_USE = ['OP LAnkle', 'OP RAnkle', 
'OP LKnee', 'OP RKnee', 
'OP LHip', 'OP RHip', 
'OP LShoulder', 'OP RShoulder', 
'OP LElbow', 'OP RElbow', 
'OP LWrist', 'OP RWrist', 
'OP Neck']

FAKE_POSE_INTERNAL_EDGES = [
    ('OP LAnkle', 'OP LKnee'), ('OP RAnkle', 'OP RKnee'),
    ('OP LKnee', 'OP LHip'), ('OP RKnee', 'OP RHip'),
    ('OP LHip', 'OP RHip'),
    ('OP LHip', 'OP LShoulder'), ('OP RHip', 'OP RShoulder'),
    ('OP LShoulder', 'OP LElbow'), ('OP RShoulder', 'OP RElbow'),
    ('OP LElbow', 'OP LWrist'), ('OP RElbow', 'OP RWrist'),
    ('OP LShoulder', 'OP RShoulder'),
    ('OP LShoulder', 'OP Neck'), ('OP RShoulder', 'OP Neck'),
]

FAKE_POSE_J3D_EDGES = [(a, a) for a in FAKE_POSE_IN_USE]

def rotation_matrix(axis, theta):
    """ Euler-Rodrigues formula:
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def to_matplotlib_axes(kps):
    new_kps = [(x, z, -y) for x, y, z in kps]
    
    axis = [1, 0, 0]
    theta = -0.2
    rot_mat = rotation_matrix(axis, theta)
    new_kps = [np.dot(rot_mat, v) for v in new_kps]

    return new_kps


def draw_points(VIBE_j3d: np.ndarray):

    enable_axis = False
    enable_pose = True
    enable_cam = True
    enable_pose_internal = True
    enable_cam_j3d = True
    enable_pose_j3d = True
    enable_heights = False
    enable_j3d_number = True

    plt.figure(figsize=(20, 20))
    ax = plt.axes(projection ='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0.75, 1.25)

    if not enable_axis:
        plt.axis('off')

    # --------Keypoints--------
    j3d_points = [VIBE_j3d[VIBE_J3D_NAME_TO_IDX[name]] for name in J3D_IN_USE]

    j3d_tl = [(x,y+0.2, z) for x,y,z in VIBE_j3d]  # The y-translate of j3d for showing the pose. Prevent overlap on graph.
    pose_points = [j3d_tl[VIBE_J3D_NAME_TO_IDX[name]] for name in FAKE_POSE_IN_USE]
    # pose_points = [(x, y+0.2, z) for x,y,z in pose_points]  # Prevent overlap

    ax.scatter3D(*zip(*j3d_points), color='tab:green', marker='o', s=80)
    if enable_j3d_number:
        for n, coord3d in enumerate(j3d_points):
            ax.text(*coord3d, str(n), color='black')


    if enable_pose:
        ax.scatter3D(*zip(*pose_points), color='tab:orange', marker='v', s=80)

    # ----------Edges-----------
    # J3D
    j3d_edge_indices = [(VIBE_J3D_NAME_TO_IDX[a], VIBE_J3D_NAME_TO_IDX[b]) for a,b in J3D_EDGES]
    j3d_edges = [(VIBE_j3d[a], VIBE_j3d[b]) for a, b in j3d_edge_indices]
    for edge in j3d_edges:
        xyz = zip(*edge)  # [(xyz1), (xyz2), (xyz3)]
        ax.plot3D(*xyz, color='tab:olive')

    # Pose internal
    if enable_pose_internal:
        pose_internal_edges = [(j3d_tl[VIBE_J3D_NAME_TO_IDX[a]], j3d_tl[VIBE_J3D_NAME_TO_IDX[b]]) 
            for a, b in FAKE_POSE_INTERNAL_EDGES]
        for edge in pose_internal_edges:
            ax.plot3D(*zip(*edge), color='tab:pink')

    # Pose-j3d
    if enable_pose_j3d:
        pose_j3d_edges = [(VIBE_j3d[VIBE_J3D_NAME_TO_IDX[a]], j3d_tl[VIBE_J3D_NAME_TO_IDX[b]]) 
            for a, b in FAKE_POSE_J3D_EDGES]
        for edge in pose_j3d_edges:
            ax.plot3D(*zip(*edge), color='tab:cyan', linestyle='dashed')

    # camera
    if enable_cam:
        camera = (0, -0.5, 1.1)
        ax.scatter3D(*camera, color='tab:red', marker='*', s=100)

    if enable_cam_j3d:
        camera_edges = [(camera, VIBE_j3d[VIBE_J3D_NAME_TO_IDX[a]]) for a in J3D_IN_USE]
        for edge in camera_edges:
            ax.plot3D(*zip(*edge), color='silver', linestyle='dotted')

    # Height surface
    if enable_heights:
        cmap = plt.get_cmap('Greys')
        
        representative_joints = ['OP LAnkle', 'OP LKnee', 'OP LHip', 'OP MidHip', 'OP LElbow', 'OP LShoulder', 'OP Neck', 'OP Nose']
        color_range = np.linspace(0.8, 0.3, len(representative_joints))
        colors = [cmap(n) for n in color_range]
        repj_heights = [VIBE_j3d[VIBE_J3D_NAME_TO_IDX[a]][2] + 0.08 for a in representative_joints]
        repj_heights[2] = repj_heights[2] - 0.05
        for n, h in enumerate(repj_heights):
            p = Rectangle((-0.3, -0.3), 0.6, 0.6, alpha=0.6, color=colors[n])
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=h, zdir="z")

    plt.show()



if __name__ == "__main__":
    # plt.style.use('dark_background')
    plt.rcParams.update({'font.size':16})

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/mtpgr_wo_rot.yaml')
    result_path = Path('output', cfg.MODEL.NAME, 'result.pkl')

    with result_path.open('rb') as f:
        result = pickle.load(f)

    seq_number = 0
    seq_res = result[seq_number]

    index = 100  # index of the sequence, not frame number. the frame number does not contain untracked frames and is larger. 
    kp = seq_res['batch_data']['kp'][0][index].numpy()
    kp = to_matplotlib_axes(kp)
    draw_points(kp)
