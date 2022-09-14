import math
from pathlib import Path
import pickle
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
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
    ('OP LShoulder', 'Neck'), ('OP RShoulder', 'Neck'),
    ('Neck', 'Head'),
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
    ax = plt.axes(projection ='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0.75, 1.25)

    j3d_points = [VIBE_j3d[VIBE_J3D_NAME_TO_IDX[name]] for name in J3D_IN_USE]
    pose_points = [VIBE_j3d[VIBE_J3D_NAME_TO_IDX[name]] for name in FAKE_POSE_IN_USE]

    ax.scatter3D(*zip(*j3d_points))

    j3d_edge_indices = [(VIBE_J3D_NAME_TO_IDX[a], VIBE_J3D_NAME_TO_IDX[b]) for a,b in J3D_EDGES]
    j3d_edges = [(VIBE_j3d[a], VIBE_j3d[b]) for a, b in j3d_edge_indices]

    plt.show()



if __name__ == "__main__":
    # plt.style.use('dark_background')
    plt.rcParams.update({'font.size':16})

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/no_camera.yaml')
    result_path = Path('output', cfg.OUTPUT, 'result.pkl')

    with result_path.open('rb') as f:
        result = pickle.load(f)

    seq_number = 0
    seq_res = result[seq_number]

    index = 100  # index of the sequence, not frame number. the frame number does not contain untracked frames and is larger. 
    kp = seq_res['batch_data']['kp'][0][index].numpy()
    kp = to_matplotlib_axes(kp)
    draw_points(kp)
