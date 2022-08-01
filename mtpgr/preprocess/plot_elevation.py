from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from mtpgr.config import get_cfg_defaults
from mtpgr.kinematic import SparseToDense

# ========== Read 3D joints data ==========

cfg = get_cfg_defaults()
cfg.merge_from_file('configs/train_nocam.yaml')

data_dir = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.VIBE_DIR / '4K9A0220.pkl' 
with open(data_dir, 'rb') as f:
    vibe_data = pickle.load(f)

def read_vibe_frame(frame: int, offset: float=0.0):
    frame1 = vibe_data[frame][1]['joints3d'][0]  # frames, person(1=police), info, 0. Result shape: (49, 3)

    s2d = SparseToDense.from_config(cfg)
    filter = s2d.get_s2d_indices()
    joints = frame1[filter]  # Dense joints

    edge_idx = s2d.get_dense_edges()  # (edge, 2). Index of joints to form edges

    # Points
    pxs = joints[:, 0] + offset
    pys = joints[:, 2]
    pzs = -joints[:, 1]
    points = {'xs': pxs, 'ys': pys, 'zs': pzs}

    edges = []  # (edge_num, line{xs, ys, zs})
    for pair in edge_idx:
        # pair_coord = joints[pair]  # (2:pair, 3:3Dcoords)
        exs, eys, ezs = pxs[pair], pys[pair], pzs[pair]
        edges.append({'xs': exs, 'ys': eys, 'zs': ezs})
    
    return points, edges


# ========== matplotlib ==========

def draw_spatial(ax: Axes, points: dict, edges: list, **kwargs):
    ax.scatter3D(points['xs'], points['ys'], points['zs'], **kwargs)
    for pair in edges:
        ax.plot3D(pair['xs'], pair['ys'], pair['zs'], **kwargs)

def draw_elevation(ax: Axes, points: dict):
    num_points = len(points['xs'])
    ground = -0.75
    for i in range(num_points):
        xs = [points['xs'][i], points['xs'][i]]
        ys = [points['ys'][i], points['ys'][i]]
        zs = [points['zs'][i], ground]
        ax.plot3D(xs, ys, zs, color='tab:green', linestyle='dotted')

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.6, 0.6)
ax.set_zlim(-0.75, 0.8)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# ax.axis('off')

points1, edges1 = read_vibe_frame(1, offset=0)

draw_spatial(ax, points1, edges1)
draw_elevation(ax, points1)

plt.show()