from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from mtpgr.config.defaults import get_cfg_defaults
from mtpgr.kinematic import SparseToDense

# ========== Read 3D joints data ==========

cfg = get_cfg_defaults()

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
        ax.plot3D(pair['xs'], pair['ys'], pair['zs'], color='tab:blue', **kwargs)

def draw_temporal(ax: Axes, points1: dict, points2: dict, **kwargs):
    num_joints = len(points1['xs'])
    for i in range(0, num_joints):
        txs = [points1['xs'][i], points2['xs'][i]]
        tys = [points1['ys'][i], points2['ys'][i]]
        tzs = [points1['zs'][i], points2['zs'][i]]
        ax.plot3D(txs, tys, tzs, **kwargs)

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_xlim(0, 2)
ax.set_ylim(-1, 1)
ax.set_zlim(-0.75, 1.25)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.axis('off')

frame = 2000
points1, edges1 = read_vibe_frame(frame, offset=0.2)
points2, edges2 = read_vibe_frame(frame + 20, offset=1.0)
points3, edges3 = read_vibe_frame(frame + 40, offset=1.8)

draw_spatial(ax, points1, edges1, linewidth=2)
draw_spatial(ax, points2, edges2, linewidth=2)
draw_spatial(ax, points3, edges3, linewidth=2)
draw_temporal(ax, points1, points2, color='tab:orange', linestyle='dashed')
draw_temporal(ax, points2, points3, color='tab:green', linestyle='dashed')

plt.show()