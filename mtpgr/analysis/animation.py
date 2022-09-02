from pathlib import Path
import pickle
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt

from mtpgr.config.defaults import get_cfg_defaults
from mtpgr.kinematic.sparse_to_dense import SparseToDense

class EdgeDrawer:

    def _draw_spatial(self, ax: Axes, points: dict, edges: list, **kwargs):
        ax.scatter3D(points['xs'], points['ys'], points['zs'], **kwargs)
        for i, (x, y, z) in enumerate(zip(points['xs'], points['ys'], points['zs'])):
            ax.text(x, y, z, str(i), color='red')  # joint number
        for pair in edges:
            ax.plot3D(pair['xs'], pair['ys'], pair['zs'], color='tab:blue', **kwargs)

    def _draw_temporal(self, ax: Axes, points1: dict, points2: dict, **kwargs):
        num_joints = len(points1['xs'])
        for i in range(0, num_joints):
            txs = [points1['xs'][i], points2['xs'][i]]
            tys = [points1['ys'][i], points2['ys'][i]]
            tzs = [points1['zs'][i], points2['zs'][i]]
            ax.plot3D(txs, tys, tzs, **kwargs)

    def _points_and_edges(self, joints, offset=0.0):
        """Return points and edges array with matplotlib format
        Expect keypoint shape: (49, 3)"""
        s2d = SparseToDense.from_config(cfg)
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

    def draw_single_character(self, joints, save_path):
        """Draw a single character with spatial edges"""

        fig = plt.figure()
        ax = plt.axes(projection ='3d')
        ax.set_xlim(0, 2)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-0.75, 1.25)

        pt, edge = self._points_and_edges(joints)
        self._draw_spatial(ax, pt, edge, linewidth=2)

        fig.savefig(save_path)
        plt.close()        


    def draw_multiple_character(self, j1, j2, j3, save_path):
        """Draw multiple character with spatial and temporal edges"""

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/default_model.yaml')

    result_path = Path('output', cfg.OUTPUT, 'result.pkl')
    anime_save_folder = Path('output', cfg.OUTPUT, 'anime')
    anime_save_folder.mkdir(exist_ok=True)

    with result_path.open('rb') as f:
        result = pickle.load(f)

    seq_number = 0

    seq_res = result[seq_number]

    kp = seq_res['batch_data']['kp']  # Shape: (1, 8767, 16, 3)

    edge_drawer = EdgeDrawer()
    edge_single_save_path = anime_save_folder / 'edge_single'
    edge_single_save_path.mkdir(exist_ok=True)
    for i in range(len(seq_res['label'])):
        edge_drawer.draw_single_character(kp[0, i], edge_single_save_path / f"{i}.pdf")
        print(f"figure {i} saved")
        


