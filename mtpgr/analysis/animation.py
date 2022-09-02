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
            ax.text(x, y, z, str(i), color='red', fontsize=6)  # joint number
        for pair in edges:
            ax.plot3D(pair['xs'], pair['ys'], pair['zs'], **kwargs)

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

    def draw_single_character(self, joints, ax):
        """Draw a single character with spatial edges"""
        ax.set_title("Spatial Edges")
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(-0.75, 0.8)

        pt, edge = self._points_and_edges(joints)
        self._draw_spatial(ax, pt, edge, linewidth=2)

    def draw_multiple_character(self, j1, j2, j3, ax):
        """Draw multiple character with spatial and temporal edges"""
        ax.set_title("Temporal Edges")
        ax.set_xlim(0, 2)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-0.75, 1.25)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        points1, edges1 = self._points_and_edges(j1, offset=0.2)
        points2, edges2 = self._points_and_edges(j2, offset=1.0)
        points3, edges3 = self._points_and_edges(j3, offset=1.8)

        self._draw_spatial(ax, points1, edges1, linewidth=2)
        self._draw_spatial(ax, points2, edges2, linewidth=2)
        self._draw_spatial(ax, points3, edges3, linewidth=2)
        self._draw_temporal(ax, points1, points2, color='tab:orange', linestyle='dashed')
        self._draw_temporal(ax, points2, points3, color='tab:green', linestyle='dashed')


if __name__ == "__main__":
    # plt.style.use('dark_background')

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/no_camera.yaml')

    result_path = Path('output', cfg.OUTPUT, 'result.pkl')
    anime_save_folder = Path('output', cfg.OUTPUT, 'anime')
    anime_save_folder.mkdir(exist_ok=True)

    with result_path.open('rb') as f:
        result = pickle.load(f)

    seq_number = 0

    seq_res = result[seq_number]

    kp = seq_res['batch_data']['kp']  # Shape: (1, 8767, 16, 3)

    edge_drawer = EdgeDrawer()
    # edge_single_save_path = anime_save_folder / 'edge_single'
    # edge_single_save_path.mkdir(exist_ok=True)
    for i in range(3, len(seq_res['label'])):

        fig = plt.figure(figsize=(15, 10))

        # Figures: 1.image 2.spatial 3.temporal 4. 5.confidence_score 6.4-way_road

        edge_drawer.draw_single_character(kp[0, i], ax=fig.add_subplot(2, 3, 2, projection='3d'))
        edge_drawer.draw_multiple_character(kp[0, i-2], kp[0, i-1], kp[0, i], ax=fig.add_subplot(2, 3, 3, projection='3d'))

        plt.savefig(anime_save_folder / f"{i}.pdf")
        plt.close()    
        print(f"figure {i} saved")
        


