from pathlib import Path
import pickle
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

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

class FrameDrawer:

    def __init__(self, video_path: Path) -> None:
        self.cap = cv2.VideoCapture(str(video_path))
        self.last_read_num = -1

    def _get_video_frame(self, v_frame_num):

        while self.last_read_num + 1 < v_frame_num:
            # Skip until corresponding frame.
            _, _ = self.cap.read()
            self.last_read_num = self.last_read_num + 1
        
        assert self.last_read_num + 1 == v_frame_num
            # Read next frame
        ret, frame = self.cap.read()
        assert ret
        self.last_read_num = self.last_read_num + 1
        return frame
    
    def draw_frame(self, v_frame_num, ax:Axes):
        img = self._get_video_frame(v_frame_num)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title("Frame")

    def close(self):
        self.cap.release()

class ConfidenceDrawer:

    def draw_confidence(self, pred, ax):
        pred_softmax = self._softmax(pred)
        xs = list(range(pred_softmax.shape[0]))
        ax.bar(xs, pred_softmax)
        ax.set_title("Confidence")
    def _softmax(self, a):
        return np.exp(a)/np.sum(np.exp(a))


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

    kp = seq_res['batch_data']['kp'][0]  # Shape: (8767, 16, 3)
    frame_num = seq_res['batch_data']['frame_ids'][0]  # Shape: (8767,)
    name = seq_res['batch_data']['name'][0]  # Shape: (,)
    pred = seq_res['pred']  # Shape: (8767, 33)

    edge_drawer = EdgeDrawer()
    # edge_single_save_path = anime_save_folder / 'edge_single'
    # edge_single_save_path.mkdir(exist_ok=True)
    video_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.DATASET.VIDEO_DIR
    video_path = video_folder / (name + ".m4v")
    frame_drawer = FrameDrawer(video_path)
    confidence_drawer = ConfidenceDrawer()

    for i in range(3, len(seq_res['label'])):

        fig = plt.figure(figsize=(30, 20))

        # Figures: 1.image 2.spatial 3.temporal 4. 5.confidence_score 6.4-way_road

        edge_drawer.draw_single_character(kp[i], ax=fig.add_subplot(2, 3, 2, projection='3d'))
        edge_drawer.draw_multiple_character(kp[i-2], kp[i-1], kp[i], ax=fig.add_subplot(2, 3, 3, projection='3d'))
        frame_drawer.draw_frame(frame_num[i], ax=fig.add_subplot(2, 3, 1))

        pred_score = pred[i]  # Shape: (33,)
  
        confidence_drawer.draw_confidence(pred_score, ax=fig.add_subplot(2, 3, 4))

        plt.savefig(anime_save_folder / f"{i}.jpg")
        plt.close()    
        print(f"figure {i} saved")
    
    frame_drawer.close()
        


