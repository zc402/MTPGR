from pathlib import Path
import pickle
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

from mtpgr.config.defaults import get_cfg_defaults
from mtpgr.kinematic.parts import Parts

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
        # s2d = SparseToDense.from_config(cfg)
        # edge_idx = s2d.get_dense_edges()  # (edge, 2). Index of joints to form edges
        edge_idx = Parts(False, False).get_edge_indices()

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

    def draw_spatial(self, joints, ax):
        """Draw a single character with spatial edges"""
        ax.set_title("Spatial Edges")
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(-0.75, 0.8)

        pt, edge = self._points_and_edges(joints)
        self._draw_spatial(ax, pt, edge, linewidth=2)

    def draw_spatial_temporal(self, j1, j2, j3, ax):
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

class ResultTextDrawer:

    def draw_result_text(self, result_num, ax:Axes):
        if result_num == 0:
            ges_text = "Stand In Attention"
            commanding_text = "Not Available"
        else:
            ges = (result_num - 1) % 8  # 0 - 7
            ges_text_list = ['Stop', 'Go Straight', 'Left Turn', 'Left Turn Waiting', 'Right Turn', 'Lane Changing', 'Slow Down', "Pull Over"]
            ori = (result_num - 1) // 8  # 0 - 3, FLBR
            direction_list = ['Subject Vehicle', 'Vehicle on the Left', 'Ongoing Vehicle', 'Vehicle on the Right']

            commanding = [[0, 1, 3, 3,  2, 0, 0, 0], [1,2,0,0, 3,1,1,1], [2,3,1,1,0,2,2,2], [3,0,2,2,1,3,3,3]]

            ges_text = ges_text_list[ges]
            commanding_text = direction_list[commanding[ori][ges]]
        
        # Gesture
        ax.text(0.5, 0.4, "Gesture:", horizontalalignment='center', verticalalignment='center', fontsize=18)
        ax.text(0.5, 0.3, ges_text, horizontalalignment='center', verticalalignment='center', fontsize=36)

        # Orientation
        ax.text(0.5, 0.7, "Commanding Direction:", horizontalalignment='center', verticalalignment='center', fontsize=18)
        ax.text(0.5, 0.6, commanding_text, horizontalalignment='center', verticalalignment='center', fontsize=30)
        ax.axis('off')
        
        
class CrossRoadDrawer:

    def draw_cross_road(self, ax:Axes):
        img_path = Path('docs', 'crossroad.jpg')
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)


if __name__ == "__main__":
    # plt.style.use('dark_background')
    plt.rcParams.update({'font.size':16})

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/no_camera.yaml')

    result_path = Path('output', cfg.OUTPUT, 'result.pkl')
    anime_save_folder = Path('output', cfg.OUTPUT, 'anime')
    anime_save_folder.mkdir(exist_ok=True)

    with result_path.open('rb') as f:
        result = pickle.load(f)

    seq_number = 0

    seq_res = result[seq_number]

    j3D = seq_res['batch_data']['kp'][0]  # Shape: (8767, 16, 3)
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
    result_text_drawer = ResultTextDrawer()
    cross_road_drawer = CrossRoadDrawer()

    for i in range(3, len(seq_res['label'])):

        fig = plt.figure(figsize=(30, 20))

        # Figures: 1.image 2.spatial 3.temporal 4. 5.confidence_score 6.4-way_road

        edge_drawer.draw_spatial(j3D[i], ax=fig.add_subplot(2, 3, 2, projection='3d'))
        edge_drawer.draw_spatial_temporal(j3D[i-2], j3D[i-1], j3D[i], ax=fig.add_subplot(2, 3, 3, projection='3d'))
        frame_drawer.draw_frame(frame_num[i], ax=fig.add_subplot(2, 3, 1))

        pred_score = pred[i]  # Shape: (33,)
        pred_class = np.argmax(pred_score)
  
        confidence_drawer.draw_confidence(pred_score, ax=fig.add_subplot(2, 3, 4))
        result_text_drawer.draw_result_text(pred_class, ax=fig.add_subplot(2, 3, 5))
        cross_road_drawer.draw_cross_road(ax=fig.add_subplot(2, 3, 6))

        plt.savefig(anime_save_folder / f"{i}.jpg")
        plt.close()    
        print(f"figure {i} saved")
    
    frame_drawer.close()
        


