from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from mtpgr.config import get_cfg_defaults
from mtpgr.network.adjacency_matrix import AdjacencyMatrix

# ========== Read 3D joints data ==========

cfg = get_cfg_defaults()
config_path:Path = Path('configs', 'stgcn.yaml')  # stgcn.yaml  stgcn_cam.yaml  mtpgr_wo_rot.yaml
cfg.merge_from_file(config_path)

adj_mat = AdjacencyMatrix.from_config(cfg)
A = adj_mat.get_adjacency()

display_joints = [0,1,2,3,10,11,12,13,14,15,16,17]
plt.yticks(range(18))
plt.xticks(range(18))
plt.imshow(A[2][0:18, 0:18], cmap='Greys')  # [display_joints][:, display_joints]  A[1][0:18, 0:18]+
plt.show()
