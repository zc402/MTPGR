"""
Map the sparse part indices to dense part indices.

Not all parts in SMPL are selected, some are ignored.
The indices of selected parts are like: 2, 5, 8, 10, ..., which is sparse.
However, the GCN requires dense part indices from 0 to N like 0, 1, 2,..., N,
otherwise independent vertices with no edge will be created.
Therefore, this module is provided to map the sparse to dense.
"""
import numpy as np

from .height import Heights  # The elements in 'height' are unique, its key (sparse part index) is used to select parts.
from .edge import Edges


class SparseToDense:
    def __init__(self, use_cam_pose):
        # self.use_came_pose = use_cam_pose
        self._heights_sparse = Heights(use_cam_pose).get()
        self._edges_sparse = Edges(use_cam_pose).get()

        self.part_id_dense = None  # The dense part index
        self._pid_s2d_map = None  # part id map {sparse: dense}
        self._construct_dense_part_idx()

    def _construct_dense_part_idx(self):
        # The indices of selected parts from all SMPL parts.
        self.part_id_dense = list(self._heights_sparse.keys())  # To select used parts from all parts: all_parts[dense_indices]
        # part id map {sparse id: dense id}
        self._pid_s2d_map = {sparse_part_idx: i for i, sparse_part_idx in enumerate(self.part_id_dense)}

    def dense_heights(self):
        # Dense height values
        heights_dense = {}  # height idx after '.take()'
        for sparse_part_idx, height_value in self._heights_sparse.items():
            dense_part_idx = self._pid_s2d_map[sparse_part_idx]
            heights_dense[dense_part_idx] = height_value
        return heights_dense

    def dense_edges(self):
        # Dense edges. array of shape (edges, 2)
        edges_1d = self._edges_sparse.reshape((-1))
        edges_dense = map(self._pid_s2d_map.get, edges_1d.tolist())
        edges_dense = list(edges_dense)
        edges_dense = np.array(edges_dense)
        edges_dense = edges_dense.reshape(self._edges_sparse.shape)
        return edges_dense







