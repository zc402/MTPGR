import numpy as np


class SparseToDense:
    """
    Map the sparse part indices to dense part indices.

    Not all parts in SMPL are useful, some are ignored.
    The indices of useful parts are like: 2, 5, 8, 10, ..., which is called "sparse indices".
    However, the GCN requires dense part indices from 0 to N like 0, 1, 2,..., N, called "dense indices".
    Otherwise independent vertices with no edge will be created.
    This module maps the sparse indices to dense indices.
    """
    def __init__(self, heights_sparse, edges_sparse):
        self._heights_sparse = heights_sparse
        self._edges_sparse = edges_sparse

        self._used_part_ids = None  # The dense part index
        self._pid_s2d_map = None  # part id map {sparse: dense}
        self._construct_sparse_to_dense_map()

    def _construct_sparse_to_dense_map(self):
        # To select used parts: all_parts[used_part_ids] = used_parts
        # This id (s2d_id) is the index from 24 sparse part array, not the joint_map id
        self._used_part_ids = list(self._heights_sparse.keys())
        # part id map {sparse id: dense id}
        self._pid_s2d_map = {sparse_part_idx: i for i, sparse_part_idx in enumerate(self._used_part_ids)}

    def get_dense_id_height_map(self):
        """
        Convert {sparse_id: height} to {dense_id: height}
        :return: {dense_id: height}
        """
        heights_dense = {self._pid_s2d_map[s_id]: height for s_id, height in self._heights_sparse.items()}
        return heights_dense

    def get_dense_edges(self):
        """
        Convert sparse_id array to dense_id array
        :return: dense id array of shape (edges, 2)
        """
        edges_dense = [self._pid_s2d_map[s_id] for s_id in self._edges_sparse.ravel()]
        edges_dense = np.array(edges_dense).reshape(self._edges_sparse.shape)
        # Dense edge ids.
        # edges_1d = self._edges_sparse.reshape((-1))
        # edges_dense = map(self._pid_s2d_map.get, edges_1d.tolist())
        # edges_dense = list(edges_dense)
        # edges_dense = np.array(edges_dense)
        # edges_dense = edges_dense.reshape(self._edges_sparse.shape)
        return edges_dense

    def get_s2d_indices(self):
        # Select features of used parts from all 24 parts,
        # return a list of sparse_to_dense_index, which is selected from all 24 parts.
        return self._used_part_ids

    @classmethod
    def from_config(cls, cfg):
        from .heights import Heights
        from .edges import Edges

        heights = Heights.from_config(cfg)
        edges = Edges.from_config(cfg)

        instance = SparseToDense(
            heights.get_heights(),
            edges.get_edges()
        )
        return instance





