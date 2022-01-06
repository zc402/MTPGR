"""
Map the **sparse** part indices to **dense** part indices.

Not all parts in SMPL are selected, some are ignored.
The indices of selected parts are like: 2, 5, 8, 10, ..., which is the **sparse** part indices.
However, the GCN requires **dense** part indices from 0 to N like 0, 1, 2,..., N
Therefore, this module is provided to map the sparse to dense.
"""
import numpy as np

from .height import heights  # The elements in 'height' are unique, its key (sparse part index) is used to select parts.
from .edge import edges

# The indices of selected parts from all SMPL parts.
dense_indices = list(heights.keys())  # To select used parts from all parts: all_parts[dense_indices]
# Map sparse_part_idx (2, 5, 10, 17,...) into dense_part_idx (0, 1, 2, 3,...)
part_after_take = {sparse_part_idx: i for i, sparse_part_idx in enumerate(dense_indices)}

# Dense height values
heights_dense = {}  # height idx after '.take()'
for sparse_part_idx, height_value in heights.items():
    dense_part_idx = part_after_take[sparse_part_idx]
    heights_dense[dense_part_idx] = height_value

# Dense edges. array of shape (edges, 2)
edges_1d = edges.reshape((-1))
edges_dense = map(part_after_take.get, edges_1d.tolist())
edges_dense = list(edges_dense)
edges_dense = np.array(edges_dense)
edges_dense = edges_dense.reshape(edges.shape)
pass
