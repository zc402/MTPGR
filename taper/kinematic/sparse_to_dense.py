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
part_after_take = {e: v for e, v in enumerate(dense_indices)}  # A dict mapping 'part idx' to 'part idx after take'

# Dense height values
heights_dense = {}  # height idx after '.take()'
for k, v in heights.items():
    k_at = part_after_take[k]
    heights_dense[k_at] = v

# Dense edges. array of shape (edges, 2)
edges_dense = np.array(map(part_after_take.get, edges))
