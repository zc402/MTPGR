# Parts: 2, 5, 8, 10...
# Parts after take: 0, 1, 2, ...
# This model maps {Parts: Parts after take}

from taper.kinematic import heights

# The part numbers in 'height' are unique.

part_indices = list(heights.keys())  # Used as: all_features[part_take]
p2pat = {e: v for e, v in enumerate(part_indices)}  # 'part idx' to 'part idx after take'