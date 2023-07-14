"""
Fail.
+ 2023.07.14: Convert permutohedralx_v2 to SavedModel.
"""

import tensorflow as tf

from permutohedralx_v2 import PermutohedralX

N, d = 512 * 512, 5
export_dir = "permx_v2/"

lattice = PermutohedralX(N, d)

tf.saved_model.save(lattice, export_dir=export_dir)
print("Write to {}.".format(export_dir))