"""
Fail.
+ 2023.07.14: Convert permutohedralx_v2 to SavedModel, for bilateral filtering w.r.t. `lenna.jpg`.
+ 2023.07.17: Convert to SavedModel, for refinement test w.r.t. `examples/`.
"""

import tensorflow as tf

from model_src.permutohedralx_v2 import PermutohedralX

# # + 2023.07.14: for bilateral filtering w.r.t. `lenna.jpg`.
# N, d = 512 * 512, 5
# export_dir = "permx_v2/"

# lattice = PermutohedralX(N, d)

# tf.saved_model.save(lattice, export_dir=export_dir)
# print("Write to {}.".format(export_dir))



# + 2023.07.17: for `examples/`
# N, d = 320 * 240, 5 # for bilateral lattice
# export_dir = "saved_model/bilateral_lattice"
N, d = 320 * 240, 2 # for spatial lattice
export_dir = "saved_model/spatial_lattice"

lattice = PermutohedralX(N, d)

tf.saved_model.save(lattice, export_dir=export_dir)
print("Write to {}.".format(export_dir))