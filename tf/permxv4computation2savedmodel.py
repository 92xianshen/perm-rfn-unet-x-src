"""
Fail.
+ 2023.07.14: Convert permutohedralx_v2 to SavedModel, for bilateral filtering w.r.t. `lenna.jpg`.
+ 2023.07.17: Convert to SavedModel, for refinement test w.r.t. `examples/`.
+ 2023.07.18: Convert permutohedralx_v3_computation to SavedModel. 
"""

import tensorflow as tf

from model_src.permutohedralx_v4_computation import PermutohedralXComputation

export_dir = "saved_model/permxv4_computation"

lattice = PermutohedralXComputation(5)

tf.saved_model.save(lattice, export_dir=export_dir)
print("Write to {}.".format(export_dir))