import numpy as np
import tensorflow as tf

h, w = 240, 320

spatial_feat_np = np.mgrid[0:h, 0:w][::-1].transpose((1, 2, 0))
# ys_tf, xs_tf = tf.meshgrid(tf.range(h), tf.range(w), indexing="ij")

print(spatial_feat_np.shape)