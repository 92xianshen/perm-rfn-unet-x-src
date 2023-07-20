# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
# from permutohedralx import PermutohedralX as PermutohedralXTF
# from permutohedralx_v2 import PermutohedralX
from model_src.permutohedralx_v4 import PermutohedralX
import matplotlib.pyplot as plt
import cv2
import time

computation_path = "saved_model/permx_v4_computation"

# - Load data
im = Image.open('../../data/lenna.png').convert("RGB")
im = np.array(im) / 255.

# - Create bilateral features
h, w, n_channels = im.shape

invSpatialStdev = 1. / 5.
invColorStdev = 1. / .25

color_feat = tf.constant(im * invColorStdev, dtype=tf.float32)
ys, xs = tf.meshgrid(tf.range(h), tf.range(w), indexing="ij")
ys, xs = tf.cast(ys, dtype=tf.float32) * invSpatialStdev, tf.cast(xs, dtype=tf.float32) * invSpatialStdev
features = tf.concat([xs[..., tf.newaxis], ys[..., tf.newaxis], color_feat], axis=-1)
features = tf.reshape(features, shape=[-1, 5])
print(features.shape)

N, d = features.shape[0], features.shape[1]

# Initialize class
lattice = PermutohedralX(d, computation_path)
print("Start...")
start = time.time()
lattice.init(features)
print('Lattice of TF initialized.')

all_ones = np.ones((N, 1), dtype=np.float32)
norms = lattice.compute(all_ones)
norms = norms.numpy().reshape((h, w, 1))

src = im.reshape((-1, n_channels))
dst = lattice.compute(src.astype(np.float32))
dst = dst.numpy().reshape((h, w, n_channels))
dst = dst / norms
dst = (dst - dst.min()) / (dst.max() - dst.min() + 1e-5)
print("Time:", time.time() - start)

cv2.imshow('im', im[..., ::-1])
cv2.imshow('dst', dst[..., ::-1])
cv2.waitKey()
