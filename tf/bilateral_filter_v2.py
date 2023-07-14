# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
# from permutohedralx import PermutohedralX as PermutohedralXTF
# from permutohedralx_v2 import PermutohedralX
import matplotlib.pyplot as plt
import cv2
import time

model_path = "permx_v2"

im = Image.open('../../data/lenna.png').convert("RGB")
im = np.array(im) / 255.

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

# SavedModel
lattice = tf.saved_model.load(model_path)
# lattice = PermutohedralX(N, d)
print("Start...")
start = time.time()
coords_1d_uniq, M, os, ws, ns = lattice.init(features)
hash_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(coords_1d_uniq, tf.range(M, dtype=tf.int32)), default_value=-1)
blur_neighbors = hash_table.lookup(ns) + 1
print('Lattice of TF initialized.')

all_ones = np.ones((N, 1), dtype=np.float32)
norms = lattice.compute(all_ones, os, ws, blur_neighbors, M)
norms = norms.numpy().reshape((h, w, 1))

src = im.reshape((-1, n_channels))
dst = lattice.compute(src.astype(np.float32), os, ws, blur_neighbors, M)
dst = dst.numpy().reshape((h, w, n_channels))
dst = dst / norms
dst = (dst - dst.min()) / (dst.max() - dst.min() + 1e-5)
print("Time:", time.time() - start)

cv2.imshow('im', im[..., ::-1])
cv2.imshow('dst', dst[..., ::-1])
# cv2.imshow('dst v2', dst_v2[..., ::-1])
# # cv2.imshow('im_filtered2', dst2[..., ::-1])
cv2.waitKey()
