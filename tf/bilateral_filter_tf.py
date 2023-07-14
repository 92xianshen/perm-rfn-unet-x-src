# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
# from permutohedral_tf_v5 import Permutohedral as PermutohedralTF
# from permutohedral_tf_v2 import Permutohedral as PermutohedralTFV2
# from permutohedral_numpified import Permutohedral as PermutohedralNP
from permutohedralx import PermutohedralX as PermutohedralXTF
import matplotlib.pyplot as plt
import cv2
import time

im = Image.open('../../data/lenna.png').convert("RGB")
# im = Image.open('../../data/lena.small.jpg').convert("RGB")
im = np.array(im) / 255.

h, w, n_channels = im.shape

invSpatialStdev = 1. / 5.
invColorStdev = 1. / .25

# features = np.zeros((h, w, 5), dtype=np.float32)
# spatial_feat = np.mgrid[0:h, 0:w][::-1].transpose((1, 2, 0)) * invSpatialStdev
# color_feat = im * invColorStdev
# features[..., :2] = spatial_feat
# features[..., 2:] = color_feat
# features = features.reshape((-1, 5))

color_feat = tf.constant(im * invColorStdev, dtype=tf.float32)
ys, xs = tf.meshgrid(tf.range(h), tf.range(w), indexing="ij")
ys, xs = tf.cast(ys, dtype=tf.float32) * invSpatialStdev, tf.cast(xs, dtype=tf.float32) * invSpatialStdev
features = tf.concat([xs[..., tf.newaxis], ys[..., tf.newaxis], color_feat], axis=-1)
features = tf.reshape(features, shape=[-1, 5])
print(features.shape)

N, d = features.shape[0], features.shape[1]

# TF impl.
lattice = PermutohedralXTF(N, d)
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
# cv2.imshow('dst v2', dst_v2[..., ::-1])
# # cv2.imshow('im_filtered2', dst2[..., ::-1])
cv2.waitKey()

# # NumPy impl.
# lattice_np = PermutohedralNP(N, d)
# lattice_np.init(features)
# print('Lattice of NumPy initialized.')

# print(1 if np.allclose(lattice_np.rank_, lattice_tf.rank_.numpy()) else 0)
# for rnk_np, rnk_tf in zip(lattice_np.rank_, lattice_tf.rank_.numpy()):
#     if not np.allclose(rnk_np, rnk_tf):
#         print(rnk_np, rnk_tf)

# print(1 if np.allclose(lattice_np.barycentric_, lattice_tf.barycentric_.numpy()) else 0)
# for bc_np, bc_tf in zip(lattice_np.barycentric_, lattice_tf.barycentric_.numpy()):
#     if not np.allclose(bc_np, bc_tf):
#         print(bc_np, bc_tf)

# all_ones = np.ones((N, 1), dtype=np.float32)
# all_ones = lattice.compute(all_ones)
# all_ones = all_ones.numpy()
# all_ones = all_ones.reshape((h, w, 1))

# src = im.reshape((-1, n_channels))
# dst = lattice.compute(src)
# dst = dst.numpy()
# dst = dst.reshape((h, w, n_channels))
# dst = dst / all_ones
# dst = (dst - dst.min()) / (dst.max() - dst.min() + 1e-5)

# all_ones2 = np.ones((N, 1), dtype=np.float32)
# all_ones2 = lattice.np_compute(all_ones2)
# all_ones2 = all_ones2.reshape((h, w, 1))

# src2 = im.reshape((-1, n_channels))
# dst2 = lattice.np_compute(src2)
# dst2 = dst2.reshape((h, w, n_channels))
# dst2 = dst2 / all_ones2
# dst2 = (dst2 - dst2.min()) / (dst2.max() - dst2.min() + 1e-5)

# print(np.max(dst - dst2))

# cv2.imshow('im', im[..., ::-1])
# cv2.imshow('im_filtered', dst[..., ::-1])
# # cv2.imshow('im_filtered2', dst2[..., ::-1])
# cv2.waitKey()

# im_filtered = np.zeros_like(im)
# for ch in range(n_channels):
#     imch = im[..., ch:ch + 1].transpose((2, 0, 1)).reshape((1, -1))
#     imch_filtered = lattice.compute(imch)
#     imch_filtered = imch_filtered.reshape((1, h, w))[0]
#     imch_filtered = imch_filtered / all_ones
#     imch_filtered = (imch_filtered - imch_filtered.min()) / (imch_filtered.max() - imch_filtered.min())
#     im_filtered[..., ch] = imch_filtered

# cv2.imshow('im', im[..., ::-1])
# cv2.imshow('im_filtered', im_filtered[..., ::-1])
# cv2.waitKey()

# im_add = im.transpose((2, 0, 1)).reshape((n_channels, -1))
# im_add = np.vstack([im_add, np.ones((1, h * w), dtype=im.dtype)])
# print(im_add.shape)

# im_filtered = lattice.compute(im_add)
# im_filtered = (im_filtered[:3] / im_filtered[-1:])
# print(im_filtered.max(), im_filtered.min())
# im_filtered = im_filtered.reshape((n_channels, h, w)).transpose((1, 2, 0))
# plt.imshow(im_filtered / im_filtered.max())
# plt.show()

# # all_ones = np.ones((1, N), dtype=np.float32)
# # all_ones = lattice.compute(all_ones)
# # all_ones = all_ones.reshape((1, h, w))[0]

# # plt.imshow(all_ones / all_ones.max(), cmap='gray')
# # plt.show()