"""
Class of permutohedral lattice x.
"""

import tensorflow as tf

from .permutohedralx_v3_initializer import PermutohedralXTFInitializer
from .permutohedralx_v3_helper import PermutohedralXHelpher

class PermutohedralX(tf.keras.Model):
    def __init__(self, N: int, d: int, computation_path: str) -> None:
        self.initializer = PermutohedralXTFInitializer(N, d)
        self.helper = PermutohedralXHelpher()
        self.computation = tf.saved_model.load(computation_path)

    def init(self, features: tf.Tensor) -> None:
        self.helper.coords_1d_uniq, self.helper.M, self.helper.os, self.helper.ws, self.helper.ns = self.computation.init(features, self.initializer.N, self.initializer.d, self.initializer.canonical, self.initializer.E, self.initializer.scale_factor, self.initializer.diff_valid, self.initializer.d_mat, self.initializer.diagone)

        hash_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(self.helper.coords_1d_uniq, tf.range(self.helper.M, dtype=tf.int32)), default_value=-1)
        self.helper.blur_neighbors = hash_table.lookup(self.helper.ns) + 1

    def compute(self, inp: tf.Tensor) -> tf.Tensor:
        out = self.computation.compute(inp, self.initializer.N, self.initializer.d, self.initializer.alpha, self.helper.os, self.helper.ws, self.helper.blur_neighbors, self.helper.M)
        return out