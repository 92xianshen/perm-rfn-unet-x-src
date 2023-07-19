"""
Class of permutohedral lattice x.
"""

import tensorflow as tf

from .permutohedralx_v4_helper import PermutohedralXHelpher

class PermutohedralX(tf.keras.Model):
    def __init__(self, computation_path: str) -> None:
        self.helper = PermutohedralXHelpher()
        self.computation = tf.saved_model.load(computation_path)

    def init(self, features: tf.Tensor) -> None:
        self.helper.coords_1d_uniq, self.helper.M, self.helper.os, self.helper.ws, self.helper.ns = self.computation.init(features)

        hash_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.helper.coords_1d_uniq, tf.range(self.helper.M, dtype=tf.int32)), 
            default_value=-1)
        self.helper.blur_neighbors = hash_table.lookup(self.helper.ns) + 1

    def compute(self, inp: tf.Tensor) -> tf.Tensor:
        out = self.computation.compute(inp, self.helper.os, self.helper.ws, self.helper.blur_neighbors, self.helper.M)
        return out