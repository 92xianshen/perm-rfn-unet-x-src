"""
Class of permutohedral lattice x.
"""

import tensorflow as tf

from .permutohedralx_initializer import PermutohedralXTFInitializer
from .permutohedralx_helper import PermutohedralXHelpher

class PermutohedralX(tf.keras.Model):
    def __init__(self, d: int, computation_path: str) -> None:
        self.initializer = PermutohedralXTFInitializer(d)
        self.helper = PermutohedralXHelpher()
        self.computation = tf.saved_model.load(computation_path)

    def init(self, features: tf.Tensor) -> None:
        self.helper.coords_1d_uniq, self.helper.M, self.helper.os, self.helper.ws, self.helper.ns = self.computation.init(
            features=features, 
            canonical=self.initializer.canonical, 
            E=self.initializer.E, 
            scale_factor=self.initializer.scale_factor, 
            diff_valid=self.initializer.diff_valid, 
            d_mat=self.initializer.d_mat, 
            diagone=self.initializer.diagone, 
        )

        hash_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.helper.coords_1d_uniq, tf.range(self.helper.M, dtype=tf.int32)), 
            default_value=-1)
        self.helper.blur_neighbors = hash_table.lookup(self.helper.ns) + 1

    def compute(self, inp: tf.Tensor) -> tf.Tensor:
        out = self.computation.compute(
            inp=inp,
            d=self.initializer.d, 
            alpha=self.initializer.alpha, 
            os=self.helper.os,
            ws=self.helper.ws,
            blur_neighbors=self.helper.blur_neighbors, 
            M=self.helper.M, 
        )
        return out