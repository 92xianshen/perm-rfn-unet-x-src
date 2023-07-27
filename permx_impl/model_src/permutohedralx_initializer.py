"""
+ Initializer of our permutohedral lattice x built with NumPy.
"""

import numpy as np
import tensorflow as tf

class PermutohedralXTFInitializer:
    def __init__(self, d: int) -> None:
        """
        Initialize this class.

        Args:
            d: the dimension of features, such as 5 for bilateral features, 2 for spatial features.

        Returns:
            None.
        """
        super().__init__()
        self.d = tf.constant(d, dtype=tf.int32) # size and dimension

        canonical = np.zeros((d + 1, d + 1), dtype=np.int32)  # (d + 1, d + 1)
        for i in range(d + 1):
            canonical[i, :d + 1 - i] = i
            canonical[i, d + 1 - i:] = i - (d + 1)
        self.canonical = tf.constant(canonical, dtype=tf.int32)  # [d + 1, d + 1]

        E = np.vstack(
            [
                np.ones((d,), dtype=np.float32),
                np.diag(-np.arange(d, dtype=np.float32) - 2)
                + np.triu(np.ones((d, d), dtype=np.float32)),
            ]
        )  # (d + 1, d)
        self.E = tf.constant(E, dtype=tf.float32)  # [d + 1, d]

        # Expected standard deviation of our filter (p.6 in [Adams et al. 2010])
        inv_std_dev = np.sqrt(2.0 / 3.0) * np.float32(d + 1)

        # Compute the diagonal part of E (p.5 in [Adams et al 2010])
        scale_factor = (
            1.0 / np.sqrt((np.arange(d) + 2) * (np.arange(d) + 1)) * inv_std_dev
        )  # (d, )
        self.scale_factor = tf.constant(scale_factor, dtype=tf.float32)  # [d, ]

        diff_valid = 1 - np.tril(np.ones((d + 1, d + 1), dtype=np.int32)) # [d + 1, d + 1]
        self.diff_valid = tf.constant(diff_valid, dtype=tf.int32) # [d + 1, d + 1]

        # Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
        self.alpha = tf.constant(1.0 / (1.0 + tf.pow(2.0, -tf.cast(d, dtype=tf.float32))), dtype=tf.float32)

        # Helper constant values (matrices).
        d_mat = np.ones((d + 1,), dtype=np.short) * d  # [d + 1, ]
        d_mat = np.diag(d_mat)  # [d + 1, d + 1]
        diagone = np.diag(np.ones(d + 1, dtype=np.short))  # [d + 1, d + 1]
        self.d_mat = tf.constant(d_mat, dtype=tf.int32)  # [d + 1, d + 1]
        self.diagone = tf.constant(diagone, dtype=tf.int32)  # [d + 1, d + 1]
