# -*- coding: utf-8 -*-

"""
CRF model.
channel-last input required.
"""

import tensorflow as tf

from permutohedralx import PermutohedralX

class CRFLayer(tf.keras.layers.Layer):
    """ CRF layer """
    def __init__(self, num_classes: int, height: int, width: int, d_bifeats: int, d_spfeats: int, theta_alpha: float, theta_beta: float, theta_gamma: float, bilateral_compat: float, spatial_compat: float, num_iterations: int) -> None:
        super(CRFLayer, self).__init__()
        # ==> Initialize parameters
        self.num_classes = num_classes # C
        self.height, self.width = height, width # H, W
        self.n_feats = self.height * self.width # N
        self.d_bifeats, self.d_spfeats = d_bifeats, d_spfeats # d_bifeats, d_spfeats
        self.theta_alpha, self.theta_beta = theta_alpha, theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations

        # ==> Initialize 
        self.spatial_compat = spatial_compat
        self.bilateral_compat = bilateral_compat
        self.compatibility = -1

        # ==> Initialize permutohedral lattice
        self.bilateral_lattice = PermutohedralX(height * width, d_bifeats)
        self.spatial_lattice = PermutohedralX(height * width, d_spfeats)

    def call(self, unary: tf.Tensor, image: tf.Tensor) -> tf.Tensor:
        """ The order of parameters: I, p """
        assert len(image.shape) == 3 and len(unary.shape) == 3

        # Get dimensions of features.
        # n_feats = self.height * self.width # N
        # d_bifeats = tf.shape(image)[-1] + 2 # channel-last, dim. of color plus spatial dim.
        # d_spfeats = 2 # 2D, spatial dim.

        # Create bilateral features.
        ys, xs = tf.meshgrid(tf.range(self.height), tf.range(self.width), indexing="ij") # [h, w]
        ys_bifeats, xs_bifeats = tf.cast(ys, dtype=tf.float32) / self.theta_alpha, tf.cast(xs, dtype=tf.float32) / self.theta_alpha
        color_feats = tf.constant(image / self.theta_beta, dtype=tf.float32)
        bilateral_feats = tf.concat([xs_bifeats[..., tf.newaxis], ys_bifeats[..., tf.newaxis], color_feats], axis=-1)
        bilateral_feats = tf.reshape(bilateral_feats, shape=[-1, self.d_bifeats])

        # Create spatial features.
        ys_spfeats, xs_spfeats = tf.cast(ys, dtype=tf.float32) / self.theta_gamma, tf.cast(xs, dtype=tf.float32) / self.theta_gamma
        spatial_feats = tf.concat([xs_spfeats[..., tf.newaxis], ys_spfeats[..., tf.newaxis]], axis=-1)
        spatial_feats = tf.reshape(spatial_feats, shape=[-1, self.d_spfeats])

        # Initialize lattice
        self.bilateral_lattice.init(bilateral_feats)
        self.spatial_lattice.init(spatial_feats)
        
        # Compute symmetric weights
        all_ones = tf.ones([self.n_feats, 1], dtype=tf.float32) # [N, 1]
        bilateral_norm_vals = self.bilateral_lattice.compute(all_ones)
        bilateral_norm_vals = 1. / (bilateral_norm_vals ** .5 + 1e-20)
        spatial_norm_vals = self.spatial_lattice.compute(all_ones)
        spatial_norm_vals = 1. / (spatial_norm_vals ** .5 + 1e-20)

        # Initialize Q
        unary_shape = tf.shape(unary) # [H, W, C] kept
        unary = tf.reshape(unary, [self.n_feats, self.num_classes]) # flatten, [N, C]
        Q = tf.nn.softmax(-unary, axis=-1) # [N, C]

        for i in range(self.num_iterations):
            tf.print("Iter. {}...".format(i + 1))
            tmp1 = -unary # [N, C]

            # Symmetric normalization and bilateral message passing
            bilateral_out = self.bilateral_lattice.compute(Q * bilateral_norm_vals) # [N, C]
            bilateral_out *= bilateral_norm_vals # [N, C]

            # Symmetric normalization and spatial message passing
            spatial_out = self.spatial_lattice.compute(Q * spatial_norm_vals) # [N, C]
            spatial_out *= spatial_norm_vals # [N, C]

            # Message passing
            message_passing = self.bilateral_compat * bilateral_out + self.spatial_compat * spatial_out # [N, C]
            
            # Compatibility transform
            pairwise = self.compatibility * message_passing # [N, C]

            # Local update
            tmp1 -= pairwise # [N, C]

            # Normalize
            Q = tf.nn.softmax(tmp1) # [N, C]

        return tf.reshape(Q, shape=unary_shape) # [H, W, C]
