import tensorflow as tf

from .permutohedralx_v4_computation import PermutohedralXComputation

class CRFComputation(tf.Module):
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # of image, [H, W, C], float32
        tf.TensorSpec(shape=[], dtype=tf.int32), # of d_bifeats, [], int32
        tf.TensorSpec(shape=[], dtype=tf.int32), # of d_spfeats, [], int32
        tf.TensorSpec(shape=[], dtype=tf.float32), # of theta_alpha, [], float32
        tf.TensorSpec(shape=[], dtype=tf.float32), # of theta_beta, [], float32
        tf.TensorSpec(shape=[], dtype=tf.float32), # of theta_gamma, [], float32
    ])
    def init_features(
        self, 
        image: tf.Tensor, 
        d_bifeats: int, 
        d_spfeats: int, 
        theta_alpha: float, 
        theta_beta: float, 
        theta_gamma: float
    ) -> tf.Tensor:
        # - Create bilateral features.
        height, width = tf.shape(image)[0], tf.shape(image)[1]

        ys, xs = tf.meshgrid(tf.range(height), tf.range(width), indexing="ij") # [h, w]
        ys_bifeats, xs_bifeats = tf.cast(ys, dtype=tf.float32) / theta_alpha, tf.cast(xs, dtype=tf.float32) / theta_alpha
        color_feats = image / theta_beta
        bilateral_feats = tf.concat([xs_bifeats[..., tf.newaxis], ys_bifeats[..., tf.newaxis], color_feats], axis=-1)
        bilateral_feats = tf.reshape(bilateral_feats, shape=[-1, d_bifeats])

        # - Create spatial features.
        ys_spfeats, xs_spfeats = tf.cast(ys, dtype=tf.float32) / theta_gamma, tf.cast(xs, dtype=tf.float32) / theta_gamma
        spatial_feats = tf.concat([xs_spfeats[..., tf.newaxis], ys_spfeats[..., tf.newaxis]], axis=-1)
        spatial_feats = tf.reshape(spatial_feats, shape=[-1, d_spfeats])

        return bilateral_feats, spatial_feats

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32), # of features, [N, d], float32
        tf.TensorSpec(shape=[None, None], dtype=tf.int32), # of canonical, [d + 1, d + 1], int32
        tf.TensorSpec(shape=[None, None], dtype=tf.float32), # of E, [d + 1, d], float32
        tf.TensorSpec(shape=[None, ], dtype=tf.float32), # of scale_factor, [d, ], float32
        tf.TensorSpec(shape=[None, None], dtype=tf.int32), # of diff_valid, [d + 1, d + 1], int32
        tf.TensorSpec(shape=[None, None], dtype=tf.int32), # of d_mat, [d + 1, d + 1], int32
        tf.TensorSpec(shape=[None, None], dtype=tf.int32), ]) # of diagone, [d + 1, d + 1], int32
    def init_lattice(
        self, 
        features: tf.Tensor, 
        canonical: tf.Tensor, 
        E: tf.Tensor, 
        scale_factor: tf.Tensor, 
        diff_valid: tf.Tensor, 
        d_mat: tf.Tensor, 
        diagone: tf.Tensor
    ) -> tf.Tensor:
        coords_1d_uniq, M, os, ws, ns = PermutohedralXComputation().init(features, canonical, E, scale_factor, diff_valid, d_mat, diagone)
        return coords_1d_uniq, M, os, ws, ns

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # of unary, [H, W, D], float32
        tf.TensorSpec(shape=[], dtype=tf.int32), # of d_bifeats, [], int32
        tf.TensorSpec(shape=[], dtype=tf.int32), # of d_spfeats, [], int32
        tf.TensorSpec(shape=[], dtype=tf.float32), # of alpha_bilateral, [], float
        tf.TensorSpec(shape=[], dtype=tf.float32), # of alpha_spatial, [], float
        tf.TensorSpec(shape=[None, ], dtype=tf.int32), # of os_bilateral, [N x (d + 1), ], int32
        tf.TensorSpec(shape=[None, ], dtype=tf.float32), # of ws_bilateral, [N x (d + 1), ], float32
        tf.TensorSpec(shape=[2, None, None], dtype=tf.int32), # of blur_neighbors_bilateral, [2, M, (d + 1)], int32
        tf.TensorSpec(shape=[], dtype=tf.int32), # of M_bilateral, [], int32
        tf.TensorSpec(shape=[None, ], dtype=tf.int32), # of os_spatial, [N x (d + 1), ], int32
        tf.TensorSpec(shape=[None, ], dtype=tf.float32), # of ws_spatial, [N x (d + 1), ], float32
        tf.TensorSpec(shape=[2, None, None], dtype=tf.int32), # of blur_neighbors_spatial, [2, M, (d + 1)], int32
        tf.TensorSpec(shape=[], dtype=tf.int32), # of M_spatial, [], int32
        tf.TensorSpec(shape=[], dtype=tf.float32), # of bilateral_compat, [], float32
        tf.TensorSpec(shape=[], dtype=tf.float32), # of spatial_compat, [], float32
        tf.TensorSpec(shape=[], dtype=tf.float32), # of compatibility, [], float32
        tf.TensorSpec(shape=[], dtype=tf.int32), # of num_iterations, [], int32
    ])
    def mean_field_approximation(
        self, 
        unary: tf.Tensor, 
        d_bifeats: int, 
        d_spfeats: int, 
        alpha_bilateral: float, 
        alpha_spatial: float, 
        os_bilateral: tf.Tensor, 
        ws_bilateral: tf.Tensor, 
        blur_neighbors_bilateral: tf.Tensor, 
        M_bilateral: int, 
        os_spatial: tf.Tensor, 
        ws_spatial: tf.Tensor, 
        blur_neighbors_spatial: tf.Tensor, 
        M_spatial: int, 
        bilateral_compat: float, 
        spatial_compat: float, 
        compatibility: float, 
        num_iterations: int
    ) -> tf.Tensor:
        unary_shape = tf.shape(unary) # [H, W, C] kept
        height, width, num_classes = unary_shape[0], unary_shape[1], unary_shape[2]
        n_feats = height * width
        # - Compute symmetric weights
        all_ones = tf.ones([n_feats, 1], dtype=tf.float32) # [N, 1]
        bilateral_norm_vals = PermutohedralXComputation().compute(all_ones, d_bifeats, alpha_bilateral, os_bilateral, ws_bilateral, blur_neighbors_bilateral, M_bilateral)
        bilateral_norm_vals = 1. / (bilateral_norm_vals ** .5 + 1e-20)
        spatial_norm_vals = PermutohedralXComputation().compute(all_ones, d_spfeats, alpha_spatial, os_spatial, ws_spatial, blur_neighbors_spatial, M_spatial)
        spatial_norm_vals = 1. / (spatial_norm_vals ** .5 + 1e-20)

        # - Initialize Q
        unary = tf.reshape(unary, [n_feats, num_classes]) # flatten, [N, C]
        Q = tf.nn.softmax(-unary, axis=-1) # [N, C]

        for i in range(num_iterations):
            tmp1 = -unary # [N, C]

            # - Symmetric normalization and bilateral message passing
            bilateral_out = PermutohedralXComputation().compute(Q * bilateral_norm_vals, d_bifeats, alpha_bilateral, os_bilateral, ws_bilateral, blur_neighbors_bilateral, M_bilateral) # [N, C]
            bilateral_out *= bilateral_norm_vals # [N, C]

            # - Symmetric normalization and spatial message passing
            spatial_out = PermutohedralXComputation().compute(Q * spatial_norm_vals, d_spfeats, alpha_spatial, os_spatial, ws_spatial, blur_neighbors_spatial, M_spatial) # [N, C]
            spatial_out *= spatial_norm_vals # [N, C]

            # - Message passing
            message_passing = bilateral_compat * bilateral_out + spatial_compat * spatial_out # [N, C]
            
            # - Compatibility transform
            pairwise = compatibility * message_passing # [N, C]

            # - Local update
            tmp1 -= pairwise # [N, C]

            # - Normalize
            Q = tf.nn.softmax(tmp1) # [N, C]

        return tf.reshape(Q, shape=unary_shape) # [H, W, C]