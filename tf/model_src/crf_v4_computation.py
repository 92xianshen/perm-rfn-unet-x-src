import tensorflow as tf

from .permutohedralx_v4_computation import PermutohedralXComputation

class CRFComputation(tf.Module):
    def __init__(self, d_bifeats: int=5, d_spfeats: int=2, theta_alpha: float=80.0, theta_beta: float=0.0625, theta_gamma: float=3.0, bilateral_compat: float=10.0, spatial_compat: float=3.0, num_iterations: int=10, name: str=None) -> None:
        super().__init__(name)
        self.d_bifeats, self.d_spfeats = d_bifeats, d_spfeats
        self.theta_alpha, self.theta_beta, self.theta_gamma = theta_alpha, theta_beta, theta_gamma
        self.num_iterations = num_iterations

        # ==> Initialize 
        self.spatial_compat = spatial_compat
        self.bilateral_compat = bilateral_compat
        self.compatibility = -1

        self.bilateral_lattice = PermutohedralXComputation(self.d_bifeats)
        self.spatial_lattice = PermutohedralXComputation(self.d_spfeats)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # of image, [H, W, C], float32
    ])
    def init_features(self, image: tf.Tensor) -> tf.Tensor:
        # - Create bilateral features.
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        ys, xs = tf.meshgrid(tf.range(height), tf.range(width), indexing="ij") # [h, w]
        ys_bifeats, xs_bifeats = tf.cast(ys, dtype=tf.float32) / self.theta_alpha, tf.cast(xs, dtype=tf.float32) / self.theta_alpha
        color_feats = image / self.theta_beta
        bilateral_feats = tf.concat([xs_bifeats[..., tf.newaxis], ys_bifeats[..., tf.newaxis], color_feats], axis=-1)
        bilateral_feats = tf.reshape(bilateral_feats, shape=[-1, self.d_bifeats])

        # - Create spatial features.
        ys_spfeats, xs_spfeats = tf.cast(ys, dtype=tf.float32) / self.theta_gamma, tf.cast(xs, dtype=tf.float32) / self.theta_gamma
        spatial_feats = tf.concat([xs_spfeats[..., tf.newaxis], ys_spfeats[..., tf.newaxis]], axis=-1)
        spatial_feats = tf.reshape(spatial_feats, shape=[-1, self.d_spfeats])

        return bilateral_feats, spatial_feats

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32), # of features, [N, d], float32
    ])
    def init_bilateral_lattice(self, bilateral_features: tf.Tensor) -> tf.Tensor:
        # - Initialize lattice
        coords_1d_uniq_bilateral, M_bilateral, os_bilateral, ws_bilateral, ns_bilateral = self.bilateral_lattice.init(bilateral_features)

        return coords_1d_uniq_bilateral, M_bilateral, os_bilateral, ws_bilateral, ns_bilateral
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32), # of features, [N, d], float32
    ])
    def init_spatial_lattice(self, spatial_features: tf.Tensor) -> tf.Tensor:
        # - Initialize lattice
        coords_1d_uniq_spatial, M_spatial, os_spatial, ws_spatial, ns_spatial = self.spatial_lattice.init(spatial_features)

        return coords_1d_uniq_spatial, M_spatial, os_spatial, ws_spatial, ns_spatial

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # of unary, [H, W, D], float32
        tf.TensorSpec(shape=[None, ], dtype=tf.int32), # of os_bilateral, [N x (d + 1), ], int32
        tf.TensorSpec(shape=[None, ], dtype=tf.float32), # of ws_bilateral, [N x (d + 1), ], float32
        tf.TensorSpec(shape=[2, None, None], dtype=tf.int32), # of blur_neighbors_bilateral, [2, M, (d + 1)], int32
        tf.TensorSpec(shape=[], dtype=tf.int32), # of M_bilateral, [], int32
        tf.TensorSpec(shape=[None, ], dtype=tf.int32), # of os_spatial, [N x (d + 1), ], int32
        tf.TensorSpec(shape=[None, ], dtype=tf.float32), # of ws_spatial, [N x (d + 1), ], float32
        tf.TensorSpec(shape=[2, None, None], dtype=tf.int32), # of blur_neighbors_spatial, [2, M, (d + 1)], int32
        tf.TensorSpec(shape=[], dtype=tf.int32), # of M_spatial, [], int32
    ])
    def mean_field_approximation(self, unary: tf.Tensor, os_bilateral: tf.Tensor, ws_bilateral: tf.Tensor, blur_neighbors_bilateral: tf.Tensor, M_bilateral: int, os_spatial: tf.Tensor, ws_spatial: tf.Tensor, blur_neighbors_spatial: tf.Tensor, M_spatial: int) -> tf.Tensor:
        unary_shape = tf.shape(unary) # [H, W, C] kept
        height, width, num_classes = unary_shape[0], unary_shape[1], unary_shape[2]
        n_feats = height * width
        # - Compute symmetric weights
        all_ones = tf.ones([n_feats, 1], dtype=tf.float32) # [N, 1]
        bilateral_norm_vals = self.bilateral_lattice.compute(all_ones, os_bilateral, ws_bilateral, blur_neighbors_bilateral, M_bilateral)
        bilateral_norm_vals = 1. / (bilateral_norm_vals ** .5 + 1e-20)
        spatial_norm_vals = self.spatial_lattice.compute(all_ones, os_spatial, ws_spatial, blur_neighbors_spatial, M_spatial)
        spatial_norm_vals = 1. / (spatial_norm_vals ** .5 + 1e-20)

        # - Initialize Q
        unary = tf.reshape(unary, [n_feats, num_classes]) # flatten, [N, C]
        Q = tf.nn.softmax(-unary, axis=-1) # [N, C]

        for i in range(self.num_iterations):
            tmp1 = -unary # [N, C]

            # - Symmetric normalization and bilateral message passing
            bilateral_out = self.bilateral_lattice.compute(Q * bilateral_norm_vals, os_bilateral, ws_bilateral, blur_neighbors_bilateral, M_bilateral) # [N, C]
            bilateral_out *= bilateral_norm_vals # [N, C]

            # - Symmetric normalization and spatial message passing
            spatial_out = self.spatial_lattice.compute(Q * spatial_norm_vals, os_spatial, ws_spatial, blur_neighbors_spatial, M_spatial) # [N, C]
            spatial_out *= spatial_norm_vals # [N, C]

            # - Message passing
            message_passing = self.bilateral_compat * bilateral_out + self.spatial_compat * spatial_out # [N, C]
            
            # - Compatibility transform
            pairwise = self.compatibility * message_passing # [N, C]

            # - Local update
            tmp1 -= pairwise # [N, C]

            # - Normalize
            Q = tf.nn.softmax(tmp1) # [N, C]

        return tf.reshape(Q, shape=unary_shape) # [H, W, C]