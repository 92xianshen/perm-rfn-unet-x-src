"""
Class v2 of Permutohedral Refined UNet X, using all bands to create bilateral features. 
"""

import tensorflow as tf

from .perm_rfn_unet_x_config import PermRfnUNetXConfig
from .permutohedralx_initializer import PermutohedralXTFInitializer

class PermutohedralRefinedUNetX(tf.keras.Model):
    """
    Class of Permutohedral Refined UNet X
    """
    def __init__(self, model_config: PermRfnUNetXConfig=None, ugenerator_path: str=None, crf_computation_path: str=None) -> None:
        super().__init__()
        self.model_config = model_config

        self.unary_generator = tf.saved_model.load(ugenerator_path)
        self.crf_computation = tf.saved_model.load(crf_computation_path)

        self.bilateral_lattice_initializer = PermutohedralXTFInitializer(self.model_config.d_bifeats)
        self.spatial_lattice_initializer = PermutohedralXTFInitializer(self.model_config.d_spfeats)

    def call(self, image: tf.Tensor) -> tf.Tensor:
        """
        Predict and refine. 
        
        Args: 
            image: 7-band input, [h, w, c], float32.

        Returns:
            rfn: [h, w], 
        """
        unary = self.unary_generator(image[tf.newaxis, ...])[0] # [h, w, c] --> [h, w, n]
        ref = image # [h, w, c]

        # Create features
        bilateral_feats, spatial_feats = self.crf_computation.init_features(
            image=ref, 
            d_bifeats=self.model_config.d_bifeats, 
            d_spfeats=self.model_config.d_spfeats, 
            theta_alpha=self.model_config.theta_alpha, 
            theta_beta=self.model_config.theta_beta, 
            theta_gamma=self.model_config.theta_gamma, 
        )

        # Initialize bilateral and spatial lattices
        coords_1d_uniq_bilateral, M_bilateral, os_bilateral, ws_bilateral, ns_bilateral = self.crf_computation.init_lattice(
            features=bilateral_feats, 
            canonical=self.bilateral_lattice_initializer.canonical, 
            E=self.bilateral_lattice_initializer.E, 
            scale_factor=self.bilateral_lattice_initializer.scale_factor, 
            diff_valid=self.bilateral_lattice_initializer.diff_valid, 
            d_mat=self.bilateral_lattice_initializer.d_mat, 
            diagone=self.bilateral_lattice_initializer.diagone, 
        )
        
        coords_1d_uniq_spatial, M_spatial, os_spatial, ws_spatial, ns_spatial = self.crf_computation.init_lattice(
            features=spatial_feats, 
            canonical=self.spatial_lattice_initializer.canonical, 
            E=self.spatial_lattice_initializer.E, 
            scale_factor=self.spatial_lattice_initializer.scale_factor, 
            diff_valid=self.spatial_lattice_initializer.diff_valid, 
            d_mat=self.spatial_lattice_initializer.d_mat, 
            diagone=self.spatial_lattice_initializer.diagone, 
        )

        hash_table_bilateral = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(coords_1d_uniq_bilateral, tf.range(M_bilateral, dtype=tf.int32)), 
            default_value=-1)
        blur_neighbors_bilateral = hash_table_bilateral.lookup(ns_bilateral) + 1

        hash_table_spatial = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(coords_1d_uniq_spatial, tf.range(M_spatial, dtype=tf.int32)), 
            default_value=-1)
        blur_neighbors_spatial = hash_table_spatial.lookup(ns_spatial) + 1

        # Mean-field approximation
        MAP = self.crf_computation.mean_field_approximation(
            unary=unary, 
            d_bifeats=self.model_config.d_bifeats, 
            d_spfeats=self.model_config.d_spfeats, 
            alpha_bilateral=self.bilateral_lattice_initializer.alpha, 
            alpha_spatial=self.spatial_lattice_initializer.alpha, 
            os_bilateral=os_bilateral, 
            ws_bilateral=ws_bilateral, 
            blur_neighbors_bilateral=blur_neighbors_bilateral, 
            M_bilateral=M_bilateral, 
            os_spatial=os_spatial, 
            ws_spatial=ws_spatial, 
            blur_neighbors_spatial=blur_neighbors_spatial, 
            M_spatial=M_spatial, 
            bilateral_compat=self.model_config.bilateral_compat, 
            spatial_compat=self.model_config.spatial_compat, 
            compatibility=self.model_config.compatibility, 
            num_iterations=self.model_config.num_iterations, 
        )
        
        return MAP
        