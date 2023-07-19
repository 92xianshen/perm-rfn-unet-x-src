import tensorflow as tf

from .crf_v3_config import CRFConfig
from .permutohedralx_v3_initializer import PermutohedralXTFInitializer

class CRF(tf.keras.Model):
    """
    CRF model
    """
    def __init__(self, crf_config: CRFConfig, crf_computation_path: str) -> None:
        super(CRF, self).__init__()
        self.crf_config = crf_config
        self.crf_computation = tf.saved_model.load(crf_computation_path)
        self.bilateral_permx_initializer = PermutohedralXTFInitializer(N=self.crf_config.n_feats, d=self.crf_config.d_bifeats)
        self.spatial_permx_initializer = PermutohedralXTFInitializer(self.crf_config.n_feats, d=self.crf_config.d_spfeats)

    def call(self, unary: tf.Tensor, image: tf.Tensor) -> tf.Tensor:
        assert len(unary.shape) == 3 and len(image.shape) == 3

        # Create features
        bilateral_feats, spatial_feats = self.crf_computation.init_features(
            image=image, 
            height=self.crf_config.height, 
            width=self.crf_config.width, 
            d_bifeats=self.crf_config.d_bifeats, 
            d_spfeats=self.crf_config.d_spfeats, 
            theta_alpha=self.crf_config.theta_alpha, 
            theta_beta=self.crf_config.theta_beta, 
            theta_gamma=self.crf_config.theta_gamma)
        
        # Initialize lattice
        coords_1d_uniq_bilateral, M_bilateral, os_bilateral, ws_bilateral, ns_bilateral = self.crf_computation.init_lattice(
            features=bilateral_feats, 
            N=self.crf_config.n_feats, 
            d=self.crf_config.d_bifeats, 
            canonical=self.bilateral_permx_initializer.canonical, 
            E=self.bilateral_permx_initializer.E, 
            scale_factor=self.bilateral_permx_initializer.scale_factor, 
            diff_valid=self.bilateral_permx_initializer.diff_valid, 
            d_mat=self.bilateral_permx_initializer.d_mat, 
            diagone=self.bilateral_permx_initializer.diagone)
        
        coords_1d_uniq_spatial, M_spatial, os_spatial, ws_spatial, ns_spatial = self.crf_computation.init_lattice(
            features=spatial_feats, 
            N=self.crf_config.n_feats, 
            d=self.crf_config.d_spfeats, 
            canonical=self.spatial_permx_initializer.canonical, 
            E=self.spatial_permx_initializer.E, 
            scale_factor=self.spatial_permx_initializer.scale_factor, 
            diff_valid=self.spatial_permx_initializer.diff_valid, 
            d_mat=self.spatial_permx_initializer.d_mat, 
            diagone=self.spatial_permx_initializer.diagone)
        
        hash_table_bilateral = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(coords_1d_uniq_bilateral, tf.range(M_bilateral, dtype=tf.int32)), 
            default_value=-1)
        blur_neighbors_bilateral = hash_table_bilateral.lookup(ns_bilateral) + 1

        hash_table_spatial = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(coords_1d_uniq_spatial, tf.range(M_spatial, dtype=tf.int32)), 
            default_value=-1)
        blur_neighbors_spatial = hash_table_spatial.lookup(ns_spatial) + 1

        # Mean-field approximation
        Q = self.crf_computation.mean_field_approximation(
            unary=unary, 
            n_feats=self.crf_config.n_feats, 
            num_classes=self.crf_config.num_classes, 
            d_bifeats=self.crf_config.d_bifeats, 
            d_spfeats=self.crf_config.d_spfeats, 
            alpha_bilateral=self.bilateral_permx_initializer.alpha, 
            alpha_spatial=self.spatial_permx_initializer.alpha, 
            os_bilateral=os_bilateral, 
            ws_bilateral=ws_bilateral, 
            blur_neighbors_bilateral=blur_neighbors_bilateral, 
            M_bilateral=M_bilateral, 
            os_spatial=os_spatial, 
            ws_spatial=ws_spatial, 
            blur_neighbors_spatial=blur_neighbors_spatial, 
            M_spatial=M_spatial, 
            bilateral_compat=self.crf_config.bilateral_compat, 
            spatial_compat=self.crf_config.spatial_compat, 
            compatibility=self.crf_config.compatibility, 
            num_iterations=self.crf_config.num_iterations)
        
        return Q