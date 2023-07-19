import tensorflow as tf

from .crf_v4_config import CRFConfig

class CRF(tf.keras.Model):
    """
    CRF model
    """
    def __init__(self, crf_computation_path: str) -> None:
        super(CRF, self).__init__()
        self.crf_computation = tf.saved_model.load(crf_computation_path)

    def call(self, unary: tf.Tensor, image: tf.Tensor) -> tf.Tensor:
        assert len(unary.shape) == 3 and len(image.shape) == 3

        # Create features
        bilateral_feats, spatial_feats = self.crf_computation.init_features(
            image=image, 
        )
        
        # Initialize lattice
        coords_1d_uniq_bilateral, M_bilateral, os_bilateral, ws_bilateral, ns_bilateral = self.crf_computation.init_bilateral_lattice(
            bilateral_features=bilateral_feats, 
        )
        
        coords_1d_uniq_spatial, M_spatial, os_spatial, ws_spatial, ns_spatial = self.crf_computation.init_spatial_lattice(
            spatial_features=spatial_feats, 
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
        Q = self.crf_computation.mean_field_approximation(
            unary=unary, 
            os_bilateral=os_bilateral, 
            ws_bilateral=ws_bilateral, 
            blur_neighbors_bilateral=blur_neighbors_bilateral, 
            M_bilateral=M_bilateral, 
            os_spatial=os_spatial, 
            ws_spatial=ws_spatial, 
            blur_neighbors_spatial=blur_neighbors_spatial, 
            M_spatial=M_spatial, 
        )
        
        return Q