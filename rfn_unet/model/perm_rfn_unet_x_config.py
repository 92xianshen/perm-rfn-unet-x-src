import tensorflow as tf

class PermRfnUNetXConfig:
    """ CRF layer """
    def __init__(self, d_bifeats: int=5, d_spfeats: int=2, theta_alpha: float=80.0, theta_beta: float=0.0625, theta_gamma: float=3.0, bilateral_compat: float=10.0, spatial_compat: float=3.0, num_iterations: int=10) -> None:
        # - Initialize CRF parameters
        self.d_bifeats, self.d_spfeats = tf.constant(d_bifeats, dtype=tf.int32), tf.constant(d_spfeats, dtype=tf.int32) # d_bifeats, d_spfeats
        self.theta_alpha, self.theta_beta = tf.constant(theta_alpha, dtype=tf.float32), tf.constant(theta_beta, dtype=tf.float32) # of bilateral hyperparameters
        self.theta_gamma = tf.constant(theta_gamma, dtype=tf.float32) # of spatial hyperparameters

        self.bilateral_compat = tf.constant(bilateral_compat, dtype=tf.float32)
        self.spatial_compat = tf.constant(spatial_compat, dtype=tf.float32)
        self.compatibility = tf.constant(-1, dtype=tf.float32) # of message-passing parameters

        self.num_iterations = tf.constant(num_iterations, dtype=tf.int32)