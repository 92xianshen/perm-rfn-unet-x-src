class CRFConfig:
    """ CRF layer """
    def __init__(self, height: int, width: int, num_classes: int, d_bifeats: int=5, d_spfeats: int=2, theta_alpha: float=80.0, theta_beta: float=0.0625, theta_gamma: float=3.0, bilateral_compat: float=10.0, spatial_compat: float=3.0, num_iterations: int=10) -> None:
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
