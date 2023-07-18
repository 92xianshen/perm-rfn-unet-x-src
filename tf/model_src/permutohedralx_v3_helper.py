"""
A helper class for our permutohedral lattice x. 
"""

class PermutohedralXHelpher:
    def __init__(self) -> None:
        self.coords_1d_uniq = None # [M, ], int32
        self.os = None # [N x (d + 1), ], int32
        self.ws = None # [N x (d + 1), ], float32
        self.ns = None # [2, M, d + 1], int32
        self.blur_neighbors = None # [2, M, d + 1], int32
        self.M = None # [], int32
