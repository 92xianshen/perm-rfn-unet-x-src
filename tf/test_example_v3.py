import time

import numpy as np
import matplotlib.pyplot as plt

from util.example_util import image2unary
from model_src.crf_v3 import CRF
from model_src.crf_v3_config import CRFConfig

image_name = "../../data/examples/im2.png"
anno_name = "../../data/examples/anno2.png"
crf_computation_path = "./saved_model/crf_computation"

unary, image, n_labels = image2unary(image_name, anno_name)
h, w, c = image.shape

crf_config = CRFConfig(height=h, width=w, num_classes=n_labels, d_bifeats=c + 2, d_spfeats=2, theta_alpha=80.0, theta_beta=0.0625, theta_gamma=3.0, bilateral_compat=10.0, spatial_compat=3.0, num_iterations=10)
crf = CRF(crf_config=crf_config, crf_computation_path=crf_computation_path)

start = time.time()
Q = crf(unary.astype(np.float32), image.astype(np.float32)).numpy()
MAP = np.argmax(Q, axis=-1)
print("Time: ", time.time() - start)
plt.imshow(MAP)
plt.show()