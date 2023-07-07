import numpy as np
import matplotlib.pyplot as plt

from example_util import image2unary
from crf_layer import CRFLayer

image_name = "../../data/examples/im2.png"
anno_name = "../../data/examples/anno2.png"

unary, image, n_labels = image2unary(image_name, anno_name)
h, w, c = image.shape

crf = CRFLayer(num_classes=n_labels, height=h, width=w, d_bifeats=c + 2, d_spfeats=2, theta_alpha=80., theta_beta=.0625, theta_gamma=3., bilateral_compat=10.,  spatial_compat=3., num_iterations=10)

Q = crf(unary.astype(np.float32), image.astype(np.float32)).numpy()
MAP = np.argmax(Q, axis=-1)
plt.imshow(MAP)
plt.show()