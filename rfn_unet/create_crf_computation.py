import tensorflow as tf

from model.crf_computation import CRFComputation

export_dir = "parameter/saved_model/crf_computation"

crf_computation = CRFComputation()

tf.saved_model.save(crf_computation, export_dir=export_dir)
print("Write to {}.".format(export_dir))