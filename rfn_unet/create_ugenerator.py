import numpy as np
import tensorflow as tf

from model.unet import UNet

input_channels = 7
pretrained_path = "checkpoint"
gt_prob = .7
export_dir = "parameter/saved_model/unary_generator"

inputs = tf.keras.Input(
    shape=[None, None, input_channels], 
    name="inputs", 
)

backbone = UNet()
checkpoint = tf.train.Checkpoint(model=backbone)
checkpoint.restore(tf.train.latest_checkpoint(pretrained_path)).expect_partial()
print("Checkpoint restored from {}.".format(tf.train.latest_checkpoint(pretrained_path)))

# Lazy load
backbone(np.ones((1, 512, 512, 7)))

logits = backbone(inputs, training=False)
probs = tf.nn.softmax(logits, name="logits2probs")
unary = -tf.math.log(probs * gt_prob, name="probs2unary")

unary_generator = tf.keras.Model(inputs=inputs, outputs=unary)

tf.saved_model.save(unary_generator, export_dir=export_dir)
print("Write to {}.".format(export_dir))