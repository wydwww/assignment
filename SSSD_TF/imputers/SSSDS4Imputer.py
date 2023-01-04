import tensorflow as tf
import tensorflow_addons as tfa
import math
from ..utils.util import calc_diffusion_step_embedding
from ..imputers.S4Model import S4Layer

def swish(x):
    return x * tf.keras.activations.sigmoid(x)

class Conv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        # yiding: Skip WeightNormalization because it requires activation.
        self.conv = tf.keras.layers.Conv1D(filters=out_channels, kernel_size=kernel_size, padding="causal", data_format="channels_first", kernel_initializer="he_normal", bias_initializer="he_normal")

    def call(self, x):
        out = self.conv(x)
        return out

class ZeroConv1d(tf.keras.layers.Layer):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = tf.keras.layers.Conv1D(filters=out_channel, kernel_size=1, padding="same", data_format="channels_first", kernel_initializer="zeros", bias_initializer="zeros")

    def call(self, x):
        out = self.conv(x)
        return out