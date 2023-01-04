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

class Residual_block(tf.keras.Model):
    def __init__(self, res_channels, skip_channels, 
                 diffusion_step_embed_dim_out, in_channels,
                s4_lmax,
                s4_d_state,
                s4_dropout,
                s4_bidirectional,
                s4_layernorm):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels


        self.fc_t = tf.keras.layers.Dense(units=self.res_channels)
        self.S41 = S4Layer(features=2*self.res_channels, 
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm)
 
        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.S42 = S4Layer(features=2*self.res_channels, 
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm)
        
        self.cond_conv = Conv(2*in_channels, 2*self.res_channels, kernel_size=1)  

        self.res_conv = tf.keras.layers.Conv1D(filters=res_channels, kernel_size=1, data_format="channels_first", kernel_initializer="he_normal", bias_initializer="he_normal")
        # yiding: Skip weight normalization because it requires activation.
        
        self.skip_conv = tf.keras.layers.Conv1D(filters=skip_channels, kernel_size=1, data_format="channels_first", kernel_initializer="he_normal", bias_initializer="he_normal")
        # yiding: Skip weight normalization because it requires activation.

    def call(self, input_data):
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels                      
                 
        part_t = self.fc_t(diffusion_step_embed)
        part_t = tf.reshape(part_t, [B, self.res_channels, 1])  
        h = h + part_t
        
        h = self.conv_layer(h)
        h = tf.transpose(self.S41(tf.transpose(h, [2, 0, 1])), [1, 2, 0])
        
        assert cond is not None
        cond = self.cond_conv(cond)
        h += cond
        
        h = tf.transpose(self.S42(tf.transpose(h, [2, 0, 1])), [1, 2, 0])
        
        out = tf.keras.activations.tanh(h[:,:self.res_channels,:]) * tf.keras.activations.sigmoid(h[:,self.res_channels:,:])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability