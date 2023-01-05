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

class Residual_group(tf.keras.Model):
    def __init__(self, res_channels, skip_channels, num_res_layers, 
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t1 = tf.keras.layers.Dense(diffusion_step_embed_dim_mid)
        self.fc_t2 = tf.keras.layers.Dense(diffusion_step_embed_dim_out)
        
        self.residual_blocks = []
        for n in range(self.num_res_layers):
            self.residual_blocks.append(
                Residual_block(
                    res_channels, 
                    skip_channels, 
                    diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                    in_channels=in_channels,
                    s4_lmax=s4_lmax,
                    s4_d_state=s4_d_state,
                    s4_dropout=s4_dropout,
                    s4_bidirectional=s4_bidirectional,
                    s4_layernorm=s4_layernorm))

            
    def call(self, input_data):
        noise, conditional, diffusion_steps = input_data

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            residual_layer = self.residual_blocks[n]
            h, skip_n = residual_layer((h, conditional, diffusion_step_embed))  
            skip += skip_n  

        return skip * math.sqrt(1.0 / self.num_res_layers)

class SSSDS4Imputer(tf.keras.Model):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
                 num_res_layers,
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(SSSDS4Imputer, self).__init__()

        self.init_conv = tf.keras.Sequential(
            [Conv(in_channels, res_channels, kernel_size=1), 
            tf.keras.layers.ReLU()])
        
        self.residual_layer = Residual_group(
                                  res_channels=res_channels, 
                                  skip_channels=skip_channels, 
                                  num_res_layers=num_res_layers, 
                                  diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                  diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                  diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                  in_channels=in_channels,
                                  s4_lmax=s4_lmax,
                                  s4_d_state=s4_d_state,
                                  s4_dropout=s4_dropout,
                                  s4_bidirectional=s4_bidirectional,
                                  s4_layernorm=s4_layernorm)
        
        self.final_conv = tf.keras.Sequential(
            [Conv(skip_channels, skip_channels, kernel_size=1),
            tf.keras.layers.ReLU(),
            ZeroConv1d(skip_channels, out_channels)])

    def call(self, input_data):
        
        noise, conditional, mask, diffusion_steps = input_data 

        conditional = conditional * mask
        conditional = tf.experimental.numpy.concatenate([conditional, tf.cast(mask, dtype=tf.float32)], axis=1)

        x = noise
        x = self.init_conv(x)
        x = self.residual_layer((x, conditional, diffusion_steps))
        y = self.final_conv(x)

        return y
