import sys
sys.path.insert(1, '/data/3glusterfs/home/yiding/sssd')

from SSSD.src.imputers.SSSDS4Imputer import Conv as TorchConv
from SSSD.src.imputers.SSSDS4Imputer import ZeroConv1d as TorchZeroConv1d
from SSSD.src.imputers.SSSDS4Imputer import Residual_block as TorchResidual_block
from SSSD_TF.imputers.SSSDS4Imputer import Conv, ZeroConv1d, Residual_block

import torch
import torch.nn as nn
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

if __name__ == '__main__':
    in_channels = 256
    out_channels = 512
    kernel_size = 3

    print("Testing Conv:")

    model_pt = TorchConv(in_channels, out_channels, kernel_size=kernel_size)
    model_pt.eval()

    input_np = np.random.uniform(0, 1, (500, 256, 100))
    input_pt = torch.Tensor(input_np)
    print(f"{input_pt.shape=}")
    input_tf = tf.convert_to_tensor(input_np)
    print(f"{input_tf.shape=}")

    output_pt = model_pt(input_pt)
    print(f"{output_pt.shape=}")

    model_tf = Conv(in_channels, out_channels, kernel_size=kernel_size)
    output_tf = model_tf(input_tf)
    print(f"{output_tf.shape=}")

    # print(output_pt)
    # print(output_tf)
    # Sanity check passed.

    print("Testing ZeroConv1d:")

    model_pt = TorchZeroConv1d(in_channels, out_channels)
    model_pt.eval()

    model_tf = ZeroConv1d(in_channels, out_channels)

    output_pt = model_pt(input_pt)
    print(f"{output_pt.shape=}")

    output_tf = model_tf(input_tf)
    print(f"{output_tf.shape=}")

    # print(output_pt)
    # print(output_tf)
    # Both are all zeros.

print("Testing Residual_block:")

# Initialize models
res_channels=256
skip_channels=256
diffusion_step_embed_dim_out=512
in_channels=14
s4_lmax=100
s4_d_state=64
s4_dropout=0.0
s4_bidirectional=1
s4_layernorm=1

model_pt = TorchResidual_block(res_channels, skip_channels, 
                            diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                            in_channels=in_channels,
                            s4_lmax=s4_lmax,
                            s4_d_state=s4_d_state,
                            s4_dropout=s4_dropout,
                            s4_bidirectional=s4_bidirectional,
                            s4_layernorm=s4_layernorm)
model_pt.eval()

model_tf = Residual_block(res_channels, skip_channels, 
                            diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                            in_channels=in_channels,
                            s4_lmax=s4_lmax,
                            s4_d_state=s4_d_state,
                            s4_dropout=s4_dropout,
                            s4_bidirectional=s4_bidirectional,
                            s4_layernorm=s4_layernorm)
model_tf.trainable = False

# h.shape=torch.Size([500, 256, 100]), conditional.shape=torch.Size([500, 28, 100]), diffusion_step_embed.shape=torch.Size([500, 512])
# Initialize inputs

h_np = np.random.uniform(0, 1, (500, 256, 100))
h_pt = torch.Tensor(h_np)
h_tf = tf.convert_to_tensor(h_np)

conditional_np = np.random.uniform(0, 1, (500, 28, 100))
conditional_pt = torch.Tensor(conditional_np)
conditional_tf = tf.convert_to_tensor(conditional_np)

diffusion_step_embed_np = np.random.uniform(0, 1, (500, 512))
diffusion_step_embed_pt = torch.Tensor(diffusion_step_embed_np)
diffusion_step_embed_tf = tf.convert_to_tensor(diffusion_step_embed_np)

output_h_pt, skip_n_pt = model_pt((h_pt, conditional_pt, diffusion_step_embed_pt))
print(f"{output_h_pt.shape=}")
print(f"{skip_n_pt.shape=}")

output_h_tf, skip_n_tf = model_tf((h_tf, conditional_tf, diffusion_step_embed_tf))
print(f"{output_h_tf.shape=}")
print(f"{skip_n_tf.shape=}")