import sys
sys.path.insert(1, '/data/3glusterfs/home/yiding/sssd')

from SSSD.src.imputers.SSSDS4Imputer import Conv as TorchConv
from SSSD_TF.imputers.SSSDS4Imputer import Conv, ZeroConv1d

from SSSD.src.imputers.SSSDS4Imputer import ZeroConv1d as TorchZeroConv1d

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