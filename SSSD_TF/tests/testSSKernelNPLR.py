import sys
sys.path.insert(1, '/data/3glusterfs/home/yiding/sssd')

from SSSD.src.imputers.S4Model import SSKernelNPLR as TorchSSKernelNPLR
from SSSD_TF.imputers.S4Model import SSKernelNPLR

from SSSD.src.imputers.S4Model import nplr as torch_nplr
from SSSD_TF.imputers.S4Model import nplr

from SSSD_TF.imputers.S4Model import HippoSSKernel

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import math

"""
Unit tests for NPLR kernel:
dt_min=0.001, dt_max=0.1
measure='legs', self.N=64, rank=1, dtype=torch.float32, self.H=512, self.N=64, channels=2
w.shape=torch.Size([32]), p.shape=torch.Size([1, 32]), B.shape=torch.Size([32]), C.shape=torch.Size([2, 512, 32]), log_dt.shape=torch.Size([512]), hurwitz=False, trainable=None, lr=None, tie_state=False, length_correction=True, verbose=False
k.float().shape=torch.Size([2, 512, 100])
self.rate=1.0, L=100
"""

class TorchTest(nn.Module):
    def __init__(self, L, w, p, B, C, log_dt):
        super(TorchTest, self).__init__()
        self.L = L
        self.w = w
        self.p = p
        self.B = B
        self.C = C
        self.log_dt = log_dt
        self.rate = 1.0

        self.kernel = TorchSSKernelNPLR(
            self.L, self.w, self.p, self.B, self.C,
            self.log_dt,
            hurwitz=False,
            trainable=None,
            lr=None,
            tie_state=None,
            length_correction=True,
            verbose=False,
        )

    def forward(self, L=None):
        k, _ = self.kernel(rate=self.rate, L=L)
        return k.float()

class TFTest(tf.keras.Model):
    def __init__(self, L, w, p, B, C, log_dt):
        super(TFTest, self).__init__()
        self.L = L
        self.w = w
        self.p = p
        self.B = B
        self.C = C
        self.log_dt = log_dt
        self.rate = 1.0

        self.kernel = SSKernelNPLR(
            self.L, self.w, self.p, self.B, self.C,
            self.log_dt,
            hurwitz=False,
            trainable=None,
            lr=None,
            tie_state=None,
            length_correction=True,
            verbose=False,
        )

    def call(self, L=None):
        k, _ = self.kernel(state=None, rate=self.rate, L=L)
        return tf.cast(k, tf.float32)

def check_error(pt_output, tf_output, epsilon=1e-5):
    pt_output = pt_output.numpy()
    tf_output = tf_output.numpy()

    error = (pt_output - tf_output)/pt_output
    error = np.max(np.abs(error))
    print('Error:', error)

    assert error < epsilon
    return error

if __name__ == '__main__':

    dt_max = 0.1
    dt_min = 0.001
    measure = 'legs'
    N = 64
    H = 512
    L = 100
    rank = 1
    channels = 2
    log_dt = torch.rand(H, dtype=torch.float32) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
    w, p, B, _ = torch_nplr(measure, N, rank, dtype=torch.float32)
    C = torch.randn(channels, H, N // 2, dtype=torch.cfloat)

    model = TorchTest(L, w, p, B, C, log_dt)
    model.eval()

    output = model(L)
    print("Testing SSKernelNPLR's output shape in PyTorch:")
    print(output.size()) # torch.Size([2, 512, 100])

    log_dt_tf = tf.experimental.numpy.random.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
    log_dt_tf = tf.cast(log_dt_tf, tf.float32)
    w_tf, p_tf, B_tf, _ = nplr(measure, N, rank, dtype=tf.float32)
    C_tf = tf.experimental.numpy.random.randn(channels, H, N // 2)
    C_tf = tf.cast(C_tf, tf.complex64)

    model_tf = TFTest(L, w_tf, p_tf, B_tf, C_tf, log_dt_tf)
    model_tf.trainable = False

    output_tf = model_tf(L)
    print("Testing SSKernelNPLR's output shape in TensorFlow:")
    print(output_tf.shape) # (2, 512, 100)

    # The output shapes of NPLR in PyTorch and TensorFlow are the same.

    hkernal = HippoSSKernel(H, N, L, channels=channels, verbose=False)
    output_tf_h = hkernal(L)
    print("Testing HippoSSKernel's output shape in TensorFlow:")
    print(output_tf_h.shape) # (2, 512, 100)