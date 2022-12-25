import sys
sys.path.insert(1, '/data/3glusterfs/home/yiding/sssd')

from SSSD.src.imputers.S4Model import power as torch_power
from SSSD_TF.imputers.S4Model import power
from SSSD.src.imputers.S4Model import rank_correction as torch_rank_correction
from SSSD_TF.imputers.S4Model import rank_correction

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

class TorchTest(nn.Module):
    def __init__(self):
        super(TorchTest, self).__init__()

    def forward(self, L, A):
        return torch_power(L, A)

class TFTest(tf.keras.Model):
    def __init__(self):
        super(TFTest, self).__init__()

    def call(self, L, A):
        return power(L, A)

def check_error(pt_output, tf_output, epsilon=1e-5):
    pt_output = pt_output.numpy()
    tf_output = tf_output.numpy()

    error = (pt_output - tf_output)/pt_output
    error = np.max(np.abs(error))
    print('Error:', error)

    assert error < epsilon
    return error

if __name__ == '__main__':

    print("Testing rank_correction():")
    rank_correction_res1 = torch_rank_correction('legs', 64)
    rank_correction_res2 = rank_correction('legs', 64)
    print(rank_correction_res1)
    print(rank_correction_res2)
    # Error: 4.479395314092927e-06
    # Pass
    check_error(rank_correction_res1, rank_correction_res2)

    print("Testing power():")
    model = TorchTest()
    model.eval()

    model2 = TFTest()
    model2.trainable = False

    input_np = np.random.uniform(0, 1, (32, 4, 4))
    input_pt = torch.Tensor(input_np)
    input_tf = tf.convert_to_tensor(input_np)

    L = 100

    output = model(L, input_pt)
    print(output.size())
    # print(output)
    output2 = model2(L, input_tf)
    print(output2.shape)
    # print(output2)

    # Error: 2.0473393863560666e-06
    # Pass
    check_error(output, output2)