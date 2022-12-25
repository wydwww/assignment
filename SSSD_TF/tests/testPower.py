import sys
sys.path.insert(1, '/data/3glusterfs/home/yiding/sssd')

from SSSD.src.imputers.S4Model import power as torchpower
from SSSD_TF.imputers.S4Model import power

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

class TorchTest(nn.Module):
    def __init__(self):
        super(TorchTest, self).__init__()

    def forward(self, L, A):
        return torchpower(L, A)

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
    model = TorchTest()
    model.eval()

    model2 = TFTest()
    model2.trainable = False

    input_np = np.random.uniform(0, 1, (32, 4, 4))
    input_pt = torch.Tensor(input_np)
    input_tf = tf.convert_to_tensor(input_np)

    L = 100

    print("Testing power():")
    output = model(L, input_pt)
    print(output.size())
    # print(output)
    output2 = model2(L, input_tf)
    print(output2.shape)
    # print(output2)

    # Error: 2.0473393863560666e-06
    # Pass
    check_error(output, output2)