import sys
sys.path.insert(1, '/data/3glusterfs/home/yiding/sssd')

from SSSD.src.imputers.S4Model import LinearActivation as TorchLinearActivation
from SSSD_TF.imputers.S4Model import LinearActivation

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

class TorchTest(nn.Module):
    def __init__(self):
        super(TorchTest, self).__init__()

    def forward(self, x, transposed=False):
        activation = TorchLinearActivation(64, 64, transposed=transposed)
        return activation(x)

class TFTest(tf.keras.Model):
    def __init__(self):
        super(TFTest, self).__init__()

    def call(self, x, transposed=False):
        activation = LinearActivation(64, 64, transposed=transposed)
        return activation(x)

if __name__ == '__main__':
    model = TorchTest()
    model.eval()

    model2 = TFTest()
    model2.trainable = False

    input_np = np.random.uniform(0, 1, (1, 128, 64))
    input_pt = torch.Tensor(input_np)
    input_tf = tf.convert_to_tensor(input_np)

    print("Testing the output shape of LinearActivation():")
    output = model(input_pt)
    print(output.size())
    output2 = model2(input_tf)
    print(output2.shape)

    output_transposed = model(input_pt.transpose(2, 1), transposed=True)
    print(output_transposed.size())
    output2_transposed = model2(tf.transpose(input_tf, perm=[0, 2, 1]), transposed=True)
    print(output2_transposed.shape)