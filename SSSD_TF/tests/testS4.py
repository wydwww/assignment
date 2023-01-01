import sys
sys.path.insert(1, '/data/3glusterfs/home/yiding/sssd')

from SSSD.src.imputers.S4Model import S4 as TorchS4
from SSSD_TF.imputers.S4Model import S4

from SSSD.src.imputers.S4Model import S4Layer as TorchS4Layer
from SSSD_TF.imputers.S4Model import S4Layer

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import math

class TorchTest(nn.Module):
    def __init__(self, features, N, lmax, bidirectional):
        super(TorchTest, self).__init__()
        self.s4_layer = TorchS4(d_model=features, 
                                d_state=N, 
                                l_max=lmax, 
                                bidirectional=bidirectional,
                                transposed=True)
    
    def forward(self, x):
        # x has shape seq, batch, feature
        x = x.permute((1,2,0)) # batch, feature, seq (as expected from S4 with transposed=True)
        xout, _ = self.s4_layer(x) # batch, feature, seq
        return xout

class TFTest(tf.keras.Model):
    def __init__(self, features, N, lmax, bidirectional):
        super(TFTest, self).__init__()
        self.s4_layer = S4(d_model=features, 
                        d_state=N, 
                        l_max=lmax, 
                        bidirectional=bidirectional,
                        transposed=True)
    
    def call(self, x):
        # x has shape batch, seq, feature
        x = tf.transpose(x, [1, 2, 0])
        xout, _ = self.s4_layer(x) # batch, seq, feature
        return xout

# features=512, lmax=100, N=64, dropout=0.0, bidirectional=1, layer_norm=1

# x.shape=torch.Size([100, 500, 512])

if __name__ == '__main__':
    features = 512
    lmax = 100
    N = 64
    bidirectional = 1
    
    model_pt = TorchTest(features, N, lmax, bidirectional)
    model_pt.eval()

    input_np = np.random.uniform(0, 1, (1, 500, 512))
    input_pt = torch.Tensor(input_np)
    input_tf = tf.convert_to_tensor(input_np)

    output_pt = model_pt(input_pt)
    print(output_pt.shape) # torch.Size([500, 512, 1])

    model_tf = TFTest(features, N, lmax, bidirectional)
    model_tf.trainable = False

    output_tf = model_tf(input_tf)
    print(output_tf.shape) # (500, 512, 1)

    # The output shapes of S4 in PyTorch and TensorFlow are the same.
    # print(output_pt)
    # print(output_tf)


    print("Testing S4Layer:")
    # 2*self.res_channels=512, s4_lmax=100, s4_d_state=64, s4_dropout=0.0, s4_bidirectional=1, s4_layernorm=1

    TestS4LayerPT = TorchS4Layer(features=512,
                                lmax=100,
                                N=64,
                                dropout=0.0,
                                bidirectional=1)
    TestS4LayerPT.eval()

    output_s4_pt = TestS4LayerPT(input_pt)

    TestS4LayerTF = S4Layer(features=512, 
                            lmax=100, 
                            N=64, 
                            dropout=0.0, 
                            bidirectional=1)
    TestS4LayerTF.trainable = False

    output_s4_tf = TestS4LayerTF(input_tf)

    print(output_s4_pt.shape) # torch.Size([1, 500, 512])
    print(output_s4_tf.shape) # (1, 500, 512)

    # print(output_s4_pt)
    # print(output_s4_tf)