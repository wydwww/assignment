import sys
sys.path.insert(1, '/data/3glusterfs/home/yiding/sssd')

from SSSD.src.imputers.S4Model import nplr as torch_nplr
from SSSD_TF.imputers.S4Model import nplr

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

def check_error(pt_output, tf_output, epsilon=1e-5):
    pt_output = pt_output.numpy()
    tf_output = tf_output.numpy()

    error = (pt_output - tf_output)/pt_output
    error = np.max(np.abs(error))
    print('Error:', error)

    assert error < epsilon
    return error

if __name__ == '__main__':

    print("Testing nplr():")
    w1, P1, B1, V1 = torch_nplr('legs', 64)
    w2, P2, B2, V2 = nplr('legs', 64)
    print(w1.size())
    print(w2.shape)
    print(P1.size())
    print(P2.shape)
    print(B1.size())
    print(B2.shape)
    print(V1.size())
    print(V2.shape)

# Pass
# torch.Size([32])
# (32,)
# torch.Size([1, 32])
# (1, 32)
# torch.Size([32])
# (32,)
# torch.Size([64, 32])
# (64, 32)