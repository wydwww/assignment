import sys
sys.path.insert(1, '/data/3glusterfs/home/yiding/sssd')

from SSSD.src.imputers.S4Model import _c2r as torch_c2r
from SSSD_TF.imputers.S4Model import _c2r
from SSSD.src.imputers.S4Model import _r2c as torch_r2c
from SSSD_TF.imputers.S4Model import _r2c
from SSSD.src.imputers.S4Model import _conj as torch_conj
from SSSD_TF.imputers.S4Model import _conj
from SSSD.src.imputers.S4Model import _resolve_conj as torch_resolve_conj
from SSSD_TF.imputers.S4Model import _resolve_conj

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

    print("Testing Lambda functions:")
    L = 100
    omega1 = torch.tensor(np.exp(-2j * np.pi / (L))) ** torch.arange(0, L // 2 + 1)
    omega2 = tf.constant(np.exp(-2j * np.pi / (L)))
    omega2 = omega2 ** tf.cast(tf.experimental.numpy.arange(0, L // 2 + 1), dtype=omega2.dtype)

    _c2r_res1 = torch_c2r(omega1)
    _c2r_res2 = _c2r(omega2)
    # print(_c2r_res1)
    # print(_c2r_res2)
    # _c2r_res1 and _c2r_res2 are the same

    _r2c_res1 = torch_r2c(_c2r_res1)
    _r2c_res2 = _r2c(_c2r_res2)
    # print(_r2c_res1)
    # print(_r2c_res2)
    # _r2c_res1 and _r2c_res2 are the same

    _conj_res1 = torch_conj(_r2c_res1)
    _conj_res2 = _conj(_r2c_res2)
    # print(_conj_res1)
    # print(_conj_res2)
    # _conj_res1 and _conj_res2 are the same
    
    _resolve_conj_res1 = torch_resolve_conj(omega1)
    _resolve_conj_res2 = _resolve_conj(omega2)
    # print(_resolve_conj_res1)
    # print(_resolve_conj_res2)
    # _resolve_conj_res1 and _resolve_conj_res2 are the same

    check_error(_conj_res1, _conj_res2)
    check_error(_resolve_conj_res1, _resolve_conj_res2)