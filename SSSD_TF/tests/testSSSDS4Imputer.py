import sys
sys.path.insert(1, '/data/3glusterfs/home/yiding/sssd')

from SSSD.src.imputers.SSSDS4Imputer import SSSDS4Imputer as TorchSSSDS4Imputer
from SSSD_TF.imputers.SSSDS4Imputer import SSSDS4Imputer

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    model_config = {
        "in_channels": 14, 
        "out_channels":14,
        "num_res_layers": 36,
        "res_channels": 256, 
        "skip_channels": 256,
        "diffusion_step_embed_dim_in": 128,
        "diffusion_step_embed_dim_mid": 512,
        "diffusion_step_embed_dim_out": 512,
        "s4_lmax": 100,
        "s4_d_state":64,
        "s4_dropout":0.0,
        "s4_bidirectional":1,
        "s4_layernorm":1
    }

    model_pt = TorchSSSDS4Imputer(**model_config).cuda()
    model_pt.eval()
    # transformed_X.shape=torch.Size([50, 14, 100]), cond.shape=torch.Size([50, 14, 100]), mask.shape=torch.Size([50, 14, 100]), diffusion_steps.view(B, 1).shape=torch.Size([50, 1])
    transformed_X_np = np.random.uniform(-1, 1, (50, 14, 100))
    transformed_X_pt = torch.Tensor(transformed_X_np).cuda()
    cond_np = np.random.uniform(0, 3, (50, 14, 100))
    cond_pt = torch.Tensor(cond_np).cuda()
    mask_np = np.random.choice([0, 1], size=(50, 14, 100), p=[0.1, 0.9])
    mask_pt = torch.Tensor(mask_np).cuda()
    diffusion_steps_np = np.random.randint(0, 200, (50, 1))
    diffusion_steps_pt = torch.Tensor(diffusion_steps_np).cuda()
    
    epsilon_theta_pt = model_pt(
        (transformed_X_pt, cond_pt, mask_pt, diffusion_steps_pt,))

    print(f"{epsilon_theta_pt.shape=}") # torch.Size([50, 14, 100])

    model_tf = SSSDS4Imputer(**model_config)
    model_tf.trainable = False

    transformed_X_tf = tf.convert_to_tensor(transformed_X_np, dtype=tf.float32)
    cond_tf = tf.convert_to_tensor(cond_np, dtype=tf.float32)
    mask_tf = tf.convert_to_tensor(mask_np, dtype=tf.float32)
    diffusion_steps_tf = tf.convert_to_tensor(diffusion_steps_np, dtype=tf.int32)

    epsilon_theta_tf = model_tf(
        (transformed_X_tf, cond_tf, mask_tf, diffusion_steps_tf,))

    print(f"{epsilon_theta_tf.shape=}") # TensorShape([50, 14, 100])

    # print(f"{epsilon_theta_pt=}")
    # print(f"{epsilon_theta_tf=}")
    # With pseudo inputs, both models produce the same output (all zeros).
    # yiding: TensorFlow model is much slower.