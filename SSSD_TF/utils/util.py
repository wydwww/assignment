import os
import numpy as np
import tensorflow as tf
import random


def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]

def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = tf.experimental.numpy.exp(tf.experimental.numpy.arange(start=0, stop=half_dim) * -_embed)
    _embed = diffusion_steps * _embed
    diffusion_step_embed = tf.experimental.numpy.concatenate((tf.experimental.numpy.sin(_embed), tf.experimental.numpy.cos(_embed)), axis=1)

    return diffusion_step_embed