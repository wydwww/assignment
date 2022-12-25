import numpy as np
import random
from tqdm import tqdm
import pickle
import math
import argparse
import datetime
import json
import yaml
import os
import logging
from functools import partial
from scipy import special as ss
from einops import rearrange, repeat
import opt_einsum as oe

import tensorflow as tf
import tensorflow_probability as tfp

from pytorch_lightning.utilities import rank_zero_only

contract = oe.contract
contract_expression = oe.contract_expression

''' Standalone CSDI + S4 imputer for random missing, non-random missing and black-out missing.
The notebook contains CSDI and S4 functions and utilities. However the imputer is located in the last Class of
the notebook, please see more documentation of use there. Additional at this file can be added for CUDA multiplication 
the cauchy kernel.'''

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
log = get_logger(__name__)

# yiding: Skip Cauchy kernel

# yiding: Skip pykeops

def Activation(activation=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return tf.identity()
    elif activation == 'tanh':
        return tf.nn.tanh()
    elif activation == 'relu':
        return tf.nn.relu()
    elif activation == 'gelu':
        return tf.nn.gelu()
    elif activation in ['swish', 'silu']:
        return tf.nn.silu()
    elif activation == 'sigmoid':
        return tf.nn.sigmoid()
    else:
        # yiding: Skip GLU
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

def get_initializer(name, activation=None):
    if activation in [ None, 'id', 'identity', 'linear', 'modrelu' ]:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu' # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")

    if name == 'uniform':
        initializer = partial(tf.keras.initializers.HeUniform, nonlinearity=nonlinearity)
    elif name == 'normal':
        initializer = partial(tf.keras.initializers.HeNormal
, nonlinearity=nonlinearity)
    elif name == 'xavier':
        initializer = tf.keras.initializers.GlorotNormal
    elif name == 'zero':
        initializer = partial(tf.keras.initializers.Constant, val=0)
    elif name == 'one':
        initializer = partial(tf.keras.initializers.Constant, val=1)
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")

    return initializer

# yiding: Test done
class TransposedLinear(tf.keras.layers.Layer):
    """ Linear module on the second-to-last dimension """

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()

        self.weight = tf.Variable(tf.experimental.numpy.empty([d_output, d_input]), trainable=True)
        tf.keras.initializers.HeUniform(self.weight) # yiding: TensorFlow does not have an `a` argument

        if bias:
            bound = 1 / math.sqrt(d_input)
            initializer = tf.keras.initializers.RandomUniform(-bound, bound)
            self.bias = tf.Variable(initializer(shape=[d_output, 1]), trainable=True)

        else:
            self.bias = 0.0
    
    def forward(self, x):
        return contract('... u l, v u -> ... v l', x, self.weight) + self.bias

# yiding: Test done
def LinearActivation(
        d_input, d_output, bias=True,
        zero_bias_init=False,
        transposed=False,
        initializer=None,
        activation=None,
        activate=False, # Apply activation as part of this module
        weight_norm=False,
        **kwargs,
    ):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """

    # Construct core module
    linear_cls = TransposedLinear if transposed else tf.keras.layers.Dense
    if activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs) if transposed \
        else linear_cls(d_output, input_shape=(d_input, ), activation=None, use_bias=bias, **kwargs)

    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    # Initialize bias
    if bias and zero_bias_init:
        tf.keras.initializers.Zeros(linear.bias, shape=[d_output, 1])
        # linear.bias = tf.Variable(tf.keras.initializers.Zeros(shape=[d_output, 1]), trainable=True)

    # Weight norm
    if weight_norm:
        linear = tfp.layers.weight_norm.WeightNorm(linear)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = tf.keras.Sequential(linear, activation)
    return linear

""" Misc functional utilities """

# yiding: Skip krylov()

# yiding: Test done
# self.L=100
# self.dA.size()=torch.Size([512, 64, 64])
# dA_L.size()=torch.Size([512, 64, 64])
def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """

    # yiding: Does .to(A) work in TensorFlow? No. Use tf.cast() instead.
    I = tf.cast(tf.eye(A.shape[-1]), dtype=A.dtype) # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)

""" HiPPO utilities """

def embed_c2r(A):
    A = rearrange(A, '... m n -> ... m () n ()')
    A = np.pad(A, ((0, 0), (0, 1), (0, 0), (0, 1))) + \
        np.pad(A, ((0, 0), (1, 0), (0, 0), (1,0)))
    return rearrange(A, 'm x n y -> (m x) (n y)')

def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures

    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    elif measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(.5 * (ss.gammaln(np.arange(N)+alpha+1) - ss.gammaln(np.arange(N)+1)))
        A = (1./L[:, None]) * A * L[None, :]
        B = (1./L[:, None]) * B * np.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == 'fourier':
        freqs = np.arange(N//2)
        d = np.stack([freqs, np.zeros(N//2)], axis=-1).reshape(-1)[:-1]
        A = 2*np.pi*(np.diag(d, 1) - np.diag(d, -1))
        A = A - embed_c2r(np.ones((N//2, N//2)))
        B = embed_c2r(np.ones((N//2, 1)))[..., :1]
    elif measure == 'random':
        A = np.random.randn(N, N) / N
        B = np.random.randn(N, 1)
    elif measure == 'diagonal':
        A = -np.diag(np.exp(np.random.randn(N)))
        B = np.random.randn(N, 1)
    else:
        raise NotImplementedError

    return A, B

def rank_correction(measure, N, rank=1, dtype=tf.float32):
    """ Return low-rank matrix L such that A + L is normal """

    if measure == 'legs':
        assert rank >= 1
        P = tf.expand_dims(tf.experimental.numpy.sqrt(.5+tf.experimental.numpy.arange(N, dtype=dtype)), 0) # (1 N)
    elif measure == 'legt':
        assert rank >= 2
        P = tf.experimental.numpy.sqrt(1+2*tf.experimental.numpy.arange(N, dtype=dtype)) # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = tf.experimental.numpy.stack([P0, P1], dim=0) # (2 N)
    elif measure == 'lagt':
        assert rank >= 1
        P = .5**.5 * tf.ones(1, N, dtype=dtype)
    elif measure == 'fourier':
        P = tf.ones(N, dtype=dtype) # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = tf.experimental.numpy.stack([P0, P1], dim=0) # (2 N)
    else: raise NotImplementedError

    d = P.shape[0]
    if rank > d:
        P = tf.experimental.numpy.concatenate([P, tf.zeros(rank-d, N, dtype=dtype)], dim=0) # (rank N)
    return P