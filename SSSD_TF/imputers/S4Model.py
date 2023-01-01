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

tf.experimental.numpy.experimental_enable_numpy_behavior()

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

# yiding: To be tested
_c2r = lambda x: tf.constant([list(a) for a in zip(tf.math.real(x).numpy(), tf.math.imag(x).numpy())])
_r2c = lambda x: tf.dtypes.complex(tf.constant([i[0].numpy() for i in x]), tf.constant([i[1].numpy() for i in x]))
_conj = lambda x: tf.experimental.numpy.concatenate([x, tf.math.conj(x)], axis=-1)
_resolve_conj = lambda x: tf.math.conj(x)

has_cauchy_extension = False
has_pykeops = False

def cauchy_slow(v, z, w):
    """
    v, w: (..., N)
    z: (..., L)
    returns: (..., L)
    """
    cauchy_matrix = tf.expand_dims(v, -1) / (tf.expand_dims(z, -2) - tf.expand_dims(w, -1)) # (... N L)
    return tf.experimental.numpy.sum(cauchy_matrix, axis=-2)

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
    
    def call(self, x):
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

# yiding: Test done
def rank_correction(measure, N, rank=1, dtype=tf.float32):
    """ Return low-rank matrix L such that A + L is normal """

    if measure == 'legs':
        assert rank >= 1
        P = tf.expand_dims(tf.experimental.numpy.sqrt(.5+tf.experimental.numpy.arange(N, dtype=dtype)), 0) # (1 N)
    elif measure == 'legt':
        assert rank >= 2
        P = tf.experimental.numpy.sqrt(1+2*tf.experimental.numpy.arange(N, dtype=dtype)) # (N)
        P0 = tf.identity(P)
        P0[0::2] = 0.
        P1 = tf.identity(P)
        P1[1::2] = 0.
        P = tf.experimental.numpy.stack([P0, P1], dim=0) # (2 N)
    elif measure == 'lagt':
        assert rank >= 1
        P = .5**.5 * tf.ones(1, N, dtype=dtype)
    elif measure == 'fourier':
        P = tf.ones(N, dtype=dtype) # (N)
        P0 = tf.identity(P)
        P0[0::2] = 0.
        P1 = tf.identity(P)
        P1[1::2] = 0.
        P = tf.experimental.numpy.stack([P0, P1], dim=0) # (2 N)
    else: raise NotImplementedError

    d = P.shape[0]
    if rank > d:
        P = tf.experimental.numpy.concatenate([P, tf.zeros(rank-d, N, dtype=dtype)], axis=0) # (rank N)
    return P

# yiding: Test done
def nplr(measure, N, rank=1, dtype=tf.float32):
    """ Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    assert dtype == tf.float32 or tf.complex64
    if measure == 'random':
        dtype = tf.complex64 if dtype == tf.float32 else tf.complex128
        # w = torch.randn(N//2, dtype=dtype)
        w = -tf.experimental.numpy.exp(tf.experimental.numpy.random.randn(N//2)) + 1j*tf.experimental.numpy.random.randn(N//2)
        P = tf.experimental.numpy.random.randn(rank, N//2, dtype=dtype)
        B = tf.experimental.numpy.random.randn(N//2, dtype=dtype)
        V = tf.eye(N, dtype=dtype)[..., :N//2] # Only used in testing
        return w, P, B, V

    A, B = transition(measure, N)
    A = tf.cast(A, dtype=dtype) # (N, N)
    B = tf.cast(B, dtype=dtype)[:, 0] # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype)
    AP = A + tf.experimental.numpy.sum(tf.expand_dims(P, -2)*tf.expand_dims(P, -1), axis=-3)
    w, V = tf.linalg.eig(AP) # (..., N) (..., N, N)
    # V w V^{-1} = A

    # Only keep one of the conjugate pairs
    w = tf.experimental.numpy.ascontiguousarray(w[..., 0::2])
    V = tf.experimental.numpy.ascontiguousarray(V[..., 0::2])

    V_inv = tf.experimental.numpy.transpose(tf.experimental.numpy.conj(V), (1, 0))

    B = contract('ij, j -> i', V_inv, tf.cast(B, dtype=V.dtype)) # V^* B
    P = contract('ij, ...j -> ...i', V_inv, tf.cast(P, dtype=V.dtype)) # V^* P


    return w, P, B, V

# yiding: Skip test
def bilinear(dt, A, B=None):
    """
    dt: (...) timescales
    A: (... N N)
    B: (... N)
    """
    N = A.shape[-1]
    I = tf.cast(tf.eye(N), dtype=A.dtype)
    A_backwards = I - dt[:, None, None] / 2 * A
    A_forwards = I + dt[:, None, None] / 2 * A

    if B is None:
        dB = None
    else:
        dB = dt[..., None] * tf.linalg.solve(
            A_backwards, tf.expand_dims(B, -1)
        )
        dB = tf.squeeze(dB, -1) # (... N)

    dA = tf.linalg.solve(A_backwards, A_forwards)  # (... N N)
    return dA, dB

class SSKernelNPLR(tf.keras.layers.Layer):
    """Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)

    The class name stands for 'State-Space SSKernel for Normal Plus Low-Rank'.
    The parameters of this function are as follows.

    A: (... N N) the state matrix
    B: (... N) input matrix
    C: (... N) output matrix
    dt: (...) timescales / discretization step size
    p, q: (... P N) low-rank correction to A, such that Ap=A+pq^T is a normal matrix

    The forward pass of this Module returns:
    (... L) that represents represents FFT SSKernel_L(A^dt, B^dt, C)

    """

    def _setup_C(self, double_length=False):
        """ Construct C~ from C

        double_length: current C is for length L, convert it to length 2L
        """
        C = _r2c(self.C)
        self._setup_state()
        dA_L = power(self.L, self.dA)

        # Multiply C by I - dA_L
        C_ = _conj(C)
        prod = contract("h m n, c h n -> c h m", tf.transpose(dA_L, perm=[0,2,1]), C_)
        if double_length: prod = -prod # Multiply by I + dA_L instead
        C_ = C_ - prod
        C_ = C_[..., :self.N] # Take conjugate pairs again
        self.C = tf.identity(_c2r(C_))

        if double_length:
            self.L *= 2
            self._omega(self.L, dtype=C.dtype, cache=True)

    # yiding: How to handle device?
    def _omega(self, L, dtype, cache=True):
        """ Calculate (and cache) FFT nodes and their "unprocessed" them with the bilinear transform
        This should be called everytime the internal length self.L changes """
        omega = tf.constant(np.exp(-2j * np.pi / (L)), dtype=dtype)  # \omega_{2L}
        omega = omega ** tf.cast(tf.experimental.numpy.arange(0, L // 2 + 1), dtype=omega.dtype)
        z = 2 * (1 - omega) / (1 + omega)
        if cache:
            # self.register_buffer("omega", _c2r(omega))
            # self.register_buffer("z", _c2r(z))
            setattr(self, "omega", _c2r(omega))
            setattr(self, "z", _c2r(z))
            self.omega.trainable = False
            self.z.trainable = False
            # omega_value = _c2r(omega)
            # z_value = _c2r(z)
            # self.omega = self.add_weight("omega", shape=omega_value.shape, dtype=omega_value.dtype, trainable=False)
            # self.z = self.add_weight("z", shape=z_value.shape, dtype=z_value.dtype, trainable=False)
            # self.omega.assign(omega_value)
            # self.z.assign(z_value)
        return omega, z

    def __init__(
        self,
        L, w, P, B, C, log_dt,
        hurwitz=False,
        trainable=None,
        lr=None,
        tie_state=False,
        length_correction=True,
        verbose=False,
    ):
        """
        L: Maximum length; this module computes an SSM kernel of length L
        w: (N)
        p: (r, N) low-rank correction to A
        q: (r, N)
        A represented by diag(w) - pq^*

        B: (N)
        dt: (H) timescale per feature
        C: (H, C, N) system is 1-D to c-D (channels)

        hurwitz: tie pq and ensure w has negative real part
        trainable: toggle which of the parameters is trainable
        lr: add hook to set lr of hippo parameters specially (everything besides C)
        tie_state: tie all state parameters across the H hidden features
        length_correction: multiply C by (I - dA^L) - can be turned off when L is large for slight speedup at initialization (only relevant when N large as well)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        super().__init__()
        self.hurwitz = hurwitz
        self.tie_state = tie_state
        self.verbose = verbose

        # Rank of low-rank correction
        self.rank = P.shape[-2]
        assert w.shape[-1] == P.shape[-1] == B.shape[-1] == C.shape[-1]
        self.H = log_dt.shape[-1]
        self.N = w.shape[-1]

        # Broadcast everything to correct shapes
        C = tf.broadcast_to(C, tf.broadcast_dynamic_shape(C.shape, (1, self.H, self.N))) # (H, C, N)
        H = 1 if self.tie_state else self.H
        B = repeat(B, 'n -> 1 h n', h=H)
        P = repeat(P, 'r n -> r h n', h=H)
        w = repeat(w, 'n -> h n', h=H)

        # Cache Fourier nodes every time we set up a desired length
        self.L = L
        if self.L is not None:
            self._omega(self.L, dtype=C.dtype, cache=True)

        # Register parameters
        # C is a regular parameter, not state
        # self.C = nn.Parameter(_c2r(C.conj().resolve_conj()))
        self.C = tf.Variable(_c2r(_resolve_conj(C)))
        train = False
        if trainable is None: trainable = {}
        if trainable == False: trainable = {}
        if trainable == True: trainable, train = {}, True
        self.register("log_dt", log_dt, trainable.get('dt', train), lr, 0.0)
        self.register("B", _c2r(B), trainable.get('B', train), lr, 0.0)
        self.register("P", _c2r(P), trainable.get('P', train), lr, 0.0)
        if self.hurwitz:
            log_w_real = tf.experimental.numpy.log(-w.real + 1e-3) # Some of the HiPPO methods have real part 0
            w_imag = w.imag
            self.register("log_w_real", log_w_real, trainable.get('A', 0), lr, 0.0)
            self.register("w_imag", w_imag, trainable.get('A', train), lr, 0.0)
            self.Q = None
        else:
            self.register("w", _c2r(w), trainable.get('A', train), lr, 0.0)
            # self.register("Q", _c2r(P.clone().conj().resolve_conj()), trainable.get('P', train), lr, 0.0)
            Q = _resolve_conj(tf.identity(P))
            self.register("Q", _c2r(Q), trainable.get('P', train), lr, 0.0)

        if length_correction:
            self._setup_C()

    def _w(self):
        # Get the internal w (diagonal) parameter
        if self.hurwitz:
            w_real = -tf.experimental.numpy.exp(self.log_w_real)
            w_imag = self.w_imag
            w = w_real + 1j * w_imag
        else:
            w = _r2c(self.w)  # (..., N)
        return w

    def call(self, state=None, rate=1.0, L=None):
        """
        state: (..., s, N) extra tensor that augments B
        rate: sampling rate factor

        returns: (..., c+s, L)
        """
        # Handle sampling rate logic
        # The idea is that this kernel's length (in continuous units) is self.L, while we are asked to provide a kernel of length L at (relative) sampling rate rate
        # If either are not passed in, assume we're not asked to change the scale of our kernel
        assert not (rate is None and L is None)
        if rate is None:
            rate = self.L / L
        if L is None:
            L = int(self.L / rate)

        # Increase the internal length if needed
        while rate * L > self.L:
            self.double_length()

        dt = tf.experimental.numpy.exp(self.log_dt) * rate
        B = _r2c(self.B)
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = tf.math.conj(P) if self.Q is None else _r2c(self.Q)
        w = self._w()

        if rate == 1.0:
            # Use cached FFT nodes
            omega, z = _r2c(self.omega), _r2c(self.z)  # (..., L)
        else:
            omega, z = self._omega(int(self.L/rate), dtype=w.dtype, cache=False)

        if self.tie_state:
            B = repeat(B, '... 1 n -> ... h n', h=self.H)
            P = repeat(P, '... 1 n -> ... h n', h=self.H)
            Q = repeat(Q, '... 1 n -> ... h n', h=self.H)

        # Augment B
        if state is not None:
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute 1/dt * (I + dt/2 A) @ state

            # Can do this without expanding (maybe minor speedup using conj symmetry in theory), but it's easier to read this way
            s = _conj(state) if state.size(-1) == self.N else state # (B H N)
            sA = (
                s * _conj(w) # (B H N)
                - contract('bhm, rhm, rhn -> bhn', s, _conj(Q), _conj(P))
            )
            s = s / tf.expand_dims(dt, -1) + sA / 2
            s = s[..., :self.N]

            B = tf.experimental.numpy.concatenate([s, B], axis=-3)  # (s+1, H, N)

        # Incorporate dt into A
        w = w * tf.expand_dims(dt, -1)  # (H N)

        # Stack B and p, C and q for convenient batching
        B = tf.experimental.numpy.concatenate([B, P], axis=-3) # (s+1+r, H, N)
        C = tf.experimental.numpy.concatenate([C, Q], axis=-3) # (c+r, H, N)

        # Incorporate B and C batch dimensions
        v = tf.expand_dims(B, -3) * tf.expand_dims(C, -4)  # (s+1+r, c+r, H, N)
        # w = w[None, None, ...]  # (1, 1, H, N)
        # z = z[None, None, None, ...]  # (1, 1, 1, L)

        # Calculate resolvent at omega
        r = cauchy_slow(v, z, w)
        r = r * dt[None, None, :, None]  # (S+1+R, C+R, H, L)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[1:, :1, :, :]
            s = (
                r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[:-self.rank, :-self.rank, :, :]
            r01 = r[:-self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, :-self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            r11 = tf.linalg.inv(tf.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            k_f = r00 - tf.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = tf.signal.irfft(k_f)  # (S+1, C, H, L)

        # Truncate to target length
        k = k[..., :L]

        if state is not None:
            k_state = k[:-1, :, :, :]  # (S, C, H, L)
        else:
            k_state = None
        k_B = k[-1, :, :, :] # (C H L)
        return k_B, k_state

    def double_length(self):
        if self.verbose: log.info(f"S4: Doubling length from L = {self.L} to {2*self.L}")
        self._setup_C(double_length=True)

    def _setup_linear(self):
        """ Create parameters that allow fast linear stepping of state """
        w = self._w()
        B = _r2c(self.B) # (H N)
        P = _r2c(self.P)
        Q = tf.experimental.numpy.concatenate(P) if self.Q is None else _r2c(self.Q)

        # Prepare Linear stepping
        dt = tf.experimental.numpy.exp(self.log_dt)
        D = tf.math.reciprocal((2.0 / tf.expand_dims(dt, -1)) - w)  # (H, N)
        R = (tf.eye(self.rank, dtype=w.dtype) + 2*tf.math.real(contract('r h n, h n, s h n -> h r s', Q, D, P))) # (H r r)
        Q_D = rearrange(Q*D, 'r h n -> h r n')
        R = tf.linalg.solve(tf.cast(R, dtype=Q_D.dtype), Q_D) # (H r N)
        R = rearrange(R, 'h r n -> r h n')

        self.step_params = {
            "D": D, # (H N)
            "R": R, # (r H N)
            "P": P, # (r H N)
            "Q": Q, # (r H N)
            "B": B, # (1 H N)
            "E": 2.0 / tf.expand_dims(dt, -1) + w, # (H N)
        }

    def _step_state_linear(self, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.

        Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster

        u: (H) input
        state: (H, N/2) state with conjugate pairs
          Optionally, the state can have last dimension N
        Returns: same shape as state
        """
        C = _r2c(self.C) # View used for dtype/device

        if u is None: # Special case used to find dA
            u = tf.zeros(self.H, dtype=C.dtype)
        if state is None: # Special case used to find dB
            state = tf.zeros([self.H, self.N], dtype=C.dtype)

        step_params = self.step_params.copy()
        if state.shape[-1] == self.N: # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[..., :self.N] # inner outer product
        else:
            assert state.shape[-1] == 2*self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            # TODO worth setting up a contract_expression in default_state if we want to use this at inference time for stepping
            contract_fn = lambda p, x, y: tf.convert_to_tensor(contract('r h n, r h m, ... h m -> ... h n', p.numpy(), x.numpy(), y.numpy())) # inner outer product
        D = step_params["D"]  # (H N)
        E = step_params["E"]  # (H N)
        R = step_params["R"]  # (r H N)
        P = step_params["P"]  # (r H N)
        Q = step_params["Q"]  # (r H N)
        B = step_params["B"]  # (1 H N)

        new_state = E * state - contract_fn(P, Q, state) # (B H N)
        new_state = new_state + 2.0 * B * tf.expand_dims(u, -1)  # (B H N)
        new_state = D * (new_state - contract_fn(P, R, new_state))

        return new_state

    def _setup_state(self):
        """ Construct dA and dB for discretized state equation """

        # Construct dA and dB by using the stepping
        self._setup_linear()
        C = _r2c(self.C) # Just returns a view that we use for finding dtype/device

        state = tf.expand_dims(tf.eye(2*self.N, dtype=C.dtype), -2) # (N 1 N)
        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, "n h m -> h m n")
        self.dA = dA # (H N N)

        u = tf.ones(self.H, dtype=C.dtype)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        self.dB = rearrange(dB, '1 h n -> h n') # (H N)

    def _step_state(self, u, state):
        """ Must be called after self.default_state() is used to construct an initial state!  """
        next_state = self.state_contraction(self.dA, state) + self.input_contraction(self.dB, u)
        return next_state


    def setup_step(self, mode='dense'):
        """ Set up dA, dB, dC discretized parameters for stepping """
        self._setup_state()

        # Calculate original C
        dA_L = power(self.L, self.dA)
        I = tf.cast(tf.eye(self.dA.size[-1]), dtype=dA_L.dtype)
        C = _conj(_r2c(self.C)) # (H C N)

        dC = tf.linalg.solve(
            I - tf.experimental.numpy.transpose(dA_L, (-1, -2)),
            tf.expand_dims(C, -1),
        )
        dC = tf.squeeze(dC, -1)
        self.dC = dC

        # Do special preprocessing for different step modes

        self._step_mode = mode
        if mode == 'linear':
            # Linear case: special step function for the state, we need to handle output
            # use conjugate symmetry by default, which affects the output projection
            self.dC = 2*self.dC[:, :, :self.N]
        elif mode == 'diagonal':
            # Eigendecomposition of the A matrix
            L, V = tf.linalg.eig(self.dA)
            V_inv = tf.linalg.inv(V)
            # Check that the eigendedecomposition is correct
            if self.verbose:
                print("Diagonalization error:", tf.norm(V @ tf.matrix_diag(L) @ V_inv - self.dA, ord=2))

            # Change the parameterization to diagonalize
            self.dA = L
            self.dB = contract('h n m, h m -> h n', V_inv, self.dB)
            self.dC = contract('h n m, c h n -> c h m', V, self.dC)

        elif mode == 'dense':
            pass
        else: raise NotImplementedError("NPLR Kernel step mode must be {'dense' | 'linear' | 'diagonal'}")


    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        N = C.size[-1]
        H = C.size[-2]

        # Cache the tensor contractions we will later do, for efficiency
        # These are put in this function because they depend on the batch size
        if self._step_mode !='linear':
            N *= 2

            if self._step_mode == 'diagonal':
                self.state_contraction = contract_expression(
                    "h n, ... h n -> ... h n",
                    (H, N),
                    batch_shape + (H, N),
                )
            else:
                # Dense (quadratic) case: expand all terms
                self.state_contraction = contract_expression(
                    "h m n, ... h n -> ... h m",
                    (H, N, N),
                    batch_shape + (H, N),
                )

            self.input_contraction = contract_expression(
                "h n, ... h -> ... h n",
                (H, N), # self.dB.shape
                batch_shape + (H,),
            )

        self.output_contraction = contract_expression(
            "c h n, ... h n -> ... c h",
            (C.shape[0], H, N), # self.dC.shape
            batch_shape + (H, N),
        )

        state = tf.zeros(*batch_shape, H, N, dtype=C.dtype)
        return state

    def step(self, u, state):
        """ Must have called self.setup_step() and created state with self.default_state() before calling this """

        if self._step_mode == 'linear':
            new_state = self._step_state_linear(u, state)
        else:
            new_state = self._step_state(u, state)
        y = self.output_contraction(self.dC, new_state)
        return y, new_state

    # yiding: Not sure if this works.
    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter"""

        setattr(self, name, tensor)

        # self.v = self.add_weight(name, shape=tensor.shape, dtype=tensor.dtype, trainable=trainable)
        # self.v.assign(tensor)

        # optim = {}
        # if trainable and lr is not None:
        #     optim["lr"] = lr
        # if trainable and wd is not None:
        #     optim["weight_decay"] = wd
        # if len(optim) > 0:
        #     setattr(getattr(self, name), "_optim", optim)