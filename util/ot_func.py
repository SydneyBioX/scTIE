import os
import math
import ctypes

import numpy as np
from numpy.ctypeslib import ndpointer

cur_folder = os.path.dirname(os.path.abspath(__file__))

lib = ctypes.cdll.LoadLibrary(os.path.join(cur_folder, 'libot.so'))

lib.dummy_float.argtypes = [
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # C
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # K
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # R
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # dx
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # dy
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # p
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # q
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # a
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # b
    ctypes.c_float, # epsilon
    ctypes.c_float, # lambda1
    ctypes.c_float, # lambda2
    ctypes.c_int, # m
    ctypes.c_int  # n
]

lib.dummy_float.restype = ctypes.c_float

lib.dummy_double.argtypes = [
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # C
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # K
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # R
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # dx
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # dy
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # p
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # q
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # a
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # b
    ctypes.c_double, # epsilon
    ctypes.c_double, # lambda1
    ctypes.c_double, # lambda2
    ctypes.c_int, # m
    ctypes.c_int  # n
]

lib.dummy_double.restype = ctypes.c_double

lib.primal_float.argtypes = [
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # C
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # K
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # R
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # dx
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # dy
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # p
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # q
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # a
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # b
    ctypes.c_float, # epsilon
    ctypes.c_float, # lambda1
    ctypes.c_float, # lambda2
    ctypes.c_int, # m
    ctypes.c_int  # n
]

lib.primal_float.restype = ctypes.c_float

lib.primal_double.argtypes = [
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # C
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # K
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # R
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # dx
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # dy
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # p
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # q
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # a
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # b
    ctypes.c_double, # epsilon
    ctypes.c_double, # lambda1
    ctypes.c_double, # lambda2
    ctypes.c_int, # m
    ctypes.c_int  # n
]

lib.primal_double.restype = ctypes.c_double

lib.dual_float.argtypes = [
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # C
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # K
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # R
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # dx
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # dy
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # p
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # q
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # a
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # b
    ctypes.c_float, # epsilon
    ctypes.c_float, # lambda1
    ctypes.c_float, # lambda2
    ctypes.c_int, # m
    ctypes.c_int  # n
]

lib.dual_float.restype = ctypes.c_float

lib.dual_double.argtypes = [
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # C
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # K
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # R
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # dx
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # dy
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # p
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # q
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # a
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # b
    ctypes.c_double, # epsilon
    ctypes.c_double, # lambda1
    ctypes.c_double, # lambda2
    ctypes.c_int, # m
    ctypes.c_int  # n
]

lib.dual_double.restype = ctypes.c_double

lib.compute_duality_gap_float.argtypes = [
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # C
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # K
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # R
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # dx
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # dy
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # p
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # q
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # a
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # b
    ctypes.c_float, # epsilon
    ctypes.c_float, # lambda1
    ctypes.c_float, # lambda2
    ctypes.c_int, # m
    ctypes.c_int  # n
]

lib.compute_duality_gap_float.restype = ctypes.c_float

lib.compute_duality_gap_double.argtypes = [
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # C
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # K
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # R
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # dx
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # dy
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # p
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # q
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # a
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # b
    ctypes.c_double, # epsilon
    ctypes.c_double, # lambda1
    ctypes.c_double, # lambda2
    ctypes.c_int, # m
    ctypes.c_int  # n
]

lib.compute_duality_gap_double.restype = ctypes.c_double

lib.update_k_float.argtypes = [
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # K
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # K_
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # C
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # u
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # v
    ctypes.c_float, # epsilon
    ctypes.c_int, # m
    ctypes.c_int  # n
]

lib.update_k_float.restype = None

lib.update_k_double.argtypes = [
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # K
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # K_
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # C
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # u
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # v
    ctypes.c_double, # epsilon
    ctypes.c_int, # m
    ctypes.c_int  # n
]

lib.update_k_double.restype = None

lib.update_R_float.argtypes = [
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # R
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # K
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # a
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # b
    ctypes.c_int, # m
    ctypes.c_int  # n
]

lib.update_R_float.restype = None

lib.update_R_double.argtypes = [
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # R
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # K
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # a
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # b
    ctypes.c_int, # m
    ctypes.c_int  # n
]

lib.update_R_double.restype = None

'''
lib.update_a_b_float.argtypes = [
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # a
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # b
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # K
    ndpointer(dtype=ctypes.c_float, ndim=2, flags='C_CONTIGUOUS'), # Kt
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # dx
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # dy
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # p
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # q
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # u
    ndpointer(dtype=ctypes.c_float, ndim=1, flags='C_CONTIGUOUS'), # v
    ctypes.c_float, # lambda1
    ctypes.c_float, # lambda2
    ctypes.c_float, # alpha1
    ctypes.c_float, # alpha2
    ctypes.c_float, # epsilon
    ctypes.c_int, # m
    ctypes.c_int  # n
]

lib.update_a_b_float.restype = None

lib.update_a_b_double.argtypes = [
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # a
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # b
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # K
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # Kt
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # dx
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # dy
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # p
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # q
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # u
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # v
    ctypes.c_double, # lambda1
    ctypes.c_double, # lambda2
    ctypes.c_double, # alpha1
    ctypes.c_double, # alpha2
    ctypes.c_double, # epsilon
    ctypes.c_int, # m
    ctypes.c_int  # n
]

lib.update_a_b_double.restype = None
'''

lib.step1_process_double.argtypes = [
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # a
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # b
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # old_a
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # old_b
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # K
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # C
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # dx
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # dy
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # p
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # q
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # u
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # v
    ctypes.c_int, # cur_iter
    ctypes.c_int, # max_iter
    ctypes.c_int, # iters
    ctypes.c_double, # tau
    ctypes.c_double, # lambda1
    ctypes.c_double, # lambda2
    ctypes.c_double, # alpha1
    ctypes.c_double, # alpha2
    ctypes.c_double, # epsilon
    ctypes.c_int, # m
    ctypes.c_int  # n
]

lib.step1_process_double.restype = ctypes.c_int

lib.update_process_double.argtypes = [
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # R
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # a
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # b
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # old_a
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # old_b
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # K
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # _K
    ndpointer(dtype=ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'), # C
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # dx
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # dy
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # p
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # q
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # u
    ndpointer(dtype=ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'), # v
    ctypes.c_int, # epsilon_scalings
    ctypes.c_int, # cur_epsilon_scaling
    ctypes.c_int, # batch_size
    ctypes.c_double, # epsilon
    ctypes.c_double, # threshold
    ctypes.c_double, # tau
    ctypes.c_double, # lambda1
    ctypes.c_double, # lambda2
    ctypes.c_double, # alpha1
    ctypes.c_double, # alpha2
    ctypes.c_int, # m
    ctypes.c_int  # n
]

lib.update_process_double.restype = ctypes.c_double

def dummy_c(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2, use_float=False):
    m, n = C.shape

    if use_float:
        return lib.dummy_float(
            np.ascontiguousarray(C, dtype=ctypes.c_float),
            np.ascontiguousarray(K, dtype=ctypes.c_float),
            np.ascontiguousarray(R, dtype=ctypes.c_float),
            np.ascontiguousarray(dx, dtype=ctypes.c_float),
            np.ascontiguousarray(dy, dtype=ctypes.c_float),
            np.ascontiguousarray(p, dtype=ctypes.c_float),
            np.ascontiguousarray(q, dtype=ctypes.c_float),
            np.ascontiguousarray(a, dtype=ctypes.c_float),
            np.ascontiguousarray(b, dtype=ctypes.c_float),
            epsilon, lambda1, lambda2, m, n)
    else:
        return lib.dummy_double(
            np.ascontiguousarray(C, dtype=ctypes.c_double),
            np.ascontiguousarray(K, dtype=ctypes.c_double),
            np.ascontiguousarray(R, dtype=ctypes.c_double),
            np.ascontiguousarray(dx, dtype=ctypes.c_double),
            np.ascontiguousarray(dy, dtype=ctypes.c_double),
            np.ascontiguousarray(p, dtype=ctypes.c_double),
            np.ascontiguousarray(q, dtype=ctypes.c_double),
            np.ascontiguousarray(a, dtype=ctypes.c_double),
            np.ascontiguousarray(b, dtype=ctypes.c_double),
            epsilon, lambda1, lambda2, m, n)

def primal_c(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2, use_float=False):
    m, n = C.shape

    if use_float:
        return lib.primal_float(
            np.ascontiguousarray(C, dtype=ctypes.c_float),
            np.ascontiguousarray(K, dtype=ctypes.c_float),
            np.ascontiguousarray(R, dtype=ctypes.c_float),
            np.ascontiguousarray(dx, dtype=ctypes.c_float),
            np.ascontiguousarray(dy, dtype=ctypes.c_float),
            np.ascontiguousarray(p, dtype=ctypes.c_float),
            np.ascontiguousarray(q, dtype=ctypes.c_float),
            np.ascontiguousarray(a, dtype=ctypes.c_float),
            np.ascontiguousarray(b, dtype=ctypes.c_float),
            epsilon, lambda1, lambda2, m, n)
    else:
        return lib.primal_double(
            np.ascontiguousarray(C, dtype=ctypes.c_double),
            np.ascontiguousarray(K, dtype=ctypes.c_double),
            np.ascontiguousarray(R, dtype=ctypes.c_double),
            np.ascontiguousarray(dx, dtype=ctypes.c_double),
            np.ascontiguousarray(dy, dtype=ctypes.c_double),
            np.ascontiguousarray(p, dtype=ctypes.c_double),
            np.ascontiguousarray(q, dtype=ctypes.c_double),
            np.ascontiguousarray(a, dtype=ctypes.c_double),
            np.ascontiguousarray(b, dtype=ctypes.c_double),
            epsilon, lambda1, lambda2, m, n)

def dual_c(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2, use_float=False):
    m, n = C.shape

    if use_float:
        return lib.dual_float(
            np.ascontiguousarray(C, dtype=ctypes.c_float),
            np.ascontiguousarray(K, dtype=ctypes.c_float),
            np.ascontiguousarray(R, dtype=ctypes.c_float),
            np.ascontiguousarray(dx, dtype=ctypes.c_float),
            np.ascontiguousarray(dy, dtype=ctypes.c_float),
            np.ascontiguousarray(p, dtype=ctypes.c_float),
            np.ascontiguousarray(q, dtype=ctypes.c_float),
            np.ascontiguousarray(a, dtype=ctypes.c_float),
            np.ascontiguousarray(b, dtype=ctypes.c_float),
            epsilon, lambda1, lambda2, m, n)
    else:
        return lib.dual_double(
            np.ascontiguousarray(C, dtype=ctypes.c_double),
            np.ascontiguousarray(K, dtype=ctypes.c_double),
            np.ascontiguousarray(R, dtype=ctypes.c_double),
            np.ascontiguousarray(dx, dtype=ctypes.c_double),
            np.ascontiguousarray(dy, dtype=ctypes.c_double),
            np.ascontiguousarray(p, dtype=ctypes.c_double),
            np.ascontiguousarray(q, dtype=ctypes.c_double),
            np.ascontiguousarray(a, dtype=ctypes.c_double),
            np.ascontiguousarray(b, dtype=ctypes.c_double),
            epsilon, lambda1, lambda2, m, n)

def compute_duality_gap_c(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2, use_float=False):
    m, n = C.shape

    if use_float:
        return lib.compute_duality_gap_float(
            np.ascontiguousarray(C, dtype=ctypes.c_float),
            np.ascontiguousarray(K, dtype=ctypes.c_float),
            np.ascontiguousarray(R, dtype=ctypes.c_float),
            np.ascontiguousarray(dx, dtype=ctypes.c_float),
            np.ascontiguousarray(dy, dtype=ctypes.c_float),
            np.ascontiguousarray(p, dtype=ctypes.c_float),
            np.ascontiguousarray(q, dtype=ctypes.c_float),
            np.ascontiguousarray(a, dtype=ctypes.c_float),
            np.ascontiguousarray(b, dtype=ctypes.c_float),
            epsilon, lambda1, lambda2, m, n)
    else:
        return lib.compute_duality_gap_double(
            C, #np.ascontiguousarray(C, dtype=ctypes.c_double),
            K, #np.ascontiguousarray(K, dtype=ctypes.c_double),
            R, #np.ascontiguousarray(R, dtype=ctypes.c_double),
            dx, #np.ascontiguousarray(dx, dtype=ctypes.c_double),
            dy, #np.ascontiguousarray(dy, dtype=ctypes.c_double),
            p, #np.ascontiguousarray(p, dtype=ctypes.c_double),
            q, #np.ascontiguousarray(q, dtype=ctypes.c_double),
            a, #np.ascontiguousarray(a, dtype=ctypes.c_double),
            b, #np.ascontiguousarray(b, dtype=ctypes.c_double),
            epsilon, lambda1, lambda2, m, n)

def update_K_c(K, _K, C, u, v, epsilon, use_float=False):
    m, n = C.shape

    if use_float:
        lib.update_k_float(
            K,
            _K,
            C,
            u,
            v,
            epsilon,
            m,
            n
        )

    else:
        lib.update_k_double(
            K,
            _K,
            C,
            u,
            v,
            epsilon,
            m,
            n
        )

def update_R_c(R, K, a, b, use_float=False):
    m, n = K.shape

    if use_float:
        lib.update_R_float(
            R,
            K,
            a,
            b,
            m,
            n
        )

    else:
        lib.update_R_double(
            R,
            K,
            a,
            b,
            m,
            n
        )

def update_a_b_c(a, b, K, Kt, dx, dy, p, q, u, v, lambda1, lambda2, alpha1, alpha2, epsilon, use_float=False):
    m, n = K.shape

    if use_float:
        lib.update_a_b_float(
            a,
            b,
            K,
            Kt,
            dx,
            dy,
            p,
            q,
            u,
            v,
            lambda1,
            lambda2,
            alpha1,
            alpha2,
            epsilon,
            m, n
        )
    else:
        lib.update_a_b_double(
            a,
            b,
            K,
            Kt,
            dx,
            dy,
            p,
            q,
            u,
            v,
            lambda1,
            lambda2,
            alpha1,
            alpha2,
            epsilon,
            m, n
        )

def step1_process_c(
    a, b, old_a, old_b, K, C, dx, dy, p, q, u, v,
    cur_iter, max_iter, iters, tau, lambda1, lambda2, alpha1, alpha2, epsilon):
    m, n = K.shape

    return lib.step1_process_double(
            a,
            b,
            old_a,
            old_b,
            K,
            C,
            dx,
            dy,
            p,
            q,
            u,
            v,
            cur_iter,
            int(max_iter),
            iters,
            float(tau),
            lambda1,
            lambda2,
            alpha1,
            alpha2,
            epsilon,
            m,
            n
        )

def update_process_c(
    R, a, b, old_a, old_b, K, _K, C,
    dx, dy, p, q, u, v,
    epsilon_scaling, cur_epsilon_scaling, batch_size, epsilon, threshold,
    tau, lambda1, lambda2, alpha1, alpha2,
    cur_iter, max_iter
    ):
    m, n = K.shape

    return lib.update_process_double(
        R, a, b, old_a, old_b, K, _K, C,
        dx, dy, p, q, u, v,
        epsilon_scaling, cur_epsilon_scaling, batch_size, epsilon, threshold,
        float(tau), lambda1, lambda2, alpha1, alpha2,
        cur_iter, int(max_iter), m, n
    )
