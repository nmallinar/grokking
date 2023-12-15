import torch
import math
import numpy as np
import scipy

def clip(u, eps=1e-10):
    return torch.clip(u, -1+eps, 1-eps)

def arc_cos_0_kernel(u):
    return 1./math.pi * (math.pi - torch.acos(clip(u)))

def arc_cos_1_kernel(u):
    return 1./math.pi * (u*(math.pi - torch.acos(clip(u))) + torch.sqrt(1 - torch.pow(clip(u), 2)))

def ntk_fn(x, y, M=None, depth=1, bias=0, jax_rescale=False):
    '''
    Feed-forward network with depth-1 hidden layers,
    e.g. depth = 2 ---> 1 hidden layer network
         f(x) = W_2*sqrt(2/h)*ReLU(W_1 * x + bias) + bias

    Inputs:
    x: (n, d)
    y: (m, D)
    M: (d, D)

    jax_rescale from empirical observations:
        divide this NTK/NNGP by 3*(2^(depth-1)) in order to obtain an NTK
        within atol ~= 1e-4 of the Jax NTK/NNGP, tolerance improves dramatically
        with higher depth networks, e.g. depth=20 -> atol ~= 1e-10

    Output:
    Sk: (n, m) NNGP
    Nk: (n, m) NTK
    '''
    if M is None:
        u = x @ y.t()
        v = x @ x.t()
        w = y @ y.t()
    else:
        u = x @ M @ y.t()
        v = (x * (x @ M)).sum(dim=-1).sqrt()
        w = (y * (y @ M)).sum(dim=-1).sqrt()
    norms = v.view(-1, 1) * w.view(-1)

    N_k_minus_1 = u + bias**2
    S_k_minus_1 = u

    for k in range(1, depth):
        Sk = norms*arc_cos_1_kernel(S_k_minus_1 / norms)
        Nk = Sk + N_k_minus_1*arc_cos_0_kernel(S_k_minus_1 / norms) + bias**2

        S_k_minus_1 = Sk
        N_k_minus_1 = Nk

    if not jax_rescale:
        return S_k_minus_1, N_k_minus_1

    scale_factor = 3*(2**(depth-1))
    return S_k_minus_1 / scale_factor, N_k_minus_1 / scale_factor

def test_ntk():
    '''
    Observations on rescale factors between this PyTorch NTK/NNGP and Jax NTK/NNGP
    for the roughly equivalent network (no bias):

        depth = 2, rescale = 6 (3 * 2)
        depth = 3, rescale = 12 (3 * 2 * 2)
        depth = 4, rescale = 24 (3 * 2 * 2 * 2)
        depth = 5, rescale = 48 (3 * 2 * 2 * 2 * 2)

    This rescale factor becomes more precise as depth increases
    '''

    depth = 20 # means depth - 1 hidden layers
    bias = 0
    n = 10
    d = 3

    X = torch.randn(n, d)
    # M = torch.eye(d, d)
    M = torch.randn(d, d)

    # use M @ M.T here so when applying to input data in
    # jax NTK function we can simply take M = sqrtm(M @ M.T)
    # and use data X @ M
    nngp_torch, ntk_torch = ntk_fn(X, X, M=M@M.T, depth=depth, bias=bias)
    nngp_torch_scaled, ntk_torch_scaled = ntk_fn(X, X, M=M@M.T, depth=depth, bias=bias, jax_rescale=True)


    from neural_tangents import stax
    import jax.dlpack

    X = X @ M
    X = torch.to_dlpack(X)
    X = jax.dlpack.from_dlpack(X)

    layers = []
    for _ in range(1, depth):
        layers += [
            stax.Dense(1, W_std=1, b_std=bias),
            stax.Relu()
        ]
    layers.append(stax.Dense(1, W_std=1, b_std=bias))

    init_fn, apply_fn, kernel_fn = stax.serial(*layers)

    ntk_nngp_jax = kernel_fn(X, X)

    ntk_jax = jax.dlpack.to_dlpack(ntk_nngp_jax.ntk)
    ntk_jax = torch.utils.dlpack.from_dlpack(ntk_jax)
    nngp_jax = jax.dlpack.to_dlpack(ntk_nngp_jax.nngp)
    nngp_jax = torch.utils.dlpack.from_dlpack(nngp_jax)

    rescale = ntk_torch[0][0] / ntk_jax[0][0]
    manually_rescaled_ntk_torch = ntk_torch / rescale

    print('NTK:')
    print(torch.allclose(manually_rescaled_ntk_torch, ntk_jax, atol=1e-4))
    print(rescale)

    diff_kernel = ntk_jax - manually_rescaled_ntk_torch
    print(diff_kernel)

    rescale = nngp_torch[0][0] / nngp_jax[0][0]
    manually_rescaled_nngp_torch = nngp_torch / rescale

    print('\nNNGP:')
    print(torch.allclose(manually_rescaled_nngp_torch, nngp_jax, atol=1e-4))
    print(rescale)

    diff_kernel = nngp_jax - manually_rescaled_nngp_torch
    print(diff_kernel)

    print('\nrescale diffs between exact (manual) scale factor and 3*(2^(depth-1))')
    print(manually_rescaled_ntk_torch - ntk_torch_scaled)
    print(manually_rescaled_nngp_torch - nngp_torch_scaled)

def jax_ntk_fn(x, y, M=None, depth=2, bias=0):
    from neural_tangents import stax
    import jax.dlpack

    if M is not None:
        sqrt_M = scipy.linalg.sqrtm(M)
        x = x @ sqrt_M
        y = y @ sqrt_M

    x = torch.to_dlpack(x)
    x = jax.dlpack.from_dlpack(x)
    y = torch.to_dlpack(y)
    y = jax.dlpack.from_dlpack(y)

    layers = []
    for _ in range(1, depth):
        layers += [
            stax.Dense(1, W_std=1, b_std=bias),
            stax.Relu()
        ]
    layers.append(stax.Dense(1, W_std=1, b_std=bias))

    _, _, kernel_fn = stax.serial(*layers)

    ntk_nngp_jax = kernel_fn(x, y)

    ntk_jax = jax.dlpack.to_dlpack(ntk_nngp_jax.ntk)
    ntk_jax = torch.utils.dlpack.from_dlpack(ntk_jax)
    nngp_jax = jax.dlpack.to_dlpack(ntk_nngp_jax.nngp)
    nngp_jax = torch.utils.dlpack.from_dlpack(nngp_jax)

    return nngp_jax, ntk_jax

if __name__=='__main__':
    test_ntk()
