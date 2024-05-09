import torch
from scipy.linalg import sqrtm
import numpy as np

torch.set_default_dtype(torch.float64)

def ntk_relu(X, Z, depth=1, bias=0., M=None):
    """
    Returns the evaluation of nngp and ntk kernels
    for fully connected neural networks
    with ReLU nonlinearity.

    depth  (int): number of layers of the network
    bias (float): (default=0.)
    """
    if M is not None:
        sqrtM = torch.tensor(np.real(sqrtm(M)))
        X = X @ M
        Z = Z @ M

    eps = 1e-12
    from torch import acos, pi
    kappa_0 = lambda u: (1-acos(u)/pi)
    kappa_1 = lambda u: u*kappa_0(u) + (1-u.pow(2)).sqrt()/pi
    Z = Z if Z is not None else X
    norm_x = X.norm(dim=-1)[:, None].clip(min=eps)
    norm_z = Z.norm(dim=-1)[None, :].clip(min=eps)
    S = X @ Z.T
    N = S + bias**2
    for k in range(1, depth):
        in_ = (S/norm_x/norm_z).clip(min=-1+eps,max=1-eps)
        S = norm_x*norm_z*kappa_1(in_)
        N = N * kappa_0(in_) + S + bias**2
    return N

def ntk_relu_M_update(alphas, centers, samples, M, ntk_depth=1):

    M_is_passed = M is not None
    sqrtM = None
    if M_is_passed:
        # sqrtM = utils.matrix_sqrt(M)
        sqrtM = torch.tensor(np.real(sqrtm(M)))

    def get_solo_grads(sol, X, x):
        if M_is_passed:
            X_M = X@M
        else:
            X_M = X

        def egop_fn(z):
            if M_is_passed:
                z_ = z@M
            else:
                z_ = z
            K = ntk_relu(z_, X_M, M=None, depth=ntk_depth)
            return (K@sol).squeeze()
        grads = torch.vmap(torch.func.jacrev(egop_fn))(x.unsqueeze(1)).squeeze()
        grads = torch.nan_to_num(grads)
        return grads

    n, d = centers.shape
    s = len(samples)

    chunk = 1000
    train_batches = torch.split(torch.arange(n), chunk)

    egop = 0
    egip = 0
    G = 0
    for btrain in train_batches:
        G += get_solo_grads(alphas[btrain,:], centers[btrain], samples)
    c = G.shape[1]

    Gd = G.reshape(-1, d)
    egop += Gd.T @ Gd/s

    egip_fake = torch.zeros((c, c))

    return torch.from_numpy(np.real(sqrtm(egop))), egip_fake
    return egop, egip_fake

def get_jacs(centers, samples, ntk_depth, M=None):
    if M is not None:
        sqrtM = torch.tensor(np.real(sqrtm(M)))
        centers_M = centers@sqrtM
    else:
        sqrtM = None
        centers_M = centers

    def egop_fn(z):
        if sqrtM is not None:
            z_ = z@sqrtM
        else:
            z_ = z
        K = ep3_ntk_relu(z_, centers_M, M=None, depth=ntk_depth)
        return K

    grads = torch.vmap(torch.func.jacrev(egop_fn))(samples.unsqueeze(1)).squeeze()
    grads = torch.nan_to_num(grads)
    return grads
