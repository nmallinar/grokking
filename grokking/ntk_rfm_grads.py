from inf_ntk import ep3_ntk_relu
from scipy.linalg import sqrtm
import numpy as np
import torch

def get_jacs(X, x, ntk_depth, sqrtM=None):
    if sqrtM is not None:
        X_M = X@sqrtM
    else:
        X_M = X

    def egop_fn(z):
        if sqrtM is not None:
            z_ = z@sqrtM
        else:
            z_ = z
        K = ep3_ntk_relu(z_, X_M, M=None, depth=ntk_depth)
        return K
    grads = torch.vmap(torch.func.jacrev(egop_fn))(x.unsqueeze(1)).squeeze()
    grads = torch.nan_to_num(grads)
    return grads

# def get_jacs(X, x, ntk_depth, sqrtM=None):
#     if sqrtM is not None:
#         X_M = X @ sqrtM
#         x = x @ sqrtM
#     else:
#         X_M = X
#
#     jacs = torch.func.jacrev(ep3_ntk_relu, argnums=(1,))(x, X, ntk_depth)
#     import ipdb; ipdb.set_trace()

def get_grads(alphas, train_X, Xs, M, ntk_depth=1):

    M_is_passed = M is not None
    sqrtM = None
    if M_is_passed:
        # sqrtM = utils.matrix_sqrt(M)
        sqrtM = torch.tensor(np.real(sqrtm(M)))

    def get_solo_grads(sol, X, x):
        if M_is_passed:
            X_M = X@sqrtM
        else:
            X_M = X

        def egop_fn(z):
            if M_is_passed:
                z_ = z@sqrtM
            else:
                z_ = z
            K = ep3_ntk_relu(z_, X_M, M=None, depth=ntk_depth)
            return (K@sol).squeeze()
        grads = torch.vmap(torch.func.jacrev(egop_fn))(x.unsqueeze(1)).squeeze()
        grads = torch.nan_to_num(grads)
        return grads

    n, d = train_X.shape
    s = len(Xs)

    chunk = 1000
    train_batches = torch.split(torch.arange(n), chunk)

    egop = 0
    G = 0
    for btrain in train_batches:
        G += get_solo_grads(alphas[btrain,:], train_X[btrain], Xs)
    G = G.reshape(-1, d)
    egop += G.T @ G/s

    return egop
