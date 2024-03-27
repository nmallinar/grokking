import torch
from . import euclidean_distances, euclidean_distances_M
import numpy as np
import scipy

torch.set_default_dtype(torch.float64)

def gaussian(samples, centers, bandwidth, return_dist=False):
    '''Gaussian kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers)

    if return_dist:
        dist = kernel_mat.clone()

    kernel_mat.clamp_(min=0)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()

    #print(samples.size(), centers.size(),
    #      kernel_mat.size())

    if return_dist:
        return kernel_mat, dist

    return kernel_mat

def gaussian_M(samples, centers, bandwidth, M, return_dist=False):
    assert bandwidth > 0
    kernel_mat = euclidean_distances_M(samples, centers, M, squared=True)

    if return_dist:
        dist = kernel_mat.clone()

    kernel_mat.clamp_(min=0)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()

    if return_dist:
        return kernel_mat, dist

    return kernel_mat

def get_grads(X, sol, L, P, batch_size=2, K=None, centering=False):
    M = 0.

    x = X

    if K is None:
        K = gaussian_M(X, x, L, P)


    a1 = sol.T
    n, d = X.shape
    n, c = a1.shape
    m, d = x.shape

    a1 = a1.reshape(n, c, 1)
    X1 = (X @ P).reshape(n, 1, d)
    step1 = a1 @ X1
    del a1, X1
    step1 = step1.reshape(-1, c*d)

    step2 = K.T @ step1
    del step1

    step2 = step2.reshape(-1, c, d)

    a2 = sol
    step3 = (a2 @ K).T

    del K, a2

    step3 = step3.reshape(m, c, 1)
    x1 = (x @ P).reshape(m, 1, d)
    step3 = step3 @ x1

    G = (step2 - step3) * -1/(L**2)

    M = 0.
    Mc = 0.

    if centering:
        G = G - G.mean(0)

    bs = batch_size
    batches = torch.split(G, bs)
    for i in range(len(batches)):
    # for i in tqdm(range(len(batches))):
        # grad = batches[i].cuda()
        grad = batches[i]
        gradT = torch.transpose(grad, 1, 2)
        M += torch.sum(gradT @ grad, dim=0).cpu()
        Mc += torch.sum(grad @ gradT, dim=0).cpu()
        del grad, gradT
    torch.cuda.empty_cache()
    M /= len(G)
    M = np.real(scipy.linalg.sqrtm(M.numpy()))

    Mc /= len(G)
    # Mc = np.real(scipy.linalg.sqrtm(Mc.numpy()))
    Mc = Mc.numpy()

    # return M, torch.tensor(Mc)
    return torch.from_numpy(M), torch.from_numpy(Mc)

def gaussian_M_update(samples, centers, bandwidth, M, weights, K=None, \
                      centers_bsize=-1, centering=False):
    return get_grads(samples, weights.T, bandwidth, M, K=K, centering=centering)

def get_jac_reg(samples, centers, bandwidth, M, K=None, \
                centering=False):
    '''
    K(x, y) = exp(-sqrt((x-y).T @ M @ (x - y)) / L)
    grad_x K(x, y) =  * -(K(x, y) / L*sqrt((x-y).T @ M @ (x - y))) * M (x - y) \in \R^{d}

    G(x)_i \in \R^{n \times d} = grad_x K(x, x_i).T

    GG^T \in \R^{n \times n}
    '''

    if K is None:
        K = gaussian_M(samples, centers, bandwidth, M)

    # samples: n x d, centers: m x d
    samples = samples.unsqueeze(0)
    centers = centers.unsqueeze(1)

    # all_diffs: n x m x d
    all_diffs = samples - centers
    del samples, centers

    all_diffs = all_diffs @ M
    all_diffs /= (bandwidth**2)

    K = K.unsqueeze(1) * K.unsqueeze(2)

    # all_diffs = K.unsqueeze(2) * all_diffs
    def outer_prod(x):
        return x@x.T

    G = 0.
    for idx in range(0, all_diffs.size(0), 32):
        prod = torch.vmap(outer_prod)(all_diffs[idx:idx+32])
        # check that prod: (32, n, n) and K[idx:idx+32]: (32, n, n)
        G += torch.sum(prod * K[idx:idx + 32], dim=0)

    return G
