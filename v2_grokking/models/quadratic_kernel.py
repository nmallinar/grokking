import torch
from . import euclidean_distances, euclidean_distances_M
import numpy as np
import scipy
import time

torch.set_default_dtype(torch.float64)

def quadratic_M(samples, centers, M):
    return 3*((samples @ M) @ centers.T)**2
    # return ((samples @ M) @ centers.T)


def quad_M_update(X, x, sol, P, batch_size=2,
                   centering=True, diag_only=False):
    M = 0.

    start = time.time()
    # num_samples = min(20000, len(X))
    # indices = np.random.randint(len(X), size=num_samples)
    # if len(X) > len(indices):
    #     x = X[indices, :]
    # else:
    #     x = X

    K = 3 * 2 * (X @ P @ x.T)**1
    # a1 = torch.from_numpy(sol.T).float()
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

    G = step2.reshape(-1, c, d)

    if centering:
        G_mean = torch.mean(G, axis=0).unsqueeze(0)
        G = G - G_mean
    M = 0.

    bs = batch_size
    batches = torch.split(G, bs)
    # for i in tqdm(range(len(batches))):
    for i in range(len(batches)):
        if torch.cuda.is_available():
            grad = batches[i].cuda()
        else:
            grad = batches[i]

        gradT = torch.transpose(grad, 1, 2)
        if diag_only:
            T = torch.sum(gradT * gradT, axis=-1)
            M += torch.sum(T, axis=0).cpu()
        else:
            #gradT = torch.swapaxes(grad, 1, 2)#.cuda()
            M += torch.sum(gradT @ grad, dim=0).cpu()
        del grad, gradT
    torch.cuda.empty_cache()
    M /= len(G)


    if diag_only:
        M = torch.diag(M)


    U, s, Vt = torch.linalg.svd(M)
    depth = 2
    alpha = (depth-1) / (depth)
    alpha = 0.5
    s = torch.pow(torch.abs(s), alpha)
    M = U @ torch.diag(s) @ Vt

    # M = np.real(scipy.linalg.sqrtm(M))
    M = M.numpy()
    # D = np.diag(1/np.sqrt(np.diag(M)))
    # M = D @ M @ D
    # p = 31
    # M[:p, :p] = np.eye(p)
    # M[p:2*p, p:2*p] = np.eye(p)

    end = time.time()
    # print(M.max(), M.min(), M.mean())

    #print("Time: ", end - start)
    return torch.from_numpy(M), torch.zeros(c, c)
