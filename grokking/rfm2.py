import argparse
import numpy as np
import torch
from numpy.linalg import solve
import classic_kernel
from tqdm import tqdm
import math
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
#import hickle
import scipy
from inf_ntk import ntk_fn, jax_ntk_fn, ep3_ntk_relu
from sklearn.metrics import top_k_accuracy_score
from data import get_augmented_data_with_agop_loader

def ntk_M(pair1, pair2, depth, M):
    _, ntk = ntk_fn(pair1, pair2, M=M, depth=int(depth), bias=0, jax_rescale=True)
    return ntk

def jax_ntk_M(pair1, pair2, depth, M):
    _, ntk = jax_ntk_fn(pair1, pair2, M, depth=int(depth))
    return ntk

def laplace_kernel_M(pair1, pair2, bandwidth, M):
    return classic_kernel.laplacian_M(pair1, pair2, bandwidth, M)

def gaussian_kernel_M(pair1, pair2, bandwidth, M):
    return classic_kernel.gaussian_M(pair1, pair2, bandwidth, M)

def kernel_fn(pair1, pair2, bandwidth, M):
    # overloading bandwidth param to be depth in NTK
    # return ntk_M(pair1, pair2, bandwidth, M)
    # return jax_ntk_M(pair1, pair2, bandwidth, M)
    # return laplace_kernel_M(pair1, pair2, bandwidth, M)
    #return gaussian_kernel_M(pair1, pair2, bandwidth, M)
    sqrtM = np.real(scipy.linalg.sqrtm(M))
    return ep3_ntk_relu(pair1 @  sqrtM, pair2 @ sqrtM, depth=bandwidth)

def get_grads(X, sol, L, P, batch_size=2):
    M = 0.

    num_samples = 20000
    indices = np.random.randint(len(X), size=num_samples)

    if len(X) > len(indices):
        x = X[indices, :]
    else:
        x = X

    K = kernel_fn(X, x, L, P)

    dist = classic_kernel.euclidean_distances_M(X, x, P, squared=False)
    dist = torch.where(dist < 1e-10, torch.zeros(1).float(), dist)

    K = K/dist
    K[K == float("Inf")] = 0.

    a1 = torch.from_numpy(sol.T)
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

    a2 = torch.from_numpy(sol)
    step3 = (a2 @ K).T

    del K, a2

    step3 = step3.reshape(m, c, 1)
    x1 = (x @ P).reshape(m, 1, d)
    step3 = step3 @ x1

    G = (step2 - step3) * -1/L

    M = 0.

    bs = batch_size
    batches = torch.split(G, bs)
    for i in range(len(batches)):
    # for i in tqdm(range(len(batches))):
        # grad = batches[i].cuda()
        grad = batches[i]
        gradT = torch.transpose(grad, 1, 2)
        M += torch.sum(gradT @ grad, dim=0).cpu()
        del grad, gradT
    torch.cuda.empty_cache()
    M /= len(G)
    M = np.real(scipy.linalg.sqrtm(M.numpy()))

    return M

def get_grad_reg(X, L, P):
    M = 0.
    num_samples = X.size(0)
    indices = np.random.randint(len(X), size=num_samples)

    if len(X) > len(indices):
        x = X[indices, :]
    else:
        x = X

    K = kernel_fn(X, x, L, P)

    dist = classic_kernel.euclidean_distances_M(X, x, P, squared=False)
    dist = torch.where(dist < 1e-10, torch.zeros(1).float(), dist)

    K = K/dist
    K[K == float("Inf")] = 0.

    # n x n x d
    all_diffs = torch.zeros(X.size(0), X.size(0), X.size(1))
    for i in range(X.size(0)):
        all_diffs[i] = X - X[i]

    G = 0.
    for i in range(X.size(0)):
        G += all_diffs[i] @ all_diffs[i].T
    # G /= X.size(0)

    return G * 1./L

def rfm(X_train, y_train_onehot, X_test, y_test_onehot, num_classes, wandb,
        iters=3, name=None, batch_size=2, reg=1e-3,
        train_acc=False, L=1, agop_weight=1e-5):
    n, d = X_train.shape
    y_train = y_train_onehot.argmax(-1)
    y_test = y_test_onehot.argmax(-1)

    M = np.eye(d, dtype='float64')
    for i in range(iters):
        K_train = kernel_fn(X_train, X_train, L, torch.from_numpy(M)).numpy()
        sol = solve(K_train + reg * np.eye(len(K_train)), y_train_onehot).T

        # G_reg = get_grad_reg(X_train, L, torch.from_numpy(M)).numpy()
        # sol = np.linalg.inv(K_train.T @ K_train + agop_weight*G_reg) @ K_train @ y_train.numpy()
        # sol = sol.T

        if train_acc:
            preds = (sol @ K_train).T
            loss = np.mean(np.square(preds - y_train_onehot.numpy()))
            y_pred = torch.from_numpy(preds)
            preds = torch.argmax(y_pred, dim=-1)
            count = torch.sum(y_train == preds).numpy()
            acc = count / len(y_train)

            #if i % 50 == 0:
            print("Round " + str(i) + " Train MSE: ", loss) # Loss function
            print("Round " + str(i) + " Train Acc: ", acc)

            metrics = {
                'training/accuracy': acc,
                'training/loss': loss,
                'rfm_iter': i
            }
            if wandb is not None:
                wandb.log(metrics)

        # if i % 50 == 0:
        top_k_val = 5
        K_test = kernel_fn(X_train, X_test, L, torch.from_numpy(M)).numpy()
        preds = (sol @ K_test).T
        loss = np.mean(np.square(preds - y_test_onehot.numpy()))
        print("Round " + str(i) + " MSE: ", loss) # Loss function
        y_pred = torch.from_numpy(preds)
        preds = torch.argmax(y_pred, dim=-1)
        top_k_acc = top_k_accuracy_score(y_test, y_pred, k=top_k_val)

        count = torch.sum(y_test == preds).numpy()
        acc = count / len(y_test)
        print("Round " + str(i) + " Acc: ", acc)
        print("Round " + str(i) + f" Top {top_k_val} Acc: ", top_k_acc)
        print()

        metrics = {
            'validation/accuracy': acc,
            'validation/loss': loss,
            'rfm_iter': i
        }
        if wandb is not None:
            wandb.log(metrics)

        M  = get_grads(X_train, sol, L, torch.from_numpy(M), batch_size=batch_size)

    K_train = kernel_fn(X_train, X_train, L, torch.from_numpy(M)).numpy()
    sol = solve(K_train + reg * np.eye(len(K_train)), y_train_onehot).T
    K_test = kernel_fn(X_train, X_test, L, torch.from_numpy(M)).numpy()
    preds = (sol @ K_test).T
    mse = np.mean(np.square(preds - y_test_onehot.numpy()))
    print("Final MSE: ", mse)
    y_pred = torch.from_numpy(preds)
    preds = torch.argmax(y_pred, dim=-1)
    count = torch.sum(y_test == preds).numpy()
    print(" Final Acc: ", count / len(y_test))
    return mse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', '-op', default="x/y")
    parser.add_argument('--prime', '-p', default=3, type=int)
    parser.add_argument('--num_tokens', '-n', default=31, type=int)
    parser.add_argument('--training_fraction', default=0.5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--agop_batch_size', default=32, type=int)
    parser.add_argument('--iters', default=3, type=int)
    parser.add_argument('--rfm_batch_size', default=2, type=int)
    parser.add_argument('--ridge', default=1e-3, type=float)
    parser.add_argument('--bandwidth', default=1, type=float)
    parser.add_argument('--agop_weight', default=1e-5, type=float)
    args = parser.parse_args()

    train_loader, agop_loader, val_loader, val_loader1, \
        context_len, train_dataset, val_dataset, \
        X_train, y_train, X_test, y_test = \
            get_augmented_data_with_agop_loader(args.operation, args.prime, args.num_tokens,
                                                args.training_fraction, args.batch_size,
                                                args.agop_batch_size, drop_last=False)

    X_train = F.one_hot(X_train, args.num_tokens).view(-1, 2*args.num_tokens).double()
    y_train = F.one_hot(y_train, args.prime).double()
    X_test = F.one_hot(X_test, args.num_tokens).view(-1, 2*args.num_tokens).double()
    y_test = F.one_hot(y_test, args.prime).double()

    num_classes = args.prime
    wandb = None
    rfm(X_train, y_train, X_test, y_test, num_classes, wandb,
        iters=args.iters, name=None, batch_size=args.rfm_batch_size, reg=args.ridge,
        train_acc=True, L=args.bandwidth, agop_weight=args.agop_weight)
