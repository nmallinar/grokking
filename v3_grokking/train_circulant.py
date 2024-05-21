import os
import sys
import argparse
import wandb
import torch
import torch.nn.functional as F
import numpy as np
import scipy
import random

from data import operation_mod_p_data, make_data_splits
from models import laplace_kernel, gaussian_kernel, torch_fcn_relu_ntk, jax_fcn_relu_ntk, quadratic_kernel, euclidean_distances_M
import utils

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.manual_seed(3143)
random.seed(253)
np.random.seed(1145)

def eval(sol, K, y_onehot):
    preds = K.T @ sol
    loss = (preds - y_onehot).pow(2).mean()

    corr = 0
    if y_onehot.shape[1] > 1:
        count = torch.sum(y_onehot.argmax(-1) == preds.argmax(-1))
        acc = count / y_onehot.shape[0]
    else:
        acc = 0.0

    return acc, loss, corr

def get_test_kernel(X_tr, X_te, M, bandwidth, ntk_depth, kernel_type):
    K_test = None
    if kernel_type == 'laplace':
        K_test = laplace_kernel.laplacian_M(X_tr, X_te, bandwidth, M, return_dist=False)
    elif kernel_type == 'gaussian':
        K_test = gaussian_kernel.gaussian_M(X_tr, X_te, bandwidth, M)
    elif kernel_type == 'fcn_relu_ntk':
        K_test = torch_fcn_relu_ntk.ntk_relu(X_tr, X_te, depth=ntk_depth, bias=0., M=M)
    elif kernel_type == 'jax_fcn_ntk':
        _, K_test = jax_fcn_relu_ntk.ntk_fn(X_tr, X_te, M=M, depth=ntk_depth, bias=0, convert=True)
    elif kernel_type == 'quadratic':
        K_test = quadratic_kernel.quadratic_M(X_tr, X_te, M)
    elif kernel_type == 'linear':
        K_test = euclidean_distances_M(X_tr, X_te, M)

    return K_test

def solve(X_tr, y_tr_onehot, M, Mc, bandwidth, ntk_depth, kernel_type,
          ridge=1e-3, jac_reg_weight=0.0, agip_rdx_weight=0.0, use_k_inv=True):

    K_train = None
    dist = None
    sol = None

    if kernel_type == 'laplace':
        K_train, dist = laplace_kernel.laplacian_M(X_tr, X_tr, bandwidth, M, return_dist=True)
        if jac_reg_weight > 0:
            jac = laplace_kernel.get_jac_reg(X_tr, X_tr, bandwidth, M, K=K_train, dist=dist)
    elif kernel_type == 'gaussian':
        K_train = gaussian_kernel.gaussian_M(X_tr, X_tr, bandwidth, M)
        if jac_reg_weight > 0:
            jac = gaussian_kernel.get_jac_reg(X_tr, X_tr, bandwidth, M, K=K_train)
    elif kernel_type == 'fcn_relu_ntk':
        K_train = torch_fcn_relu_ntk.ntk_relu(X_tr, X_tr, depth=ntk_depth, bias=0., M=M)
        if jac_reg_weight > 0:
            raise Exception('to do')
    elif kernel_type == 'jax_fcn_ntk':
        _, K_train = jax_fcn_relu_ntk.ntk_fn(X_tr, X_tr, M=M, depth=ntk_depth, bias=0, convert=True)
        if jac_reg_weight > 0:
            jac = jax_fcn_relu_ntk.get_jac_reg(X_tr, X_tr, bandwidth, M, ntk_depth, K=K_train)
    elif kernel_type == 'quadratic':
        K_train = quadratic_kernel.quadratic_M(X_tr, X_tr, M)
        if jac_reg_weight > 0:
            raise Exception()
    elif kernel_type == 'linear':
        K_train = euclidean_distances_M(X_tr, X_tr, M)
        if jac_reg_weight > 0:
            raise Exception()

    if agip_rdx_weight > 0:
        if jac_reg_weight > 0:
            raise Exception('to do - derive the proper expression for this')
        else:
            S, U = torch.linalg.eigh(K_train + ridge * torch.eye(len(K_train)))
            R, V = torch.linalg.eigh(agip_rdx_weight*Mc)
            labels = y_tr_onehot
            sol = U @((U.T @ labels @ V)/(R + S.unsqueeze(1))) @ V.T
            return sol, K_train, dist
    else:
        if jac_reg_weight > 0:
            if use_k_inv:
                sol = torch.from_numpy(np.linalg.inv((K_train.T @ K_train + ridge * np.eye(len(K_train)) \
                                                      + jac_reg_weight * jac).numpy())) @ y_tr_onehot
            else:
                sol = torch.from_numpy(np.linalg.inv((K_train.T @ K_train + ridge * np.eye(len(K_train)) \
                                                      + jac_reg_weight * jac).numpy())) @ K_train @ y_tr_onehot
        else:
            if use_k_inv:
                sol = torch.from_numpy(np.linalg.solve(K_train.T @ K_train.numpy() + ridge * np.eye(len(K_train)), y_tr_onehot.numpy()).T)
                sol = sol.T
                preds = K_train @ sol
            else:
                sol = torch.from_numpy(np.linalg.solve(K_train.numpy() + ridge * np.eye(len(K_train)), y_tr_onehot.numpy()).T)
                sol = sol.T
                preds = K_train @ sol

    return sol, K_train, dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='./')
    parser.add_argument('--operation', '-op', default="x+y")
    parser.add_argument('--prime', '-p', default=97, type=int)
    parser.add_argument('--training_fraction', default=0.5, type=float)
    parser.add_argument('--ridge', default=1e-3, type=float)
    parser.add_argument('--bandwidth', default=1, type=float)
    parser.add_argument('--ntk_depth', default=2, type=int)
    parser.add_argument('--jac_reg_weight', default=0, type=float)
    parser.add_argument('--agip_rdx_weight', default=0, type=float)
    parser.add_argument('--kernel_type', default='gaussian', choices={'linear', 'gaussian', 'laplace', 'fcn_relu_ntk', 'quadratic', 'jax_fcn_ntk'})
    parser.add_argument('--use_k_inv', default=False, action='store_true')
    args = parser.parse_args()

    all_inputs, all_labels = operation_mod_p_data(args.operation, args.prime)
    X_tr, y_tr, X_te, y_te = make_data_splits(all_inputs, all_labels, args.training_fraction)

    proj_dim = args.prime
    X_tr = F.one_hot(X_tr, args.prime).view(-1, 2*args.prime).double()
    y_tr_onehot = F.one_hot(y_tr, args.prime).double()
    X_te = F.one_hot(X_te, args.prime).view(-1, 2*args.prime).double()
    y_te_onehot = F.one_hot(y_te, args.prime).double()

    # proj_dim = 200
    # w1 = torch.randn(args.prime, proj_dim)
    # X_tr = (F.one_hot(X_tr, args.prime).double() @ w1).view(-1, 2*proj_dim)
    # X_te = (F.one_hot(X_te, args.prime).double() @ w1).view(-1, 2*proj_dim)
    # # X_tr = torch.cat((torch.cos(X_tr), torch.sin(X_tr)), dim=1)
    # # X_te = torch.cat((torch.cos(X_te), torch.sin(X_te)), dim=1)
    # # proj_dim *= 2
    # y_tr_onehot = F.one_hot(y_tr, args.prime).double()
    # y_te_onehot = F.one_hot(y_te, args.prime).double()

    # M = torch.eye(X_tr.shape[1]).double()
    M = torch.zeros(X_tr.shape[1], X_tr.shape[1]).double()
    Mc = torch.eye(y_tr_onehot.shape[1]).double()

    # M[:proj_dim,:proj_dim] = 1./(proj_dim)*torch.eye(proj_dim) - 1./(proj_dim)*torch.ones(proj_dim, proj_dim)
    # M[:proj_dim,:proj_dim] += torch.eye(proj_dim)
    # M[proj_dim:,proj_dim:] = M[:proj_dim,:proj_dim].clone()

    # col1 = torch.rand(5)
    # test1 = torch.normal(0, 1.0, size=(5,))
    # test2 = torch.normal(0, 1.0, size=(5,))
    # test3 = torch.normal(0, 1.0, size=(5,))
    # test4 = torch.normal(0, 1.0, size=(5,))
    # col = torch.cat((col1 + test1, col1 + test2, col1 + test3, col1 + test4))[:19]
    # col = torch.rand(args.prime)*2
    col = torch.randn(proj_dim)
    # col = torch.normal(0, 1, size=(proj_dim,))
    # col = [np.cos(2*np.pi*i/proj_dim) for i in range(int((proj_dim)))]
            # [np.sin(2*np.pi*i/proj_dim) for i in range(int((proj_dim-1)/2))]
    # col = torch.tensor(col) * 2
    # print(col)
    # col = torch.arange(0, 1, (1. / proj_dim))
    circ = torch.from_numpy(scipy.linalg.circulant(col.numpy()))
    # circ = torch.from_numpy(scipy.linalg.toeplitz(col.numpy()))
    # circ = torch.from_numpy(np.load('notebooks/circ2.npy')).double()

    if args.operation == 'x-y':
        M[:proj_dim,proj_dim:] = torch.rot90(circ)
    elif args.operation == 'x+y':
        M[:proj_dim,proj_dim:] = circ

    M[proj_dim:,:proj_dim] = M[:proj_dim,proj_dim:].clone().T
    # M = torch.from_numpy(np.load('../v2_grokking/saved_agops/relu_small_init_p19/right_nfm.npy')).double()
    M = torch.from_numpy(np.real(scipy.linalg.sqrtm(M)))

    sol, K_train, dist = solve(X_tr, y_tr_onehot, M, Mc, args.bandwidth, args.ntk_depth, args.kernel_type,
                               ridge=args.ridge, jac_reg_weight=args.jac_reg_weight, agip_rdx_weight=args.agip_rdx_weight,
                               use_k_inv=args.use_k_inv)
    acc, loss, corr = eval(sol, K_train, y_tr_onehot)
    print(f'Train MSE:\t{loss}')
    print(f'Train Acc:\t{acc}')

    K_test = get_test_kernel(X_tr, X_te, M, args.bandwidth, args.ntk_depth, args.kernel_type)

    acc, loss, corr = eval(sol, K_test, y_te_onehot)
    print(f'Test MSE:\t{loss}')
    print(f'Test Acc:\t{acc}')
    print()

if __name__=='__main__':
    main()
