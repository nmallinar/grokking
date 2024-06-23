import os
import sys
import argparse
import wandb
import torch
import torch.nn.functional as F
import numpy as np
import scipy
import random

from data import operation_mod_p_data, make_data_splits, held_out_op_mod_p_data, \
                 multitask_op_mod_p_data
from models import laplace_kernel, gaussian_kernel, torch_fcn_relu_ntk, \
                   jax_fcn_relu_ntk, quadratic_kernel
import utils

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.manual_seed(3143)
random.seed(253)
np.random.seed(1145)
#
# def eval2(sol, K, y_onehot, p):
#     preds = K.T @ sol
#     # preds = preds[:,p:]
#     # y_onehot = y_onehot[:,p:]
#     loss = (preds - y_onehot).pow(2).mean()
#     mask = preds > 0.5
#     preds[mask] = 1
#     preds[~mask] = 0
#     acc = torch.mean((preds == y_onehot).double())
#
#     return acc, loss

def eval(sol, K, y_onehot, p, first_p=True, eval_all=False):
    preds = K.T @ sol
    loss = (preds - y_onehot).pow(2).mean()

    corr = 0
    if y_onehot.shape[1] > 1:
        if eval_all:
            count = torch.sum(y_onehot.argmax(-1) == preds.argmax(-1))
        else:
            if first_p:
                count = torch.sum(y_onehot[:,:p].argmax(-1) == preds[:,:p].argmax(-1))
            else:
                count = torch.sum(y_onehot[:,p:].argmax(-1) == preds[:,p:].argmax(-1))
        acc = count / y_onehot.shape[0]
    elif y_onehot.shape[1] == 1 or len(y_onehot.shape) == 1:
        count = torch.sum((y_onehot > 0.5) == (preds > 0.5))
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
    elif kernel_type == 'general_quadratic':
        K_test = quadratic_kernel.general_quadratic_M(X_tr, X_te, M)

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
            jac = quadratic_kernel.get_jac_reg(X_tr, X_tr, bandwidth, M)
    elif kernel_type == 'general_quadratic':
        K_train = quadratic_kernel.general_quadratic_M(X_tr, X_tr, M)
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
                # K_inv = np.linalg.pinv(K_train)
                # # vals, vecs = np.linalg.eig(y_tr_onehot.T @ y_tr_onehot)
                # # vecs *= -0.5
                # # y_cov = vecs.T @ np.diag(vals) @ vecs
                # # Mc = y_cov @ y_tr_onehot.T.numpy() @ K_inv @ y_tr_onehot.numpy() @ y_cov
                # # Mc = y_tr_onehot.T.numpy() @ K_inv @ y_tr_onehot.numpy()
                # # sol = torch.from_numpy(np.linalg.solve(K_train.numpy() + ridge * np.eye(len(K_train)), (y_tr_onehot @ Mc / torch.max(y_tr_onehot @ Mc)).numpy())).T
                # new_labels = (K_inv @ y_tr_onehot.numpy())
                # new_labels /= np.max(new_labels)
                # sol = torch.from_numpy(np.linalg.solve(K_train.numpy() + ridge * np.eye(len(K_train)), new_labels).T)

                sol = torch.from_numpy(np.linalg.solve(K_train.T @ K_train.numpy() + ridge * np.eye(len(K_train)), y_tr_onehot.numpy()).T)
                sol = sol.T
                preds = K_train @ sol
            else:
                sol = torch.from_numpy(np.linalg.solve(K_train.numpy() + ridge * np.eye(len(K_train)), y_tr_onehot.numpy()).T)
                sol = sol.T
                preds = K_train @ sol

    return sol, K_train, dist

def update(samples, centers, bandwidth, M, weights, K, dist, \
           kernel_type, ntk_depth, centers_bsize=-1, centering=False,
           agop_power=0.5, agip_power=1.0):
    if kernel_type == 'laplace':
        M, Mc = laplace_kernel.laplacian_M_update(samples, centers, bandwidth, M, weights, K=K, dist=dist, \
                                   centers_bsize=centers_bsize, centering=centering)
    elif kernel_type == 'gaussian':
        M, Mc = gaussian_kernel.gaussian_M_update(samples, centers, bandwidth, M, weights, K=K, \
                              centers_bsize=centers_bsize, centering=centering, agop_power=agop_power,
                              agip_power=agip_power)
    elif kernel_type == 'fcn_relu_ntk':
        M, Mc = torch_fcn_relu_ntk.ntk_relu_M_update(weights, centers, samples, M, ntk_depth=ntk_depth)
    elif kernel_type == 'jax_fcn_ntk':
        M, Mc = jax_fcn_relu_ntk.ntk_relu_M_update(weights, centers, samples, M, ntk_depth=ntk_depth)
    elif kernel_type == 'quadratic':
        M, Mc = quadratic_kernel.quad_M_update(samples, centers, weights.T, M, centering=centering)
    elif kernel_type == 'general_quadratic':
        M, Mc = quadratic_kernel.general_quadratic_M_update(samples, centers, weights.T, M, centering=centering)

    return M, Mc

def compute_train_class_freqs(y_tr):
    return y_tr.T @ y_tr / y_tr.shape[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', default='belkinlab')
    parser.add_argument('--wandb_proj_name', default='mar22-rfm-grokking')
    parser.add_argument('--wandb_offline', default=False, action='store_true')
    parser.add_argument('--group_key', default='', type=str)
    parser.add_argument('--out_dir', default='./')
    parser.add_argument('--operation', '-op', default="x+y")
    parser.add_argument('--prime', '-p', default=97, type=int)
    parser.add_argument('--training_fraction', default=0.5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--agop_batch_size', default=32, type=int)
    parser.add_argument('--iters', default=50, type=int)
    parser.add_argument('--rfm_batch_size', default=2, type=int)
    parser.add_argument('--ridge', default=1e-3, type=float)
    parser.add_argument('--bandwidth', default=1, type=float)
    parser.add_argument('--ntk_depth', default=2, type=int)
    parser.add_argument('--jac_reg_weight', default=0, type=float)
    parser.add_argument('--agip_rdx_weight', default=0, type=float)
    parser.add_argument('--agop_sma_size', default=10, type=int)
    parser.add_argument('--ema_alpha', default=0.9, type=float)
    parser.add_argument('--use_ema', default=False, action='store_true')
    parser.add_argument('--kernel_type', default='gaussian', choices={'gaussian', 'laplace', 'fcn_relu_ntk',
                                                                      'quadratic', 'general_quadratic', 'jax_fcn_ntk'})
    parser.add_argument('--save_agops', default=False, action='store_true')
    parser.add_argument('--agop_power', default=0.5, type=float)
    parser.add_argument('--agip_power', default=1.0, type=float)
    parser.add_argument('--use_k_inv', default=False, action='store_true')
    args = parser.parse_args()

    mode = 'online'
    if args.wandb_offline:
        mode = 'offline'

    wandb.init(entity=args.wandb_entity, project=args.wandb_proj_name, mode=mode, config=args,
               dir=args.out_dir)

    out_dir = os.path.join(args.out_dir, args.wandb_proj_name, wandb.run.id)
    os.makedirs(out_dir, exist_ok=True)

    wandb.run.name = f'{wandb.run.id} - p: {args.prime}, train_frac: {args.training_fraction}, ' + \
                     f'agop_power: {args.agop_power}, ema_alpha: {args.ema_alpha}, agip_power: {args.agip_power}' + \
                     f'jac_reg_weight: {args.jac_reg_weight}, ridge: {args.ridge}, bdwth: {args.bandwidth}, ' + \
                     f'agip_rdx_weight: {args.agip_rdx_weight}, agop_sma_size: {args.agop_sma_size}'

    X_tr, y_tr_onehot, X_te1, y_te1_onehot, X_te2, y_te2_onehot = multitask_op_mod_p_data("x+y", "x*y", args.prime, args.training_fraction)

    M = torch.eye(X_tr.shape[1]).double()
    Mc = torch.eye(y_tr_onehot.shape[1]).double()

    Ms = []
    Mcs = []
    ema_alpha = args.ema_alpha
    for rfm_iter in range(args.iters):
        sol, K_train, dist = solve(X_tr, y_tr_onehot, M, Mc, args.bandwidth, args.ntk_depth, args.kernel_type,
                                   ridge=args.ridge, jac_reg_weight=args.jac_reg_weight, agip_rdx_weight=args.agip_rdx_weight,
                                   use_k_inv=args.use_k_inv)

        acc, loss, corr = eval(sol, K_train, y_tr_onehot, args.prime, first_p=True, eval_all=True)
        print(f'Round {rfm_iter} Train MSE:\t{loss}')
        print(f'Round {rfm_iter} Train Acc:\t{acc}')
        wandb.log({
            'training/accuracy': acc,
            'training/loss': loss
        }, step=rfm_iter)

        K_test1 = get_test_kernel(X_tr, X_te1, M, args.bandwidth, args.ntk_depth, args.kernel_type)

        acc1, loss1, corr = eval(sol, K_test1, y_te1_onehot, args.prime, first_p=True, eval_all=False)
        print(f'Round {rfm_iter} Test add MSE:\t{loss1}')
        print(f'Round {rfm_iter} Test add Acc:\t{acc1}')

        wandb.log({
            'validation/add_accuracy': acc1,
            'validation/add_loss': loss1
        }, step=rfm_iter)

        K_test2 = get_test_kernel(X_tr, X_te2, M, args.bandwidth, args.ntk_depth, args.kernel_type)

        acc2, loss2, corr = eval(sol, K_test2, y_te2_onehot, args.prime, first_p=False, eval_all=False)
        # acc2, loss2 = eval2(sol, K_test2, y_te2_onehot, args.prime)
        # acc2, loss2, corr = eval(sol, K_test2, y_te2_onehot, args.prime, first_p=True, eval_all=False)

        print(f'Round {rfm_iter} Test xor MSE:\t{loss2}')
        print(f'Round {rfm_iter} Test xor Acc:\t{acc2}')

        wandb.log({
            'validation/xor_accuracy': acc2,
            'validation/xor_loss': loss2
        }, step=rfm_iter)

        acc = ((acc1 * y_te1_onehot.shape[0]) + (acc2 * y_te2_onehot.shape[0])) / (y_te1_onehot.shape[0] + y_te2_onehot.shape[0])
        loss = ((loss1 * y_te1_onehot.shape[0]) + (loss2 * y_te2_onehot.shape[0])) / (y_te1_onehot.shape[0] + y_te2_onehot.shape[0])

        print(f'Round {rfm_iter} Test total MSE:\t{loss}')
        print(f'Round {rfm_iter} Test total Acc:\t{acc}')
        print()

        wandb.log({
            'validation/total_accuracy': acc,
            'validation/total_loss': loss
        }, step=rfm_iter)

        M_new, Mc_new = update(X_tr, X_tr, args.bandwidth, M, sol, K_train, dist, \
                       args.kernel_type, args.ntk_depth, centers_bsize=-1, centering=True,
                       agop_power=args.agop_power, agip_power=args.agip_power)

        if args.use_ema:
            # use exponential moving average
            M = ema_alpha * M_new + (1 - ema_alpha) * M
            Mc = ema_alpha * Mc_new + (1 - ema_alpha) * Mc
        else:
            # use simple moving average
            if len(Ms) == args.agop_sma_size:
                Ms.pop(0)
                Mcs.pop(0)

            Ms.append(M_new)
            Mcs.append(Mc_new)

            M = torch.mean(torch.stack(Ms), dim=0)
            Mc = torch.mean(torch.stack(Mcs), dim=0)

        with torch.no_grad():
            wandb.log({
                'training/agop_tr': torch.trace(M),
                'training/agip_tr': torch.trace(Mc)
            }, step=rfm_iter)

        if (rfm_iter < 25) or \
            (rfm_iter < 100 and rfm_iter % 25 == 0) or \
            (rfm_iter < 500 and rfm_iter % 50 == 0):

            if args.save_agops:
                os.makedirs(os.path.join(out_dir, f'iter_{rfm_iter}'), exist_ok=True)
                np.save(os.path.join(out_dir, f'iter_{rfm_iter}/M.npy'), M.numpy())
                np.save(os.path.join(out_dir, f'iter_{rfm_iter}/Mc.npy'), Mc.numpy())

            # if using WANDB we will log images of M, Mc and their spectra
            if not args.wandb_offline:
                plt.clf()
                plt.imshow(M)
                plt.colorbar()
                img = wandb.Image(
                    plt,
                    caption=f'M'
                )
                wandb.log({'M': img}, step=rfm_iter)

                plt.clf()
                plt.imshow(M - torch.diag(torch.diag(M)))
                plt.colorbar()
                img = wandb.Image(
                    plt,
                    caption=f'M_no_diag'
                )
                wandb.log({'M_no_diag': img}, step=rfm_iter)

                Mviz = torch.zeros((2*args.prime, 2*args.prime))
                Mviz[:args.prime,:args.prime] = M[:args.prime,:args.prime]
                Mviz[:args.prime,args.prime:] = M[:args.prime,args.prime:-1]
                Mviz[args.prime:,:args.prime] = M[args.prime:-1,:args.prime]
                Mviz[args.prime:,args.prime:] = M[args.prime:-1,args.prime:-1]
                Mviz[:args.prime,:args.prime] -= torch.diag(torch.diag(Mviz[:args.prime,:args.prime]))
                Mviz[:args.prime,args.prime:] -= torch.diag(torch.diag(Mviz[:args.prime,args.prime:]))
                Mviz[args.prime:,:args.prime] -= torch.diag(torch.diag(Mviz[args.prime:,:args.prime]))
                Mviz[args.prime:,args.prime:] -= torch.diag(torch.diag(Mviz[args.prime:,args.prime:]))

                # Mviz = M.clone()
                # Mviz -= torch.diag(torch.diag(Mviz))
                # submat1 = Mviz[:args.prime,args.prime:2*args.prime]
                # submat1 -= torch.diag(torch.diag(submat1))
                # Mviz[:args.prime,args.prime:2*args.prime] = submat1
                # submat2 = Mviz[args.prime:2*args.prime,:args.prime]
                # submat2 -= torch.diag(torch.diag(submat2))
                # Mviz[args.prime:2*args.prime,:args.prime] = submat2
                plt.clf()
                plt.imshow(Mviz)
                plt.colorbar()
                img = wandb.Image(
                    plt,
                    caption=f'M_no_diags_all'
                )
                wandb.log({'M_no_diags_all': img}, step=rfm_iter)

                # plt.clf()
                # plt.imshow(Mc)
                # plt.colorbar()
                # img = wandb.Image(
                #     plt,
                #     caption=f'Mc'
                # )
                # wandb.log({'Mc': img}, step=rfm_iter)
                #
                # M_vals = torch.flip(torch.linalg.eigvalsh(M), (0,))
                # Mc_vals = torch.flip(torch.linalg.eigvalsh(Mc), (0,))
                #
                # plt.clf()
                # plt.plot(range(len(M_vals)), np.log(M_vals))
                # plt.grid()
                # plt.xlabel('eigenvalue idx')
                # plt.ylabel('ln(eigenvalue)')
                # img = wandb.Image(
                #     plt,
                #     caption='M_eigenvalues'
                # )
                # wandb.log({'M_eigs': img}, step=rfm_iter)
                #
                # plt.clf()
                # plt.plot(range(len(Mc_vals)), np.log(Mc_vals))
                # plt.grid()
                # plt.xlabel('eigenvalue idx')
                # plt.ylabel('ln(eigenvalue)')
                # img = wandb.Image(
                #     plt,
                #     caption='Mc_eigenvalues'
                # )
                # wandb.log({'Mc_eigs': img}, step=rfm_iter)

if __name__=='__main__':
    main()
