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
                 operation_mod_p_data_binarized
from models import laplace_kernel, gaussian_kernel, torch_fcn_relu_ntk, \
                   jax_fcn_relu_ntk, quadratic_kernel
import utils

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
# torch.manual_seed(3143)
# random.seed(253)
# np.random.seed(1145)

def eval_clfs(sols, Ks, y_onehot, cls_labels):
    per_class_logits = []
    per_class_accs = {}
    per_class_losses = {}
    for cls in sorted(sols.keys()):
        preds = (Ks[cls].T @ sols[cls]).squeeze()
        cls_preds = (preds > 0.0).long()
        cls_preds[cls_preds == 0] = -1
        per_class_losses[cls] = (preds - cls_labels[cls]).pow(2).mean()
        per_class_accs[cls] = torch.sum(cls_preds == cls_labels[cls]) / y_onehot.shape[0]
        per_class_logits.append(preds)

    per_class_logits = torch.stack(per_class_logits, axis=1).squeeze()
    loss = (per_class_logits - y_onehot).pow(2).mean()
    count = torch.sum(per_class_logits.argmax(-1) == y_onehot.argmax(-1))
    acc = count / y_onehot.shape[0]
    return acc, loss, per_class_accs, per_class_losses

    # accs, losses = {}, {}
    # for cls in sols.keys():
    #     acc, loss, _ = eval(sols[cls], Ks[cls], y_onehots[cls])
    #     accs[cls] = acc
    #     losses[cls] = loss

    # return accs, losses

def eval(sol, K, y_onehot):
    preds = K.T @ sol
    loss = (preds - y_onehot).pow(2).mean()

    corr = 0
    if y_onehot.shape[1] > 1:
        count = torch.sum(y_onehot.argmax(-1) == preds.argmax(-1))
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

def solve(X_tr, y_tr_onehot, M, bandwidth, ntk_depth, kernel_type,
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

def update_clfs(X_tr, Ms, sols, K_trains, dists, args):
    newMs = {}
    for cls in sols.keys():
        M, Mc, per_class_agops = update(X_tr, X_tr, args.bandwidth, Ms[cls], sols[cls], K_trains[cls], dists[cls], \
                       args.kernel_type, args.ntk_depth, centers_bsize=-1, centering=True,
                       agop_power=args.agop_power, agip_power=args.agip_power, return_per_class_agop=False)
        newMs[cls] = M
    return newMs

def update(samples, centers, bandwidth, M, weights, K, dist, \
           kernel_type, ntk_depth, centers_bsize=-1, centering=False,
           agop_power=0.5, agip_power=1.0, return_per_class_agop=False):
    if kernel_type == 'laplace':
        M, Mc = laplace_kernel.laplacian_M_update(samples, centers, bandwidth, M, weights, K=K, dist=dist, \
                                   centers_bsize=centers_bsize, centering=centering)
    elif kernel_type == 'gaussian':
        M, Mc, per_class_agops = gaussian_kernel.gaussian_M_update(samples, centers, bandwidth, M, weights, K=K, \
                              centers_bsize=centers_bsize, centering=centering, agop_power=agop_power,
                              agip_power=agip_power, return_per_class_agop=return_per_class_agop)
    elif kernel_type == 'fcn_relu_ntk':
        M, Mc = torch_fcn_relu_ntk.ntk_relu_M_update(weights, centers, samples, M, ntk_depth=ntk_depth)
    elif kernel_type == 'jax_fcn_ntk':
        M, Mc = jax_fcn_relu_ntk.ntk_relu_M_update(weights, centers, samples, M, ntk_depth=ntk_depth)
    elif kernel_type == 'quadratic':
        M, Mc, per_class_agops = quadratic_kernel.quad_M_update(samples, centers, weights.T, M, centering=centering, \
                                                                return_per_class_agop=return_per_class_agop)
    elif kernel_type == 'general_quadratic':
        M, Mc = quadratic_kernel.general_quadratic_M_update(samples, centers, weights.T, M, centering=centering)

    return M, Mc, per_class_agops

def compute_train_class_freqs(y_tr):
    return y_tr.T @ y_tr / y_tr.shape[0]

def fit_binary_clfs(X_tr, cls_labels, Ms, args):
    sols = {}
    K_trains = {}
    dists = {}
    for cls in cls_labels.keys():
        sol, K_train, dist = solve(X_tr, cls_labels[cls], Ms[cls], args.bandwidth, args.ntk_depth, args.kernel_type,
                                   ridge=args.ridge, jac_reg_weight=args.jac_reg_weight, agip_rdx_weight=args.agip_rdx_weight,
                                   use_k_inv=args.use_k_inv)
        sols[cls] = sol.unsqueeze(-1)
        K_trains[cls] = K_train
        dists[cls] = dist

    return sols, K_trains, dists

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

    X_tr, cls_labels, y_tr_onehot, X_te, te_cls_labels, y_te_onehot = operation_mod_p_data_binarized(args.operation, args.prime, args.training_fraction)

    Ms = {}
    for idx in range(args.prime):
        Ms[idx] = torch.eye(X_tr.shape[1]).double()
    # M = torch.eye(X_tr.shape[1]).double()
    # Mc = torch.eye(y_tr_onehot.shape[1]).double()

    Mcs = []
    ema_alpha = args.ema_alpha
    for rfm_iter in range(args.iters):
        sols, K_trains, dists = fit_binary_clfs(X_tr, cls_labels, Ms, args)

        acc, loss, per_class_accs, per_class_losses = eval_clfs(sols, K_trains, y_tr_onehot, cls_labels)
        print(f'Round {rfm_iter} Train MSE:\t{loss}')
        print(f'Round {rfm_iter} Train Acc:\t{acc}')
        wandb.log({
            f'training/accuracy': acc,
            f'training/loss': loss
        }, step=rfm_iter)

        K_tests = {}
        for cls in range(args.prime):
            # print(f'Round {rfm_iter} Train Class {cls} MSE:\t{losses[cls]}')
            # print(f'Round {rfm_iter} Train Class {cls} Acc:\t{accs[cls]}')
            # wandb.log({
            #     f'training/cls_{cls}_accuracy': accs[cls],
            #     f'training/cls_{cls}_loss': losses[cls]
            # }, step=rfm_iter)

            K_tests[cls] = get_test_kernel(X_tr, X_te, Ms[cls], args.bandwidth, args.ntk_depth, args.kernel_type)

        acc, loss, per_class_accs, per_class_losses = eval_clfs(sols, K_tests, y_te_onehot, te_cls_labels)
        print(f'Round {rfm_iter} Test MSE:\t{loss}')
        print(f'Round {rfm_iter} Test Acc:\t{acc}')
        print()

        wandb.log({
            'validation/accuracy': acc,
            'validation/loss': loss
        }, step=rfm_iter)

        for cls in range(args.prime):
            wandb.log({
                f'per_class_val_acc/cls_{cls}_accuracy': per_class_accs[cls],
                f'per_class_val_loss/cls_{cls}_loss': per_class_losses[cls]
            }, step=rfm_iter)

        Ms = update_clfs(X_tr, Ms, sols, K_trains, dists, args)
        # M, Mc, per_class_agops = update(X_tr, X_tr, args.bandwidth, M, sol, K_train, dist, \
        #                args.kernel_type, args.ntk_depth, centers_bsize=-1, centering=True,
        #                agop_power=args.agop_power, agip_power=args.agip_power)
        #
        # if args.use_ema:
        #     # use exponential moving average
        #     M = ema_alpha * M_new + (1 - ema_alpha) * M
        #     Mc = ema_alpha * Mc_new + (1 - ema_alpha) * Mc
        # else:
        #     # use simple moving average
        #     if len(Ms) == args.agop_sma_size:
        #         Ms.pop(0)
        #         Mcs.pop(0)
        #
        #     Ms.append(M_new)
        #     Mcs.append(Mc_new)
        #
        #     M = torch.mean(torch.stack(Ms), dim=0)
        #     Mc = torch.mean(torch.stack(Mcs), dim=0)

        # with torch.no_grad():
        #     wandb.log({
        #         'training/agop_tr': torch.trace(M),
        #         'training/agip_tr': torch.trace(Mc)
        #     }, step=rfm_iter)

        if (rfm_iter < 31) or \
            (rfm_iter < 100 and rfm_iter % 25 == 0) or \
            (rfm_iter < 500 and rfm_iter % 50 == 0):

            # if args.save_agops:
            #     os.makedirs(os.path.join(out_dir, f'iter_{rfm_iter}'), exist_ok=True)
            #     np.save(os.path.join(out_dir, f'iter_{rfm_iter}/M.npy'), M.numpy())
            #     np.save(os.path.join(out_dir, f'iter_{rfm_iter}/Mc.npy'), Mc.numpy())
            #
            #     subdir = os.path.join(out_dir, f'iter_{rfm_iter}', 'per_class_agops')
            #     os.makedirs(subdir, exist_ok=True)
            #     for cls_idx in range(len(per_class_agops)):
            #         np.save(os.path.join(subdir, f'M_cls_{cls_idx}.npy'), per_class_agops[cls_idx].numpy())

            # if using WANDB we will log images of M, Mc and their spectra
            if not args.wandb_offline:
                for cls in Ms.keys():
                    plt.clf()
                    plt.imshow(Ms[cls])
                    plt.colorbar()
                    img = wandb.Image(
                        plt,
                        caption=f'M_{cls}'
                    )
                    wandb.log({f'M/M_{cls}': img}, step=rfm_iter)

                    plt.clf()
                    plt.imshow(Ms[cls] - torch.diag(torch.diag(Ms[cls])))
                    plt.colorbar()
                    img = wandb.Image(
                        plt,
                        caption=f'M_{cls}_no_diag'
                    )
                    wandb.log({f'M_no_diag/M_{cls}_no_diag': img}, step=rfm_iter)

                # for cls_idx in range(len(per_class_agops)):
                #     plt.clf()
                #     plt.imshow(per_class_agops[cls_idx] - torch.diag(torch.diag(per_class_agops[cls_idx])))
                #     plt.colorbar()
                #     img = wandb.Image(
                #         plt,
                #         caption=f'cls_{cls_idx} M_no_diag'
                #     )
                #     wandb.log({f'per_class/{cls_idx}_M_no_diag': img}, step=rfm_iter)

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
