import os
import sys
import argparse
import wandb
import torch
import torch.nn.functional as F
import numpy as np
import scipy
import random

from data import operation_mod_p_data, make_data_splits, get_s5_data
from models import laplace_kernel, gaussian_kernel, torch_fcn_relu_ntk, jax_fcn_relu_ntk, quadratic_kernel
import utils

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.manual_seed(3143)
random.seed(253)
np.random.seed(1145)

def dft(sol, K, rfm_iter):
    preds = K.T @ sol
    freqs = np.mean(np.abs(np.fft.fft(preds, axis=1)), axis=0)
    for f_idx in range(31):
        wandb.log({
            f'dft_tr/coef_{f_idx}': freqs[f_idx],
        }, step=rfm_iter)

def eval(sol, K, y_onehot):
    # preds = (sol @ K).T
    preds = K.T @ sol
    loss = (preds - y_onehot).pow(2).mean()

    corr = 0
    if y_onehot.shape[1] > 1:
        # y_onehot = y_onehot.view(-1, 2, 31)
        # preds = preds.view(-1, 2, 31)
        # count = torch.sum(y_onehot[:,0].argmax(-1) == preds[:,0].argmax(-1)) + torch.sum(y_onehot[:,1].argmax(-1) == preds[:,1].argmax(-1))
        # acc = count / (2*y_onehot.shape[0])

        count = torch.sum(y_onehot.argmax(-1) == preds.argmax(-1))
        acc = count / y_onehot.shape[0]
        # corr = utils.compute_correlation_one_hot(y_onehot, preds).mean()
    else:
        # use correlation if regressing directly on label since accuracy will not work
        # corr = compute_correlation(y_onehot, preds)
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

    return K_test

def solve(X_tr, y_tr_onehot, M, Mc, bandwidth, ntk_depth, kernel_type,
          ridge=1e-3, jac_reg_weight=0.0, agip_rdx_weight=0.0):

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
        # import ipdb; ipdb.set_trace()
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

    if agip_rdx_weight > 0:
        if jac_reg_weight > 0:
            # inv = torch.from_numpy(np.linalg.pinv((agip_rdx_weight * K_train + jac_reg_weight * jac)))
            # S, U = torch.linalg.eigh(inv @ K_train.T @ K_train)
            # R, V = torch.linalg.eigh(Mc)
            # labels = K_train.T @ y_tr_onehot
            # sol = U @((U.T @ labels @ V)/(R + S.unsqueeze(1))) @ V.T
            #
            # return sol, K_train, dist

            mat1 = K_train + jac_reg_weight*(torch.from_numpy(np.linalg.pinv(K_train.numpy())) @ jac)
            S, U = torch.linalg.eigh(mat1)
            R, V = torch.linalg.eigh(agip_rdx_weight * Mc)
            labels = y_tr_onehot
            # labels = y_tr_onehot @ torch.from_numpy(np.real(scipy.linalg.sqrtm(Mc)))
            sol = U @((U.T @ labels @ V)/(R + S.unsqueeze(1))) @ V.T
            return sol, K_train, dist
        else:
            # class_size = y_tr_onehot.shape[1]
            # K_inv = torch.from_numpy(np.linalg.inv(K_train))
            # normalization = torch.eye(class_size) - (1./K_train.shape[0] * torch.ones(class_size, class_size))
            # # normalization = torch.from_numpy(np.linalg.pinv(y_tr_onehot.T @ y_tr_onehot))
            # Mc = (normalization @ y_tr_onehot.T @ K_inv @ y_tr_onehot @ normalization)

            K_train = K_train + ridge * torch.eye(len(K_train))
            S, U = torch.linalg.eigh(K_train.T @ K_train)
            R, V = torch.linalg.eigh(agip_rdx_weight*Mc)
            labels = y_tr_onehot
            sol = U @((U.T @ labels @ V)/(R + S.unsqueeze(1))) @ V.T
            return sol, K_train, dist
    else:
        if jac_reg_weight > 0:
            sol = torch.from_numpy(np.linalg.inv((K_train.T @ K_train + ridge * np.eye(len(K_train)) \
                                                  + jac_reg_weight * jac).numpy())) @ y_tr_onehot
        else:
            # K_inv = np.linalg.inv(K_train)
            # class_size = y_tr_onehot.shape[1]
            # # normalization = torch.eye(class_size) - (1./K_train.shape[0] * torch.ones(class_size, class_size))
            # # normalization = torch.from_numpy(np.linalg.inv(y_tr_onehot.T @ y_tr_onehot))
            # # normalization = torch.from_numpy(np.linalg.pinv(np.real(scipy.linalg.sqrtm(y_tr_onehot.T @ y_tr_onehot))))
            # # # labels = K_inv @ y_tr_onehot.numpy() / class_size
            # # label_cov = (normalization @ y_tr_onehot.T @ K_inv @ y_tr_onehot @ normalization)
            # label_cov = y_tr_onehot.T @ K_inv @ y_tr_onehot
            # # label_cov /= torch.max(label_cov, dim=1).values
            #
            # labels = y_tr_onehot @ label_cov
            # labels = F.sigmoid(labels)
            # # labels = F.softmax(labels + torch.min(labels, dim=1, keepdim=True).values)
            # # labels /= torch.max(labels, dim=1, keepdim=True).values
            # # labels = y_tr_onehot @ Mc
            # # label_cov = np.linalg.pinv(K_train.T @ K_train)
            # # labels = label_cov @ y_tr_onehot.numpy()
            # sol = torch.from_numpy(np.linalg.solve(K_train.numpy() + ridge * np.eye(len(K_train)), labels.numpy()).T)
            # sol = sol.T

            # K_inv = np.linalg.inv(K_train)
            # sol = torch.from_numpy(np.linalg.solve(K_train.numpy() + ridge * np.eye(len(K_train)), y_tr_onehot.numpy()).T)
            K_train = K_train + ridge*np.eye(len(K_train))
            # K_sqrt = torch.from_numpy(np.real(scipy.linalg.sqrtm(K_train.numpy())))
            sol = torch.from_numpy(np.linalg.solve(K_train.numpy(), y_tr_onehot.numpy()).T)

            # sol = torch.from_numpy(np.linalg.solve(K_train.T @ K_train.numpy() + ridge * np.eye(len(K_train)), y_tr_onehot.numpy()).T)
            sol = sol.T
            # preds = K_train @ sol
            # new_labels = y_tr_onehot @ (preds.T @ preds)
            # sol = torch.from_numpy(np.linalg.solve(K_train.numpy() + ridge * np.eye(len(K_train)), new_labels.numpy()).T)
            # sol = sol.T
            # import ipdb; ipdb.set_trace()

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
    parser.add_argument('--kernel_type', default='gaussian', choices={'gaussian', 'laplace', 'fcn_relu_ntk', 'jax_fcn_ntk', 'quadratic'})
    parser.add_argument('--save_agops', default=False, action='store_true')
    parser.add_argument('--agop_power', default=0.5, type=float)
    parser.add_argument('--agip_power', default=1.0, type=float)
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

    all_inputs, all_labels = operation_mod_p_data(args.operation, args.prime)
    X_tr, y_tr, X_te, y_te = make_data_splits(all_inputs, all_labels, args.training_fraction)

    '''
    n_inp_toks = args.prime
    X_tr = F.one_hot(X_tr, n_inp_toks).view(-1, 2*n_inp_toks).double()
    # y_tr_onehot = F.one_hot(y_tr, args.prime).view(-1, 2*n_inp_toks).double()
    y_tr_onehot = F.one_hot(y_tr, args.prime).double()
    X_te = F.one_hot(X_te, n_inp_toks).view(-1, 2*n_inp_toks).double()
    y_te_onehot = F.one_hot(y_te, args.prime).double()
    # y_te_onehot = F.one_hot(y_te, args.prime).view(-1, 2*n_inp_toks).double()
    '''

    proj_dim = 500
    w1 = torch.randn(args.prime, proj_dim)
    X_tr = (F.one_hot(X_tr, args.prime).double() @ w1).view(-1, 2*proj_dim)
    X_te = (F.one_hot(X_te, args.prime).double() @ w1).view(-1, 2*proj_dim)
    X_tr = torch.cat((torch.cos(X_tr), torch.sin(X_tr)), dim=1)
    X_te = torch.cat((torch.cos(X_te), torch.sin(X_te)), dim=1)
    y_tr_onehot = F.one_hot(y_tr, args.prime).double()
    y_te_onehot = F.one_hot(y_te, args.prime).double()

    Mc_true = compute_train_class_freqs(y_tr_onehot)

    X_agop = torch.cat((X_tr, X_te))
    # X_tr = X_agop
    # y_tr_onehot = torch.cat((y_tr_onehot, y_te_onehot))
    # X_agop = torch.randn(2000, X_agop.shape[1])

    # M = torch.from_numpy(np.load('saved_agops/relu_small_init_p19/right_nfm.npy')).double()
    # _M = torch.from_numpy(np.load('saved_agops/x+y_M_p19.npy')).double()
    # M = torch.from_numpy(np.load('saved_agops/right_nfm_p31.npy')).double()
    # _M = torch.from_numpy(np.load('saved_agops/x+y_M.npy')).double()
    # scale = torch.linalg.norm(M) / torch.linalg.norm(_M)
    # M /= scale
    # M = torch.flip(M, [0, 1])
    # import ipdb; ipdb.set_trace()
    M = torch.eye(X_tr.shape[1]).double()
    Mc = torch.eye(y_tr_onehot.shape[1]).double()

    M_prev = M.clone()
    Ms = []
    Mcs = []
    ema_alpha = args.ema_alpha
    for rfm_iter in range(args.iters):
        # submat = torch.rand(31,31)
        # circ = torch.from_numpy(scipy.linalg.circulant(submat[0].numpy()))
        # M[:31,31:] = circ
        # M[31:,:31] = circ.T
        sol, K_train, dist = solve(X_tr, y_tr_onehot, M, Mc, args.bandwidth, args.ntk_depth, args.kernel_type,
                                   ridge=args.ridge, jac_reg_weight=args.jac_reg_weight, agip_rdx_weight=args.agip_rdx_weight)

        # K_inv = torch.linalg.pinv(K_train.T @ K_train)
        # K_sqrt = torch.from_numpy(np.real(scipy.linalg.sqrtm(K_train)))
        wandb.log({
            'training/sol_norm': torch.linalg.norm(sol)
        }, step=rfm_iter)

        acc, loss, corr = eval(sol, K_train, y_tr_onehot)
        print(f'Round {rfm_iter} Train MSE:\t{loss}')
        print(f'Round {rfm_iter} Train Acc:\t{acc}')
        # print(f'Round {rfm_iter} Train Corr:\t{corr}')
        wandb.log({
            'training/accuracy': acc,
            'training/loss': loss
        }, step=rfm_iter)

        # shape: (train, test)
        K_test = get_test_kernel(X_tr, X_te, M, args.bandwidth, args.ntk_depth, args.kernel_type)

        # K_inv = torch.linalg.pinv(K_test.T @ K_test)
        # KK = K_inv @ K_inv.T
        # if rfm_iter > 100:
        #     import ipdb; ipdb.set_trace()
        acc, loss, corr = eval(sol, K_test, y_te_onehot)
        # dft(sol, K_test, rfm_iter)
        print(f'Round {rfm_iter} Test MSE:\t{loss}')
        print(f'Round {rfm_iter} Test Acc:\t{acc}')
        # print(f'Round {rfm_iter} Test Corr:\t{corr}')
        print()

        wandb.log({
            'validation/accuracy': acc,
            'validation/loss': loss
        }, step=rfm_iter)

        # M_new, Mc_new = torch.zeros(X_tr.shape[1], X_tr.shape[1]), torch.zeros(y_tr_onehot.shape[1], y_tr_onehot.shape[1])
        # for _ in range(10):
        #     perm = torch.randperm(X_agop.shape[0])
        #     X_samp = X_agop[perm[:X_tr.shape[0]],:]
        #     M_temp, Mc_temp = update(X_tr, X_samp, args.bandwidth, M, sol, None, None, \
        #                              args.kernel_type, centers_bsize=-1, centering=True,
        #                              agop_power=1.0, agip_power=args.agip_power)
        #     M_new += M_temp
        #     Mc_new += Mc_temp
        #
        # M_new /= 10
        # Mc_new /= 10
        #
        # M_new = torch.from_numpy(np.real(scipy.linalg.sqrtm(M_new.numpy())))

        # M_new, Mc_new = update(X_tr, X_te, args.bandwidth, M, sol, None, None, \
        #                args.kernel_type, centers_bsize=-1, centering=True,
        #                agop_power=args.agop_power, agip_power=args.agip_power)
        # M_new, Mc_new = update(X_tr, X_agop, args.bandwidth, M, sol, None, None, \
        #                args.kernel_type, args.ntk_depth, centers_bsize=-1, centering=True,
        #                agop_power=args.agop_power, agip_power=args.agip_power)
        M_new, Mc_new = update(X_tr, X_tr, args.bandwidth, M, sol, K_train, dist, \
                       args.kernel_type, args.ntk_depth, centers_bsize=-1, centering=True,
                       agop_power=args.agop_power, agip_power=args.agip_power)

        if args.use_ema:
            M = ema_alpha * M_new + (1 - ema_alpha) * M
            Mc = ema_alpha * Mc_new + (1 - ema_alpha) * Mc
            #
            # M_prev = M.clone()
            # M = torch.from_numpy(np.real(scipy.linalg.sqrtm(M)))
        else:
            # use simple moving average
            if len(Ms) == args.agop_sma_size:
                Ms.pop(0)
                Mcs.pop(0)

            Ms.append(M_new)
            Mcs.append(Mc_new)

            M = torch.mean(torch.stack(Ms), dim=0)
            Mc = torch.mean(torch.stack(Mcs), dim=0)

            # submat = M[:args.prime,args.prime:]
            # dist = 0.0
            # for p_idx in range(1, args.prime-1):
            #     dist += np.linalg.norm(np.roll(submat[p_idx], 1) - submat[p_idx+1].numpy())
            # dist += np.linalg.norm(np.roll(submat[-1], 1) - submat[0].numpy())
            # dist /= args.prime
            # wandb.log({
            #     'training/dist_to_circ': dist
            # }, step=rfm_iter)

            # diag1 = torch.diag(M[:31,:31])
            # diag2 = torch.diag(M[31:,31:])
            # plt.clf()
            # plt.plot(range(len(diag1)), diag1)
            # plt.grid()
            # img = wandb.Image(
            #     plt,
            #     caption=f'M_diag1'
            # )
            # wandb.log({'M_diag1': img}, step=rfm_iter)
            # plt.clf()
            # plt.plot(range(len(diag2)), diag2)
            # plt.grid()
            # img = wandb.Image(
            #     plt,
            #     caption=f'M_diag2'
            # )
            # wandb.log({'M_diag2': img}, step=rfm_iter)
            # D = torch.diag(M)
            # D = torch.linalg.inv(torch.diag(torch.sqrt(D))).clone()
            # M = D @ M @ D

            if rfm_iter > 497:
                # M -= torch.diag(torch.diag(M))
                # M += torch.diag(torch.ones(M.shape[0]))

                # D = torch.diag(M[0][0]*torch.ones(62))
                D = torch.diag(M)
                # max_idx = torch.argmax(D[:args.prime])
                D = torch.linalg.inv(torch.diag(torch.sqrt(D))).clone()
                # M[:args.prime,:args.prime] = torch.eye(args.prime)
                # M[args.prime:,args.prime:] = torch.eye(args.prime)
                M = D @ M @ D
                M[:args.prime,:args.prime] = 1./(args.prime - 1)*torch.eye(args.prime) - 1./(args.prime - 1) * torch.ones(args.prime, args.prime)
                M[args.prime:,args.prime:] = 1./(args.prime - 1)*torch.eye(args.prime) - 1./(args.prime - 1) * torch.ones(args.prime, args.prime)
                M[:args.prime,:args.prime] += torch.eye(args.prime)
                M[args.prime:,args.prime:] += torch.eye(args.prime)

                max_idx = 0
                row1 = M[:args.prime,args.prime:][:,max_idx]
                circ = torch.from_numpy(scipy.linalg.circulant(row1.numpy()))
                # circ = np.roll(circ, max_idx, axis=1)
                M[:args.prime,args.prime:] = circ
                M[args.prime:,:args.prime] = circ.T

                # diag1 = torch.diag(M[:31,:31]).clone()
                # diag2 = torch.diag(M[31:,31:]).clone()
                # M[:31,:31] = torch.zeros(31, 31)
                # M[31:, 31:] = torch.zeros(31, 31)
                # M[:31,:31] = torch.eye(31) - 1./31 * torch.ones(31, 31)
                # M[31:,31:] = torch.eye(31) - 1./31 * torch.ones(31, 31)
                # M[:31,:31] = torch.eye(31) - 1./31 * torch.ones(31, 31)
                # M[31:,31:] = torch.eye(31) - 1./31 * torch.ones(31, 31)
                # M[:31,:31] = torch.zeros(31) - 1./31 * torch.ones(31, 31)
                # M[31:,31:] = torch.zeros(31) - 1./31 * torch.ones(31, 31)
                # M[:31,:31] += torch.diag(1./31 * torch.ones(31))
                # M[31:,31:] += torch.diag(1./31 * torch.ones(31))
                # M[:31,:31] += torch.diag(diag1)
                # M[31:,31:] += torch.diag(diag2)
        with torch.no_grad():
            # dft_M_re = np.real(np.fft.fft(M[31:,:31]))
            # dft_M_im = np.imag(np.fft.fft(M[31:,:31]))
            # for f_idx in range(31):
            #     avg_coeff_re = np.mean(dft_M_re[:,f_idx])
            #     avg_coeff_im = np.mean(dft_M_im[:,f_idx])
            #     wandb.log({
            #         f'dft_tr/re_avg_coef_{f_idx}': avg_coeff_re,
            #         f'dft_tr/im_avg_coef_{f_idx}': avg_coeff_im
            #     }, step=rfm_iter)

            wandb.log({
                'training/agop_tr': torch.trace(M),
                'training/agip_tr': torch.trace(Mc),
                'training/agop_norm': torch.linalg.matrix_norm(M)
            }, step=rfm_iter)

        if (rfm_iter < 100) or \
            (rfm_iter < 200 and rfm_iter % 25 == 0) or \
            (rfm_iter < 500 and rfm_iter % 50 == 0):

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

            plt.clf()
            plt.imshow(K_train)
            plt.colorbar()
            img = wandb.Image(
                plt,
                caption=f'K_train'
            )
            wandb.log({'K_train': img}, step=rfm_iter)

            plt.clf()
            plt.imshow(K_train[31:62,:31])
            plt.colorbar()
            img = wandb.Image(
                plt,
                caption=f'K_train_block'
            )
            wandb.log({'K_train_block': img}, step=rfm_iter)

            plt.clf()
            plt.imshow(np.real(np.fft.fft(M)))
            plt.colorbar()
            img = wandb.Image(
                plt,
                caption=f'M_fft'
            )
            wandb.log({'M_fft': img}, step=rfm_iter)

            plt.clf()
            plt.imshow(Mc)
            plt.colorbar()
            img = wandb.Image(
                plt,
                caption=f'Mc'
            )
            wandb.log({'Mc': img}, step=rfm_iter)
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

            if args.save_agops:
                os.makedirs(os.path.join(out_dir, f'iter_{rfm_iter}'), exist_ok=True)
                np.save(os.path.join(out_dir, f'iter_{rfm_iter}/M.npy'), M.numpy())
                np.save(os.path.join(out_dir, f'iter_{rfm_iter}/Mc.npy'), Mc.numpy())

if __name__=='__main__':
    main()
