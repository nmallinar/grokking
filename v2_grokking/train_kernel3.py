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
from models import laplace_kernel, gaussian_kernel, torch_fcn_relu_ntk
import utils

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
# torch.manual_seed(3143)
# random.seed(253)
# np.random.seed(1145)

def eval(sol, K, y_onehot):
    # preds = (sol @ K).T
    preds = K.T @ sol
    loss = (preds - y_onehot).pow(2).mean()

    corr = 0
    if y_onehot.shape[1] > 1:
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
        if jac_reg_weight > 0:
            raise Exception('to do')

    if agip_rdx_weight > 0:
        if jac_reg_weight > 0:
            # inv = torch.from_numpy(np.linalg.pinv((agip_rdx_weight * K_train + jac_reg_weight * jac)))
            # S, U = torch.linalg.eigh(inv @ K_train.T @ K_train)
            # R, V = torch.linalg.eigh(Mc)
            # labels = K_train.T @ y_tr_onehot
            # sol = U @((U.T @ labels @ V)/(R + S.unsqueeze(1))) @ V.T
            #
            # return sol, K_train, dist

            mat1 = K_train + ridge * torch.eye(len(K_train)) + jac_reg_weight*(torch.from_numpy(np.linalg.pinv((K_train + ridge * torch.eye(len(K_train))).numpy())) @ jac)
            S, U = torch.linalg.eigh(mat1)
            R, V = torch.linalg.eigh(agip_rdx_weight * Mc)
            labels = y_tr_onehot @ Mc
            # labels = y_tr_onehot @ torch.from_numpy(np.real(scipy.linalg.sqrtm(Mc)))
            sol = U @((U.T @ labels @ V)/(R + S.unsqueeze(1))) @ V.T
            return sol, K_train, dist
        else:
            S, U = torch.linalg.eigh(K_train + ridge * torch.eye(len(K_train)))
            R, V = torch.linalg.eigh(agip_rdx_weight*Mc)
            labels = y_tr_onehot
            sol = U @((U.T @ labels @ V)/(R + S.unsqueeze(1))) @ V.T
            return sol, K_train, dist
    else:
        if jac_reg_weight > 0:
            # jac_inv = torch.from_numpy(np.linalg.pinv(jac.numpy()))
            # import ipdb; ipdb.set_trace()
            # y_tr_onehot = y_tr_onehot @ torch.linalg.pinv(Mc)
            # K_inv = np.linalg.pinv(K_train.numpy())
            # sol = torch.from_numpy(np.linalg.pinv(K_train.numpy() + jac_reg_weight * K_inv @ jac.numpy()) @ y_tr_onehot.numpy())
            # return sol, K_train, dist

            sol = torch.from_numpy(np.linalg.inv((K_train.T @ K_train + ridge * np.eye(len(K_train)) \
                                                  + jac_reg_weight * jac).numpy())) @ K_train.T @ y_tr_onehot
        else:
            # K_tr_inv = np.linalg.pinv(np.real(scipy.linalg.sqrtm(K_train)))
            # K_tr_inv = np.linalg.pinv(K_train.numpy())
            # K_sqrt = np.real(scipy.linalg.sqrtm(K_train))
            K_inv = np.linalg.inv(K_train)
            class_size = y_tr_onehot.shape[1]
            normalization = torch.eye(class_size) - (1./K_train.shape[0] * torch.ones(class_size, class_size))
            label_cov = (normalization @ y_tr_onehot.T @ K_inv @ y_tr_onehot @ normalization)
            # label_cov = y_tr_onehot.T @ K_inv @ y_tr_onehot
            label_cov /= torch.max(label_cov, dim=1).values

            labels = y_tr_onehot @ label_cov
            sol1 = torch.from_numpy(np.linalg.solve(K_train.numpy() + ridge * np.eye(len(K_train)), labels.numpy()).T)
            sol1 = sol1.T
            # import ipdb; ipdb.set_trace()
            return sol1, K_train, dist

            sol = torch.from_numpy(np.linalg.solve(K_train.T @ K_train.numpy() + ridge * np.eye(len(K_train)), y_tr_onehot.numpy()).T)
            sol = sol.T

            import ipdb; ipdb.set_trace()

    return sol, K_train, dist

def update(samples, centers, bandwidth, M, weights, K, dist, \
           kernel_type, centers_bsize=-1, centering=False,
           agop_power=0.5, agip_power=1.0, ntk_depth=2):
    if kernel_type == 'laplace':
        M, Mc = laplace_kernel.laplacian_M_update(samples, centers, bandwidth, M, weights, K=K, dist=dist, \
                                   centers_bsize=centers_bsize, centering=centering)
    elif kernel_type == 'gaussian':
        M, Mc, per_class_agops = gaussian_kernel.gaussian_M_update(samples, centers, bandwidth, M, weights, K=K, \
                              centers_bsize=centers_bsize, centering=centering, agop_power=agop_power,
                              agip_power=agip_power, return_per_class_agop=True)
    elif kernel_type == 'fcn_relu_ntk':
        M, Mc = torch_fcn_relu_ntk.ntk_relu_M_update(weights, centers, samples, M, ntk_depth=ntk_depth)

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
    parser.add_argument('--kernel_type', default='gaussian', choices={'gaussian', 'laplace', 'fcn_relu_ntk'})
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

    X_tr = F.one_hot(X_tr, args.prime).view(-1, 2*args.prime).double()
    y_tr_onehot = F.one_hot(y_tr, args.prime).double()
    # y_tr_onehot = 40 * y_tr_onehot + 2*torch.randn(y_tr_onehot.shape)
    X_te = F.one_hot(X_te, args.prime).view(-1, 2*args.prime).double()
    y_te_onehot = F.one_hot(y_te, args.prime).double()

    Mc_true = compute_train_class_freqs(y_tr_onehot)
    eigs, vecs = np.linalg.eigh(Mc_true.numpy())
    eigs = np.power(eigs, args.agip_power)
    eigs[np.isnan(eigs)] = 0.0
    Mc_true = torch.from_numpy(vecs @ np.diag(eigs) @ vecs.T)

    plt.clf()
    plt.imshow(Mc_true)
    plt.colorbar()
    img = wandb.Image(
        plt,
        caption=f'train class cov'
    )
    wandb.log({'train_class_cov': img}, step=0)

    Mc_true_inv = torch.from_numpy(np.linalg.pinv(Mc_true.numpy()))

    X_agop = torch.cat((X_tr, X_te))
    # X_agop = torch.randn(2000, X_agop.shape[1])

    M = torch.eye(X_tr.shape[1]).double()
    Mc = torch.eye(y_tr_onehot.shape[1]).double()
    Mc_inv = torch.eye(y_tr_onehot.shape[1]).double()

    Ms = []
    Mcs = []
    ema_alpha = args.ema_alpha
    for rfm_iter in range(args.iters):
        sol, K_train, dist = solve(X_tr, y_tr_onehot, M, Mc, args.bandwidth, args.ntk_depth, args.kernel_type,
                                   ridge=args.ridge, jac_reg_weight=args.jac_reg_weight, agip_rdx_weight=args.agip_rdx_weight)

        # Mc_true_inv = torch.from_numpy(np.linalg.pinv(K_train.numpy()))
        # plt.clf()
        # plt.imshow(y_tr_onehot.T @ Mc_true_inv @ y_tr_onehot)
        # plt.colorbar()
        # img = wandb.Image(
        #     plt,
        #     caption=f'train class cov'
        # )
        # wandb.log({'train_class_cov': img}, step=rfm_iter)

        acc, loss, corr = eval(sol, K_train, y_tr_onehot)
        print(f'Round {rfm_iter} Train MSE:\t{loss}')
        print(f'Round {rfm_iter} Train Acc:\t{acc}')
        # print(f'Round {rfm_iter} Train Corr:\t{corr}')
        wandb.log({
            'training/accuracy': acc,
            'training/loss': loss
        }, step=rfm_iter)

        K_test = get_test_kernel(X_tr, X_te, M, args.bandwidth, args.ntk_depth, args.kernel_type)

        acc, loss, corr = eval(sol, K_test, y_te_onehot)
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
        #     # perm = torch.randperm(X_agop.shape[0])
        #     # X_samp = X_agop[perm[:X_tr.shape[0]],:]
        #     X_samp = X_tr.clone() + torch.randn(X_tr.shape)
        #     M_temp, Mc_temp = update(X_tr, X_samp, args.bandwidth, M, sol, None, None, \
        #                              args.kernel_type, centers_bsize=-1, centering=True,
        #                              agop_power=0.5, agip_power=args.agip_power)
        #     M_new += M_temp
        #     Mc_new += Mc_temp
        #
        # M_new /= 10
        # Mc_new /= 10

        # M_new = torch.from_numpy(np.real(scipy.linalg.sqrtm(M_new.numpy())))

        # M_new, Mc_new = update(X_tr, X_te, args.bandwidth, M, sol, None, None, \
        #                args.kernel_type, centers_bsize=-1, centering=True,
        #                agop_power=args.agop_power, agip_power=args.agip_power)
        # M_new, Mc_new = update(X_tr, X_tr, args.bandwidth, M, sol, None, None, \
                       # args.kernel_type, centers_bsize=-1, centering=True)
        M_new, Mc_new = update(X_tr, X_tr, args.bandwidth, M, sol, K_train, dist, \
                       args.kernel_type, centers_bsize=-1, centering=True,
                       agop_power=args.agop_power, agip_power=args.agip_power,
                       ntk_depth=args.ntk_depth)

        if args.use_ema:
            M = ema_alpha * M_new + (1 - ema_alpha) * M
            Mc = ema_alpha * Mc_new + (1 - ema_alpha) * Mc

            # Mc_inv = torch.from_numpy(np.linalg.pinv(Mc.numpy()))
            # eigs, vecs = np.linalg.eigh(Mc.numpy())
            # eigs = np.power(eigs, -0.5)
            # eigs[np.isnan(eigs)] = 0.0
            # Mc_inv = torch.from_numpy(vecs @ np.diag(eigs) @ vecs.T)
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

        '''
        if (rfm_iter < 25) or \
            (rfm_iter < 100 and rfm_iter % 25 == 0) or \
            (rfm_iter < 500 and rfm_iter % 50 == 0):

            # if rfm_iter == 150:
            #     os.makedirs(os.path.join(out_dir, f'iter_{rfm_iter}'), exist_ok=True)

            plt.clf()
            plt.imshow(M)
            plt.colorbar()
            # if rfm_iter == 150:
            #     plt.title('M all classes')
            #     plt.savefig(os.path.join(out_dir, f'iter_{rfm_iter}/M.pdf'), bbox_inches='tight')

            img = wandb.Image(
                plt,
                caption=f'M'
            )
            wandb.log({'M': img}, step=rfm_iter)

            # for class_i in range(args.prime):
            #     plt.clf()
            #     plt.imshow(per_class_agops[class_i])
            #     plt.colorbar()
            #     if rfm_iter == 150:
            #         plt.title(f'M class {class_i}')
            #         plt.savefig(os.path.join(out_dir, f'iter_{rfm_iter}/M_{class_i}.pdf'), bbox_inches='tight')
            #
            #     img = wandb.Image(
            #         plt,
            #         caption=f'M_class_{class_i}'
            #     )
            #     wandb.log({f'M_class_{class_i}': img}, step=rfm_iter)
            #
            #
            #     Mi_vals = torch.flip(torch.linalg.eigvalsh(per_class_agops[class_i]), (0,))
            #     plt.clf()
            #     plt.plot(range(len(Mi_vals)), np.log(Mi_vals))
            #     plt.grid()
            #     plt.xlabel('eigenvalue idx')
            #     plt.ylabel('ln(eigenvalue)')
            #     img = wandb.Image(
            #         plt,
            #         caption=f'M_class_{class_i}_eigenvalues'
            #     )
            #     wandb.log({f'spectra/M_class_{class_i}_eigs': img}, step=rfm_iter)

            plt.clf()
            plt.imshow(Mc)
            plt.colorbar()
            img = wandb.Image(
                plt,
                caption=f'Mc'
            )
            wandb.log({'Mc': img}, step=rfm_iter)

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
            # wandb.log({'spectra/M_eigs': img}, step=rfm_iter)
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
            # wandb.log({'spectra/Mc_eigs': img}, step=rfm_iter)
            #
            # if args.save_agops:
            #     os.makedirs(os.path.join(out_dir, f'iter_{rfm_iter}'), exist_ok=True)
            #     np.save(os.path.join(out_dir, f'iter_{rfm_iter}/M.npy'), M.numpy())
            #     np.save(os.path.join(out_dir, f'iter_{rfm_iter}/Mc.npy'), Mc.numpy())

        '''
if __name__=='__main__':
    main()
