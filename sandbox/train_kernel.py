import sys
import argparse
import wandb
import torch
import torch.nn.functional as F
import numpy as np
import random

from data import get_raw_splits, block_inputs, get_encode_decode_fns
from models import gaussian_kernel, torch_fcn_relu_ntk
import utils

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.manual_seed(13)
random.seed(25)
np.random.seed(15)

def generate(sol, encoded_prompt, X_tr, M, bandwidth, ntk_depth, kernel_type, \
             block_size, vocab_size, temperature=1.0, top_k=None, next_token_only=False,
             max_new_tokens=1000):
    idx = encoded_prompt

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        idx_cond = F.one_hot(idx_cond, vocab_size).view(-1, block_size*vocab_size)
        K_prompt = get_test_kernel(X_tr, idx_cond, M, bandwidth, ntk_depth, kernel_type)
        preds = K_prompt.T @ sol

        if next_token_only:
            preds = preds / temperature
            preds = preds.view(-1, 1, vocab_size)
        else:
            preds = preds.view(-1, block_size, vocab_size)
            preds = preds[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(preds, min(top_k, preds.size(-1)))
            preds[preds < v[:, [-1]]] = -float('Inf')

         probs = F.softmax(preds, dim=-1)
         idx_next = torch.multinomial(probs, num_samples=1)

         idx = torch.cat((idx, idx_next), dim=1)

    return idx

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
            S, U = torch.linalg.eigh((K_train.T @ K_train) + ridge * torch.eye(len(K_train)) + jac_reg_weight * jac)
        else:
            S, U = torch.linalg.eigh(K_train + ridge * torch.eye(len(K_train)))
        R, V = torch.linalg.eigh(agip_rdx_weight*Mc)
        sol = U @((U.T @ y_tr_onehot @ V)/(R + S.unsqueeze(1))) @ V.T
    else:
        if jac_reg_weight > 0:
            sol = torch.from_numpy(np.linalg.inv((K_train.T @ K_train + ridge * np.eye(len(K_train)) \
                                                  + jac_reg_weight * jac).numpy())) @ K_train @ y_tr_onehot
        else:
            sol = torch.from_numpy(np.linalg.solve(K_train.numpy() + ridge * np.eye(len(K_train)), y_tr_onehot.numpy()).T)
            sol = sol.T

    return sol, K_train, dist

def update(samples, centers, bandwidth, M, weights, K, dist, \
           kernel_type, centers_bsize=-1, centering=False):
    if kernel_type == 'laplace':
        M, Mc = laplace_kernel.laplacian_M_update(samples, centers, bandwidth, M, weights, K=K, dist=dist, \
                                   centers_bsize=centers_bsize, centering=centering)
    elif kernel_type == 'gaussian':
        M, Mc = gaussian_kernel.gaussian_M_update(samples, centers, bandwidth, M, weights, K=K, \
                              centers_bsize=centers_bsize, centering=centering)

    return M, Mc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', default='belkinlab')
    parser.add_argument('--wandb_proj_name', default='mar27-rfm-lm')
    parser.add_argument('--wandb_offline', default=False, action='store_true')
    parser.add_argument('--out_dir', default='./')
    parser.add_argument('--n_train_samps', default=100, type=int)
    parser.add_argument('--n_val_samps', default=100, type=int)
    parser.add_argument('--block_size', default=32, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--next_token_only', default=False, action='store_true')
    parser.add_argument('--iters', default=50, type=int)
    parser.add_argument('--ridge', default=1e-3, type=float)
    parser.add_argument('--bandwidth', default=2.5, type=float)
    parser.add_argument('--ntk_depth', default=2, type=int)
    parser.add_argument('--jac_reg_weight', default=1e-3, type=float)
    parser.add_argument('--agip_rdx_weight', default=1e-3, type=float)
    parser.add_argument('--agop_avg_size', default=10, type=int)
    parser.add_argument('--kernel_type', default='gaussian', choices={'gaussian', 'laplace', 'fcn_relu_ntk'})
    args = parser.parse_args()

    device='cpu'

    mode = 'online'
    if args.wandb_offline:
        mode = 'offline'

    wandb.init(entity=args.wandb_entity, project=args.wandb_proj_name, mode=mode, config=args,
               dir=args.out_dir)

    wandb.run.name = f'{wandb.run.id} - p: {args.prime}, train_frac: {args.training_fraction}, ' + \
                     f'jac_reg_weight: {args.jac_reg_weight}, ridge: {args.ridge}, bdwth: {args.bandwidth}, ' + \
                     f'agip_rdx_weight: {args.agip_rdx_weight}'

    encode, decode = get_encode_decode_fns()
    with open('prompt.txt', 'r', encoding='utf-8') as f:
        start = f.read()
    start_ids = encode(start)[:args.block_size]
    encoded_prompt = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    raw_train, raw_val = get_raw_splits()
    X_tr, y_tr = block_inputs(raw_train, args.block_size, args.n_train_samps)
    X_te, y_te = block_inputs(raw_val, args.block_size, args.n_val_samps, next_token_only=args.next_token_only)

    vocab_size = len(np.unique(X_tr))
    X_tr = F.one_hot(X_tr, vocab_size).view(-1, args.block_size * vocab_size).float()
    if args.next_token_only:
        y_tr_onehot = F.one_hot(y_tr, vocab_size).float()
    else:
        y_tr_onehot = F.one_hot(y_tr, vocab_size).view(-1, args.block_size * vocab_size).float()

    X_te = F.one_hot(X_te, vocab_size).view(-1, args.block_size * vocab_size).float()
    if args.next_token_only:
        y_te_onehot = F.one_hot(y_te, vocab_size).float()
    else:
        y_te_onehot = F.one_hot(y_te, vocab_size).view(-1, args.block_size * vocab_size).float()

    M = torch.eye(X_tr.shape[1]).double()
    Mc = torch.eye(y_tr_onehot.shape[1]).double()

    Ms = []
    Mcs = []
    for rfm_iter in range(args.iters):
        sol, K_train, dist = solve(X_tr, y_tr_onehot, M, Mc, args.bandwidth, args.ntk_depth, args.kernel_type,
                                   ridge=args.ridge, jac_reg_weight=args.jac_reg_weight, agip_rdx_weight=args.agip_rdx_weight)

        acc, loss, corr = eval(sol, K_train, y_tr_onehot)
        print(f'Round {rfm_iter} Train MSE:\t{loss}')
        print(f'Round {rfm_iter} Train Acc:\t{acc}')
        # print(f'Round {rfm_iter} Train Corr:\t{corr}')
        wandb.log({
            'training/accuracy': acc,
            'training/loss': loss
        }, step=rfm_iter)

        if acc < 0.9:
            # kill the run if train accuracy diverges, we want to stay interpolating / near-interpolation
            sys.exit(1)

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

        M_new, Mc_new = update(X_tr, X_tr, args.bandwidth, M, sol, K_train, dist, \
                       args.kernel_type, centers_bsize=-1, centering=True)

        if len(Ms) == args.agop_avg_size:
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

        # if rfm_iter % 25 == 0:
        #     plt.clf()
        #     plt.imshow(M)
        #     plt.colorbar()
        #     img = wandb.Image(
        #         plt,
        #         caption=f'M'
        #     )
        #     wandb.log({'M': img}, step=rfm_iter)
        #
        #     plt.clf()
        #     plt.imshow(Mc)
        #     plt.colorbar()
        #     img = wandb.Image(
        #         plt,
        #         caption=f'Mc'
        #     )
        #     wandb.log({'Mc': img}, step=rfm_iter)
        #
        #     M_vals = torch.flip(torch.linalg.eigvalsh(M), (0,))
        #     Mc_vals = torch.flip(torch.linalg.eigvalsh(Mc), (0,))
        #
        #     plt.clf()
        #     plt.plot(range(len(M_vals)), np.log(M_vals))
        #     plt.grid()
        #     plt.xlabel('eigenvalue idx')
        #     plt.ylabel('ln(eigenvalue)')
        #     img = wandb.Image(
        #         plt,
        #         caption='M_eigenvalues'
        #     )
        #     wandb.log({'M_eigs': img}, step=rfm_iter)
        #
        #     plt.clf()
        #     plt.plot(range(len(Mc_vals)), np.log(Mc_vals))
        #     plt.grid()
        #     plt.xlabel('eigenvalue idx')
        #     plt.ylabel('ln(eigenvalue)')
        #     img = wandb.Image(
        #         plt,
        #         caption='Mc_eigenvalues'
        #     )
        #     wandb.log({'Mc_eigs': img}, step=rfm_iter)

if __name__=='__main__':
    main()
