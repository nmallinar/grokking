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
import wandb
from multiprocessing import Pool
from eigenpro2 import KernelModel

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
    return laplace_kernel_M(pair1, pair2, bandwidth, M)
    #return gaussian_kernel_M(pair1, pair2, bandwidth, M)
    # sqrtM = np.real(scipy.linalg.sqrtm(M))
    # return ep3_ntk_relu(pair1 @  sqrtM, pair2 @ sqrtM, depth=bandwidth)

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

def batch_matrix_product_sum(input_tensor):
    # Define a function to compute matrix product and sum for a single batch
    def compute_product_sum(batch):
        return torch.matmul(batch, batch.transpose(1, 0)).sum(dim=0)

    product_sum = torch.vmap(compute_product_sum)(input_tensor)

    return product_sum

def get_grad_reg(X, L, P):
    # n x n x d
    # all_diffs = torch.zeros(X.size(0), X.size(0), X.size(1))

    X1 = X.unsqueeze(0)
    X2 = X.unsqueeze(1)
    all_diffs = X1 - X2

    def outer_prod(x):
        return x@x.T

    G = 0.
    for idx in range(0, X.size(0), 32):
        prod = torch.vmap(outer_prod)(all_diffs[idx:idx+4])
        G += torch.sum(prod, dim=0)

    # product = np.einsum('nmd,nkd->mk', all_diffs.numpy(), all_diffs.numpy(), optimize=True)
    # n, m, d = all_diffs.size()

    # prod = torch.matmul(all_diffs.view(n*m, d), all_diffs.view(n*m, d).transpose(1, 0))

    # prod = prod.view(n, m, m)
    # prod = all_diffs @ all_diffs.transpose(1,2)

    # for i in range(X.size(0)):
    #     all_diffs[i] = X - X[i]
    # G = 0.
    # for i in range(X.size(0)):
    #     G += all_diffs[i] @ all_diffs[i].T
    # import ipdb; ipdb.set_trace()

    return G * 1./L

def eval(X_train, X_test, L, M, y_test, y_test_onehot, sol, i, log_key=''):
    K_test = kernel_fn(X_train, X_test, L, torch.from_numpy(M)).numpy()
    preds = (sol @ K_test).T
    loss = np.mean(np.square(preds - y_test_onehot.numpy()))
    print("Round " + str(i) + f"{log_key} MSE: ", loss) # Loss function
    y_pred = torch.from_numpy(preds)
    preds = torch.argmax(y_pred, dim=-1)

    count = torch.sum(y_test == preds).numpy()
    acc = count / len(y_test)
    print("Round " + str(i) + f"{log_key} Acc: ", acc)
    return acc, loss

def rfm(X_train, y_train_onehot, X_test, y_test_onehot, num_classes, wandb,
        iters=3, name=None, batch_size=2, reg=1e-3,
        train_acc=False, L=1, agop_weight=1e-5,
        X_test_alt=None, y_test_alt_onehot=None, use_jac_reg=False,
        X_test_alt2=None, y_test_alt2_onehot=None):
    n, d = X_train.shape
    y_train = y_train_onehot.argmax(-1)
    y_test = y_test_onehot.argmax(-1)
    if y_test_alt_onehot is not None:
        y_test_alt = y_test_alt_onehot.argmax(-1)
    if y_test_alt2_onehot is not None:
        y_test_alt2 = y_test_alt2_onehot.argmax(-1)

    M = np.eye(d, dtype='float64')
    M = torch.from_numpy(M).to('cuda')
    for i in range(iters):
        epro_kernel_fn = lambda x, y: kernel_fn(x, y, L, M)

        epro = KernelModel(epro_kernel_fn, X_train, y_train_onehot.shape[-1], L, device='cuda')
        epro.fit(X_train, y_train_onehot, x_val=X_test, y_val=y_test_onehot, epochs=10,
                 mem_gb=12, print_every=1)
        # K_train = kernel_fn(X_train, X_train, L, torch.from_numpy(M)).numpy()
        #
        # if use_jac_reg:
        #     G_reg = get_grad_reg(X_train, L, torch.from_numpy(M)).numpy()
        #     sol = np.linalg.inv(K_train.T @ K_train + reg * np.eye(len(K_train)) + agop_weight*G_reg) @ K_train @ y_train_onehot.numpy()
        #     sol = sol.T
        # else:
        #     sol = solve(K_train + reg * np.eye(len(K_train)), y_train_onehot).T
        #
        # if train_acc:
        #     preds = (sol @ K_train).T
        #     loss = np.mean(np.square(preds - y_train_onehot.numpy()))
        #     y_pred = torch.from_numpy(preds)
        #     preds = torch.argmax(y_pred, dim=-1)
        #     count = torch.sum(y_train == preds).numpy()
        #     acc = count / len(y_train)
        #
        #     #if i % 50 == 0:
        #     print("Round " + str(i) + " Train MSE: ", loss) # Loss function
        #     print("Round " + str(i) + " Train Acc: ", acc)
        #
        #     metrics = {
        #         'training/accuracy': acc,
        #         'training/loss': loss,
        #         'rfm_iter': i
        #     }
        #     if wandb is not None:
        #         wandb.log(metrics, commit=False)

        # if i % 50 == 0:
        # top_k_val = 5
        # K_test = kernel_fn(X_train, X_test, L, torch.from_numpy(M)).numpy()
        # preds = (sol @ K_test).T
        # loss = np.mean(np.square(preds - y_test_onehot.numpy()))
        # print("Round " + str(i) + " MSE: ", loss) # Loss function
        # y_pred = torch.from_numpy(preds)
        # preds = torch.argmax(y_pred, dim=-1)
        # # top_k_acc = top_k_accuracy_score(y_test, y_pred, k=top_k_val)
        #
        # count = torch.sum(y_test == preds).numpy()
        # acc = count / len(y_test)
        # print("Round " + str(i) + " Acc: ", acc)
        # print("Round " + str(i) + f" Top {top_k_val} Acc: ", top_k_acc)
        # print()
        # acc, loss = eval(X_train, X_test, L, M, y_test, y_test_onehot, sol, i)
        #
        # metrics = {
        #     'validation/accuracy': acc,
        #     'validation/loss': loss,
        #     'rfm_iter': i
        # }
        #
        # if X_test_alt is not None:
        #     # K_test = kernel_fn(X_train, X_test_alt, L, torch.from_numpy(M)).numpy()
        #     # preds = (sol @ K_test).T
        #     # loss = np.mean(np.square(preds - y_test_alt_onehot.numpy()))
        #     # print("Round " + str(i) + " MSE: ", loss)
        #     # y_pred = torch.from_numpy(preds)
        #     # preds = torch.argmax(y_pred, dim=-1)
        #     # count = torch.sum(y_test_alt == preds).numpy()
        #     # acc = count / len(y_test_alt)
        #     # print("Round " + str(i) + " Alt Acc: ", acc)
        #     acc, loss = eval(X_train, X_test_alt, L, M, y_test_alt, y_test_alt_onehot, sol, i, log_key=' Alt')
        #
        #     metrics['validation/alt_accuracy'] = acc
        #     metrics['validation/alt_loss'] = loss
        #
        # if X_test_alt2 is not None:
        #     acc, loss = eval(X_train, X_test_alt2, L, M, y_test_alt2, y_test_alt2_onehot, sol, i, log_key=' Alt 2')
        #     metrics['validation/alt2_accuracy'] = acc
        #     metrics['validation/alt2_loss'] = loss
        #
        # print()
        #
        # if wandb is not None:
        #     wandb.log(metrics)
        M  = get_grads(X_train, epro.weight.T.cpu().numpy(), L, M.cpu(), batch_size=batch_size)
        M = torch.from_numpy(M).to('cuda')

    K_train = kernel_fn(X_train, X_train, L, torch.from_numpy(M)).numpy()
    if use_jac_reg:
        G_reg = get_grad_reg(X_train, L, torch.from_numpy(M)).numpy()
        sol = np.linalg.inv(K_train.T @ K_train + reg * np.eye(len(K_train)) + agop_weight*G_reg) @ K_train @ y_train_onehot.numpy()
        sol = sol.T
    else:
        sol = solve(K_train + reg * np.eye(len(K_train)), y_train_onehot).T

    K_test = kernel_fn(X_train, X_test, L, torch.from_numpy(M)).numpy()
    preds = (sol @ K_test).T
    mse = np.mean(np.square(preds - y_test_onehot.numpy()))
    print("Final MSE: ", mse)
    y_pred = torch.from_numpy(preds)
    preds = torch.argmax(y_pred, dim=-1)
    count = torch.sum(y_test == preds).numpy()
    print(" Final Acc: ", count / len(y_test))

    metrics = {
        'validation/accuracy': count / len(y_test),
        'validation/loss': mse,
        'rfm_iter': iters
    }

    if X_test_alt is not None:
        K_test = kernel_fn(X_train, X_test_alt, L, torch.from_numpy(M)).numpy()
        preds = (sol @ K_test).T
        loss = np.mean(np.square(preds - y_test_alt_onehot.numpy()))
        print("Round " + str(i) + " MSE: ", loss)
        y_pred = torch.from_numpy(preds)
        preds = torch.argmax(y_pred, dim=-1)
        count = torch.sum(y_test_alt == preds).numpy()
        acc = count / len(y_test_alt)
        print("Round " + str(i) + " Alt Acc: ", acc)
        metrics['validation/alt_accuracy'] = acc
        metrics['validation/alt_loss'] = loss

    return mse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', default='belkinlab')
    parser.add_argument('--wandb_proj_name', default='mar19-rfm-grokking')
    parser.add_argument('--wandb_offline', default=False, action='store_true')
    parser.add_argument('--out_dir', default='./')
    parser.add_argument('--operation', '-op', default="x+y")
    parser.add_argument('--prime', '-p', default=97, type=int)
    parser.add_argument('--num_tokens', '-n', default=97, type=int)
    parser.add_argument('--training_fraction', default=0.5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--agop_batch_size', default=32, type=int)
    parser.add_argument('--iters', default=50, type=int)
    parser.add_argument('--rfm_batch_size', default=2, type=int)
    parser.add_argument('--ridge', default=1e-3, type=float)
    parser.add_argument('--bandwidth', default=1, type=float)
    parser.add_argument('--agop_weight', default=10, type=float)
    parser.add_argument('--use_jac_reg', default=False, action='store_true')
    args = parser.parse_args()

    mode = 'online'
    if args.wandb_offline:
        mode = 'offline'

    wandb.init(entity=args.wandb_entity, project=args.wandb_proj_name, mode=mode, config=args,
               dir=args.out_dir)

    wandb.run.name = f'{wandb.run.id} - n_toks: {args.num_tokens}, train_frac: {args.training_fraction}, p: {args.prime}, agop_weight: {args.agop_weight}, jac_reg: {args.use_jac_reg}'

    train_loader, agop_loader, val_loader, val_loader1, \
        context_len, train_dataset, val_dataset, \
        X_train, y_train, X_test, y_test = \
            get_augmented_data_with_agop_loader(args.operation, args.prime, args.num_tokens,
                                                args.training_fraction, args.batch_size,
                                                args.agop_batch_size, drop_last=False)

    if args.num_tokens > args.prime:
        X_test2 = []
        y_test2 = []
        for idx in range(y_test.shape[0]):
            if len((X_test[idx] >= args.prime).nonzero()) > 0:
                X_test2.append(X_test[idx])
                y_test2.append(y_test[idx])
        X_test2 = torch.stack(X_test2)
        y_test2 = torch.stack(y_test2)

        X_test2 = F.one_hot(X_test2, args.num_tokens).view(-1, 2*args.num_tokens).double()
        y_test2 = F.one_hot(y_test2, args.prime).double()

        X_test1 = []
        y_test1 = []
        for (inputs, labels) in val_loader1:
            X_test1.append(F.one_hot(inputs, args.num_tokens).view(-1, 2*args.num_tokens).double())
            y_test1.append(F.one_hot(labels, args.prime).double())

        X_test1 = torch.concatenate(X_test1)
        y_test1 = torch.concatenate(y_test1)

        print(X_train.shape)
        print(X_test.shape)
        print(X_test1.shape)
        print(X_test2.shape)
    else:
        X_test2, y_test2, X_test1, y_test1 = None, None, None, None

    X_train = F.one_hot(X_train, args.num_tokens).view(-1, 2*args.num_tokens).double()
    y_train = F.one_hot(y_train, args.prime).double()
    X_test = F.one_hot(X_test, args.num_tokens).view(-1, 2*args.num_tokens).double()
    y_test = F.one_hot(y_test, args.prime).double()

    num_classes = args.prime
    rfm(X_train, y_train, X_test, y_test, num_classes, wandb,
        iters=args.iters, name=None, batch_size=args.rfm_batch_size, reg=args.ridge,
        train_acc=True, L=args.bandwidth, agop_weight=args.agop_weight, X_test_alt=X_test1, y_test_alt_onehot=y_test1,
        use_jac_reg=args.use_jac_reg, X_test_alt2=X_test2, y_test_alt2_onehot=y_test2)
