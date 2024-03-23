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
from inf_ntk import ntk_fn, jax_ntk_fn
from sklearn.metrics import top_k_accuracy_score

def main(num_tokens, dim_model, train_loader, test_loader, wandb, L, embedding_layer, agop_weight):
    rfm(train_loader, test_loader, embedding_layer, num_tokens, wandb,
            iters=10, name=None, batch_size=2, reg=1e-3,
            train_acc=True, L=L, agop_weight=agop_weight)

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
    # return ntk_M(pair1, pair2, 1, M)
    return jax_ntk_M(pair1, pair2, 5 , M)
    # return laplace_kernel_M(pair1, pair2, bandwidth, M)
    #return gaussian_kernel_M(pair1, pair2, bandwidth, M)

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


def rfm(train_loader, test_loader, embedding_layer, num_classes, wandb,
        iters=3, name=None, batch_size=2, reg=1e-3,
        train_acc=False, L=1, agop_weight=1e-5):
    num_classes = 31
    # X_train, y_train, true_y_train = get_data(train_loader, embedding_layer, num_classes)
    # X_test, y_test, true_y_test = get_data(test_loader, embedding_layer, num_classes)

    # X_train = torch.tensor(np.load('data/10k/base_train_data.npy')).double()
    # true_y_train = torch.tensor(np.load('data/10k/base_train_labels.npy'))
    # y_train = F.one_hot(true_y_train, num_classes).double().double()

    X_test = torch.tensor(np.load('data/100k_4layer/base_val_feats.npy')).double()
    true_y_test = torch.tensor(np.load('data/100k_4layer/base_val_labels.npy'))
    y_test = F.one_hot(true_y_test, num_classes).double()

    X_train = torch.tensor(np.load('data/100k_4layer/synthetic_data.npy')).double()
    y_train = torch.tensor(np.load('data/100k_4layer/synthetic_labels.npy')).double()
    # import ipdb; ipdb.set_trace()
    X_train = X_train[:10000]
    y_train = y_train[:10000]
    true_y_train = torch.tensor(np.argmax(y_train, axis=-1))
    n, d = X_train.shape

    M = np.eye(d, dtype='float64')
    # M = np.load('grokking_outputs/nov27_proper_agop/ep_1000_neural_feature_matrix.npy').astype(np.float64)
    # l, v = np.linalg.eigh(M)
    # idx = np.argsort(l)[::-1]
    # l = l[idx]
    # v = v[idx]
    # # M = v @ np.diag(l / 256) @ v.T
    # M = v[:,:128] @ np.diag(l[:128]) @ v[:,:128].T

    # M = scipy.linalg.sqrtm(M)
    # M = np.maximum(M, 0)
    # M = M.T @ M
    for i in range(iters):
        # G_reg = get_grad_reg(X_train, L, torch.from_numpy(M)).numpy()
        K_train = kernel_fn(X_train, X_train, L, torch.from_numpy(M)).numpy()
        # sol = np.linalg.inv(K_train.T @ K_train + agop_weight*G_reg) @ K_train @ y_train.numpy()
        # sol = sol.T
        sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T

        if train_acc:
            preds = (sol @ K_train).T
            loss = np.mean(np.square(preds - y_train.numpy()))
            y_pred = torch.from_numpy(preds)
            preds = torch.argmax(y_pred, dim=-1)
            labels = torch.argmax(y_train, dim=-1)
            count = torch.sum(true_y_train == preds).numpy()
            acc = count / len(labels)

            #if i % 50 == 0:
            print("Round " + str(i) + " Train MSE: ", loss) # Loss function
            print("Round " + str(i) + " Train Acc: ", acc)

            metrics = {
                'training/accuracy': acc,
                'training/loss': loss,
                'rfm_iter': i
            }
            wandb.log(metrics)

        # if i % 50 == 0:
        top_k_val = 5
        K_test = kernel_fn(X_train, X_test, L, torch.from_numpy(M)).numpy()
        preds = (sol @ K_test).T
        loss = np.mean(np.square(preds - y_test.numpy()))
        print("Round " + str(i) + " MSE: ", loss) # Loss function
        y_pred = torch.from_numpy(preds)
        preds = torch.argmax(y_pred, dim=-1)
        labels = torch.argmax(y_test, dim=-1)
        top_k_acc = top_k_accuracy_score(labels, y_pred, k=top_k_val)

        count = torch.sum(true_y_test == preds).numpy()
        acc = count / len(labels)
        print("Round " + str(i) + " Acc: ", acc)
        print("Round " + str(i) + f" Top {top_k_val} Acc: ", top_k_acc)
        print()

        metrics = {
            'validation/accuracy': acc,
            'validation/loss': loss,
            'rfm_iter': i
        }
        wandb.log(metrics)


        M  = get_grads(X_train, sol, L, torch.from_numpy(M), batch_size=batch_size)
        # plt.imshow(M)
        # plt.savefig(f'M_unique_round{i}.pdf')

        # if name is not None:
        #     hickle.dump(M, 'saved_Ms/M_' + name + '_' + str(i) + '.h')

    K_train = kernel_fn(X_train, X_train, L, torch.from_numpy(M)).numpy()
    sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T
    K_test = kernel_fn(X_train, X_test, L, torch.from_numpy(M)).numpy()
    preds = (sol @ K_test).T
    mse = np.mean(np.square(preds - y_test.numpy()))
    print("Final MSE: ", mse)
    y_pred = torch.from_numpy(preds)
    preds = torch.argmax(y_pred, dim=-1)
    labels = torch.argmax(y_test, dim=-1)
    count = torch.sum(labels == preds).numpy()
    print(" Final Acc: ", count / len(labels))
    return mse


def get_data(loader, embedding_layer, num_classes):
    X = []
    y = []
    true_y = []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        batch_size = inputs.shape[0]
        # inputs = torch.stack((inputs[:,0], inputs[:,2]), dim=1)
        X.append(embedding_layer(inputs).view(batch_size, -1))

        true_y.append(labels)
        y.append(F.one_hot(labels, num_classes).double())
    return torch.cat(X, dim=0).double(), torch.cat(y, dim=0).double(), torch.cat(true_y, dim=0)
