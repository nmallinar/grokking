import numpy as np
import torch
from numpy.linalg import solve
import classic_kernel
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
#import hickle

def main(num_tokens, dim_model, train_loader, test_loader, wandb, L):
    embedding_layer = nn.Embedding(num_tokens, dim_model).requires_grad_(False)

    rfm(train_loader, test_loader, embedding_layer, num_tokens, wandb,
            iters=1000, name=None, batch_size=2, reg=1e-3,
            train_acc=True, L=L)

def laplace_kernel_M(pair1, pair2, bandwidth, M):
    return classic_kernel.laplacian_M(pair1, pair2, bandwidth, M)

def gaussian_kernel_M(pair1, pair2, bandwidth, M):
    return classic_kernel.gaussian_M(pair1, pair2, bandwidth, M)

def get_grads(X, sol, L, P, batch_size=2):
    M = 0.

    num_samples = 20000
    indices = np.random.randint(len(X), size=num_samples)

    if len(X) > len(indices):
        x = X[indices, :]
    else:
        x = X

    K = laplace_kernel_M(X, x, L, P)

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
    for i in tqdm(range(len(batches))):
        # grad = batches[i].cuda()
        grad = batches[i]
        gradT = torch.transpose(grad, 1, 2)
        M += torch.sum(gradT @ grad, dim=0).cpu()
        del grad, gradT
    torch.cuda.empty_cache()
    M /= len(G)
    M = M.numpy()

    return M


def rfm(train_loader, test_loader, embedding_layer, num_classes, wandb,
        iters=3, name=None, batch_size=2, reg=1e-3,
        train_acc=False, L=1):


    label_proj = torch.randn(num_classes, num_classes, dtype=torch.double)
    # label_proj = embedding_layer.weight.double()
    X_train, y_train, true_y_train = get_data(train_loader, embedding_layer, num_classes, label_proj)
    X_test, y_test, true_y_test = get_data(test_loader, embedding_layer, num_classes, label_proj)
    train_mean = X_train.mean(0, keepdim=True)
    train_std = X_train.std(0, keepdim=True)
    X_train -= train_mean
    X_train /= train_std
    X_test -= train_mean
    X_test /= train_std
    y_train -= 1.0/num_classes
    y_test -= 1.0/num_classes

    n, d = X_train.shape

    M = np.eye(d, dtype='float64')
    # EEt = embedding_layer.weight @ embedding_layer.weight.T
    # labelE = embedding_layer.weight.T @ torch.linalg.pinv(EEt)
    # labelE = label_proj.T @ torch.linalg.pinv(label_proj @ label_proj.T)
    labelE = np.linalg.pinv(label_proj)
    # labelE = torch.eye(num_classes).double()
    # labelE = label_proj.T
    for i in range(iters):
        K_train = laplace_kernel_M(X_train, X_train, L, torch.from_numpy(M)).numpy()
        sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T

        if train_acc:
            preds = (sol @ K_train).T
            loss = np.mean(np.square(preds - y_train.numpy()))
            print("Round " + str(i) + " Train MSE: ", loss) # Loss function
            y_pred = torch.from_numpy(preds)
            # preds = torch.argmax(y_pred, dim=-1)
            preds = torch.argmax(y_pred@labelE, dim=-1)
            labels = torch.argmax(y_train@labelE, dim=-1)
            count = torch.sum(true_y_train == preds).numpy()
            acc = count / len(labels)
            print("Round " + str(i) + " Train Acc: ", acc)

            metrics = {
                'training/accuracy': acc,
                'training/loss': loss,
                'rfm_iter': i
            }
            wandb.log(metrics)

        K_test = laplace_kernel_M(X_train, X_test, L, torch.from_numpy(M)).numpy()
        preds = (sol @ K_test).T
        loss = np.mean(np.square(preds - y_test.numpy()))
        print("Round " + str(i) + " MSE: ", loss) # Loss function
        y_pred = torch.from_numpy(preds)
        # preds = torch.argmax(y_pred, dim=-1)
        preds = torch.argmax(y_pred@labelE, dim=-1)
        labels = torch.argmax(y_test@labelE, dim=-1)
        count = torch.sum(true_y_test == preds).numpy()
        acc = count / len(labels)
        print("Round " + str(i) + " Acc: ", acc)

        metrics = {
            'validation/accuracy': acc,
            'validation/loss': loss,
            'rfm_iter': i
        }
        wandb.log(metrics)


        M  = get_grads(X_train, sol, L, torch.from_numpy(M), batch_size=batch_size)
        # if name is not None:
        #     hickle.dump(M, 'saved_Ms/M_' + name + '_' + str(i) + '.h')

    K_train = laplace_kernel_M(X_train, X_train, L, torch.from_numpy(M)).numpy()
    sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T
    K_test = laplace_kernel_M(X_train, X_test, L, torch.from_numpy(M)).numpy()
    preds = (sol @ K_test).T
    mse = np.mean(np.square(preds - y_test.numpy()))
    print("Final MSE: ", mse)
    y_pred = torch.from_numpy(preds)
    preds = torch.argmax(y_pred, dim=-1)
    labels = torch.argmax(y_test, dim=-1)
    count = torch.sum(labels == preds).numpy()
    print(" Final Acc: ", count / len(labels))
    return mse


def get_data(loader, embedding_layer, num_classes, label_proj):
    X = []
    y = []
    true_y = []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        batch_size = inputs.shape[0]
        inputs = torch.stack((inputs[:,0], inputs[:,2]), dim=1)
        # embedding_layer(inputs): (n, seq_len, d)
        X.append(torch.mean(embedding_layer(inputs), 1))
        # X.append(embedding_layer(inputs).view(batch_size, -1))

        true_y.append(labels)
        # y.append(F.one_hot(labels, num_classes))
        # embedding_layer(labels): (n, d)
        y.append(F.one_hot(labels, num_classes).double() @ label_proj)
    return torch.cat(X, dim=0).double(), torch.cat(y, dim=0).double(), torch.cat(true_y, dim=0)
