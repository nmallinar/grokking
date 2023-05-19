import numpy as np
import torch
from scipy.linalg import solve
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from empirical_ntk import get_eNTK_batched, get_eNTK_grad

torch.set_default_dtype(torch.float64)

def main(model, num_tokens, dim_model, train_loader, test_loader, wandb, device):
    embedding_layer = nn.Embedding(num_tokens, dim_model).requires_grad_(False)

    rfm(model, train_loader, test_loader, embedding_layer, num_tokens, wandb, device,
        iters=1000, name=None, batch_size=128, reg=1e-1,
        train_acc=True)

def rfm(model, train_loader, test_loader, embedding_layer, num_classes, wandb, device,
        iters=3, name=None, batch_size=128, reg=1e-3,
        train_acc=False):

    X_train, y_train, true_y_train = get_data(train_loader, embedding_layer, num_classes)
    X_test, y_test, true_y_test = get_data(test_loader, embedding_layer, num_classes)

    if len(X_train.shape) == 3:
        n, _, d = X_train.shape
    else:
        n, d = X_train.shape

    M = np.eye(d, dtype='float64')

    for i in range(iters):
        grad = get_eNTK_grad(model, X_train[:2], X_train[:2], num_classes)
        import ipdb; ipdb.set_trace()

        K_train = get_eNTK_batched(model, X_train, num_classes, device, batch_size).numpy()
        sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T

        if train_acc:
            preds = (sol @ K_train).T
            loss = np.mean(np.square(preds - y_train.numpy()))
            print("Round " + str(i) + " Train MSE: ", loss) # Loss function
            y_pred = torch.from_numpy(preds)
            preds = torch.argmax(y_pred, dim=-1)
            labels = torch.argmax(y_train, dim=-1)
            count = torch.sum(true_y_train == preds).numpy()
            acc = count / len(labels)
            print("Round " + str(i) + " Train Acc: ", acc)

            metrics = {
                'training/accuracy': acc,
                'training/loss': loss,
                'rfm_iter': i
            }
            wandb.log(metrics)

        K_test = get_eNTK_batched(model, X_train, num_classes, device, batch_size, val_dataset=X_test).numpy()
        preds = (sol @ K_test).T
        loss = np.mean(np.square(preds - y_test.numpy()))
        print("Round " + str(i) + " MSE: ", loss) # Loss function
        y_pred = torch.from_numpy(preds)
        preds = torch.argmax(y_pred, dim=-1)
        labels = torch.argmax(y_test, dim=-1)
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


def get_data(loader, embedding_layer, num_classes):
    X = []
    y = []
    true_y = []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        batch_size = inputs.shape[0]
        X.append(embedding_layer(inputs).view(batch_size, -1))
        true_y.append(labels)
        y.append(F.one_hot(labels, num_classes).double())
    return torch.cat(X, dim=0).double(), torch.cat(y, dim=0).double(), torch.cat(true_y, dim=0)
