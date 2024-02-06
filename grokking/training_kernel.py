import numpy as np
import scipy.linalg
import torch
import os

from classic_kernel import euclidean_distances, laplacian, gaussian
from inf_ntk import ep3_ntk_relu

DATA_DIR = '/scratch/bbjr/mallina1/grokking_output/feb2-grokking/tmwu613g/epoch_460'
n_classes = 31
n_centers = 40000
n_train = 1000000
ridge = 0.0

# kernel_fn = lambda x, z: ep3_ntk_relu(x, z, depth=10)
kernel_fn = lambda x, z: euclidean_distances(x, z, squared=True)

X_train = torch.tensor(np.load(os.path.join(DATA_DIR, 'synthetic_data.npy'))).float()
y_train = torch.tensor(np.load(os.path.join(DATA_DIR, 'synthetic_labels.npy'))).float()

perm_idx = torch.randperm(X_train.shape[0])[:n_train]
X_train = X_train[perm_idx]
y_train = y_train[perm_idx]
y_train_idx = y_train.argmax(-1)

X_test = torch.tensor(np.load(os.path.join(DATA_DIR, 'base_val_feats.npy'))).float()
y_test_idx = torch.tensor(np.load(os.path.join(DATA_DIR, 'base_val_labels.npy')))
y_test = one_hot(y_test_idx, n_classes).float()

right_nfm = torch.tensor(np.load(os.path.join(DATA_DIR, 'right_nfm.npy'))).float()
right_agop = torch.tensor(np.load(os.path.join(DATA_DIR, 'right_agop_0.npy'))).float()

# (z, d)
perm_idx = torch.randperm(n_train)[:n_centers]
centers = X_train[perm_idx]

# (z, d) * (d, n) = (z, n)
K_zx = kernel_fn(centers, X_train)

# (n, z) * (z, n) = (n, n)
# K_xx = K_zx.T @ K_zx

K_zz = K_zx @ K_zx.T

alpha = torch.linalg.pinv(K_zz + ridge * torch.eye(n_train), hermitian=True) @ K_zx @ y_train
train_preds = K_xx @ alpha
train_preds = train_preds.argmax(-1)
accuracy = (train_preds == y_train_idx).mean()
print(f'Training accuracy: {accuracy}')

del K_xx, K_zx

# (m, n)
K_te = kernel_fn(X_test, centers)
test_preds = K_te @ alpha
test_preds = test_preds.argmax(-1)
accuracy = (test_preds == y_test_idx).mean()
print(f'Testing accuracy: {accuracy}')
