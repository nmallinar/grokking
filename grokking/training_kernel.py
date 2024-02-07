import numpy as np
import scipy.linalg
import torch
import os
import torch.nn.functional as F

from classic_kernel import euclidean_distances, laplacian, gaussian
from inf_ntk import ep3_ntk_relu, jax_ntk_fn
from eigenpro2.models import KernelModel

DATA_DIR = '/scratch/bbjr/mallina1/grokking_output/feb2-grokking/tmwu613g/epoch_460'
n_classes = 31
n_centers = 10000
n_train = 5000
ridge = 0.0

#kernel_fn = lambda x, z: ep3_ntk_relu(x, z, depth=10)
kernel_fn = lambda x, z: laplacian(x, z, 1.0)
#kernel_fn = lambda x, z: euclidean_distances(x, z, squared=True)
#kernel_fn = lambda x, z: jax_ntk_fn(x, z, M=None, depth=10, bias=0, kernel_fn=None)

X_train = torch.tensor(np.load(os.path.join(DATA_DIR, 'synthetic_data.npy'))).float()
y_train = torch.tensor(np.load(os.path.join(DATA_DIR, 'synthetic_labels.npy'))).float()

perm_idx = torch.randperm(X_train.shape[0])[:n_train]
X_train = X_train[perm_idx]
y_train = y_train[perm_idx]
y_train_idx = y_train.argmax(-1)

X_test = torch.tensor(np.load(os.path.join(DATA_DIR, 'base_val_feats.npy'))).float()
y_test_idx = torch.tensor(np.load(os.path.join(DATA_DIR, 'base_val_labels.npy')))
y_test = F.one_hot(y_test_idx, n_classes).float()

right_nfm = torch.tensor(np.load(os.path.join(DATA_DIR, 'right_nfm.npy'))).float()
right_agop = torch.tensor(np.load(os.path.join(DATA_DIR, 'right_agop_0.npy'))).float()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM = torch.cuda.get_device_properties(DEVICE).total_memory//1024**3 - 1 # GPU memory in GB, keeping aside 1GB for safety
else:
    DEVICE = torch.device("cpu")
    DEV_MEM = 8 # RAM available for computing

model = KernelModel(kernel_fn, X_train, n_classes, device=DEVICE)
result = model.fit(X_train, y_train, X_test, y_test, epochs=30, print_every=1, mem_gb=DEV_MEM)

K_xx = kernel_fn(X_train, X_train).numpy()
alpha = scipy.linalg.solve(K_xx + ridge * np.eye(n_train), y_train)
# alpha = torch.linalg.pinv(K_xx + ridge * torch.eye(n_train)) @ y_train
train_preds = K_xx @ alpha
train_preds = train_preds.argmax(-1)
accuracy = sum(train_preds == y_train_idx.numpy()) / n_train
print(f'Training accuracy: {accuracy}')

# (m, n)
K_te = kernel_fn(X_test, X_train).numpy()
test_preds = K_te @ alpha
test_preds = test_preds.argmax(-1)
accuracy = sum(test_preds == y_test_idx.numpy()) / len(test_preds)
print(f'Testing accuracy: {accuracy}')
import sys
sys.exit(0)

# (z, d)
perm_idx = torch.randperm(n_train)[:n_centers]
centers = X_train[perm_idx]

# (z, d) * (d, n) = (z, n)
K_zx = kernel_fn(centers, X_train)

# (n, z) * (z, n) = (n, n)
# K_xx = K_zx.T @ K_zx

K_zz = K_zx @ K_zx.T

alpha = torch.linalg.pinv(K_zz + ridge * torch.eye(n_centers), hermitian=True) @ K_zx @ y_train
#alpha = torch.linalg.pinv(K_zx) @ y_train
train_preds = K_zx.T @ alpha
train_preds = train_preds.argmax(-1)
accuracy = sum(train_preds == y_train_idx) / n_train
print(f'Training accuracy: {accuracy}')

del K_zz

# (m, n)
K_te = kernel_fn(X_test, centers)
test_preds = K_te @ alpha
test_preds = test_preds.argmax(-1)
accuracy = sum(test_preds == y_test_idx) / len(test_preds)
print(f'Testing accuracy: {accuracy}')


#del K_te

#K_te = kernel_fn(X_test, X_train)
#test_preds = K_te @ K_zx.T @ alpha
#test_preds = test_preds.argmax(-1)
#accuracy = sum(test_preds == y_test_idx) / len(test_preds)
#print(f'Testing accuracy 2: {accuracy}')
