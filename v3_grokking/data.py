from math import ceil
import torch
import itertools
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

torch.set_default_dtype(torch.float64)

DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: (x*y % p, y, x),
    "(x//y)if(y%2==1)else(x-y)": lambda x, y, _: torch.where(y % 2 == 1, x // y, x - y)
}

ALL_MODULO_OPERATIONS = {
    "x+y": lambda x, y, _: (x, y, x + y),
    "x-y": lambda x, y, _: (x, y, x - y),
    "x*y": lambda x, y, _: (x, y, x*y),
    **DIVISION_MODULO_OPERATIONS,
    "x^2+y^2": lambda x, y, _: (x, y, x**2 + y**2),
    "x^2+xy+y^2": lambda x, y, _: (x, y, x**2 + x*y + y**2),
    "x^2+xy+y^2+x": lambda x, y, _: (x, y, x**2 + x*y + y**2 + x),
    "x^3+xy": lambda x, y, _: (x, y, x**3 + x*y),
    "x^3+xy^2+x": lambda x, y, _: (x, y, x**3 + x*y**2 + y)
}

ALL_OPERATIONS = {
    **ALL_MODULO_OPERATIONS,
}

def get_sym_group_data(order=5):
    perms = torch.tensor(list(itertools.permutations(list(torch.arange(order)))))
    perm_idx = torch.arange(len(perms))
    prod = torch.cartesian_prod(perm_idx, perm_idx)
    perms_prod = perms[prod]

    total_n = len(perms) * len(perms)
    labels = []
    for idx in range(perms_prod.shape[0]):
        prod_idx = prod[idx]
        a = perms_prod[idx][0]
        b = perms_prod[idx][1]
        c = a[b]

        lab_idx = torch.where((perms==c).all(dim=1))[0][0]
        labels.append(lab_idx)
    labels = torch.tensor(labels)

    labels = F.one_hot(labels, len(perms)).double()
    data = F.one_hot(prod, len(perms)).view(-1, len(perms)*2).double()

    return data, labels

def held_out_op_mod_p_data(operation, p):
    x_tr = torch.arange(0, p)
    y_tr = torch.arange(0 if not operation in DIVISION_MODULO_OPERATIONS else 1, p)

    x_tr = torch.cat((x_tr[0:1], x_tr[2:9], x_tr[10:]))
    y_tr = torch.cat((y_tr[0:1], y_tr[2:9], y_tr[10:]))

    # x_te = torch.tensor([1, 9])
    # y_te = torch.tensor([1, 9])

    x_tr, y_tr = torch.cartesian_prod(x_tr, y_tr).T
    # x_te, y_te = torch.cartesian_prod(x_te, y_te).T

    x_te = []
    y_te = []
    for i in [1, 9]:
        for j in range(p):
            x_te.append(i)
            y_te.append(j)

            if i != j:
                x_te.append(j)
                y_te.append(i)
    x_te = torch.tensor(x_te)
    y_te = torch.tensor(y_te)

    x_tr, y_tr, z_tr = ALL_OPERATIONS[operation](x_tr, y_tr, p)
    tr_labels = z_tr.remainder(p)

    x_te, y_te, z_te = ALL_OPERATIONS[operation](x_te, y_te, p)
    te_labels = z_te.remainder(p)

    tr_inputs = torch.stack([x_tr, y_tr], dim=1)
    te_inputs = torch.stack([x_te, y_te], dim=1)

    return tr_inputs, tr_labels, te_inputs, te_labels

def operation_mod_p_data(operation: str, p: int):
    """
    x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
    x◦y (mod p) for 0 <= x, y < p otherwise
    """
    x = torch.arange(0, p)
    y = torch.arange(0 if not operation in DIVISION_MODULO_OPERATIONS else 1, p)
    x, y = torch.cartesian_prod(x, y).T

    # eq = torch.ones_like(x) * eq_token
    # op = torch.ones_like(x) * op_token

    x, y, z = ALL_OPERATIONS[operation](x, y, p)
    results = z.remainder(p)

    inputs = torch.stack([x, y], dim=1)
    # inputs = torch.stack([x, op, y, eq], dim=1)
    labels = results
    # # labels[labels!=3] = -(1./19)
    # # labels[labels==3] = (18./19)
    # labels[labels!=3] = 0
    # labels[labels==3] = 1

    return inputs, labels

def make_data_splits(inputs, labels, training_fraction):
    train_size = int(training_fraction * inputs.shape[0])
    val_size = inputs.shape[0] - train_size
    # X_tr, X_te, y_tr, y_te = train_test_split(inputs, labels, test_size=val_size, stratify=labels)
    # return X_tr, y_tr, X_te, y_te

    perm = torch.randperm(inputs.shape[0])
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    return inputs[train_idx], labels[train_idx], inputs[val_idx], labels[val_idx]

def make_dataloader(inputs, labels, batch_size, shuffle=False, drop_last=False):
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    batch_size = min(batch_size, ceil(len(dataset) / 2))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
