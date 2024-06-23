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

def operation_mod_p_data_binarized(operation: str, p: int, training_fraction):
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
    labels = results

    X_tr, y_tr, X_te, y_te = make_data_splits(inputs, labels, training_fraction)
    cls_labels = {}
    te_cls_labels = {}
    for i in range(p):
        cls_idx = torch.where(y_tr == i)
        not_cls_idx = torch.where(y_tr != i)
        cls_y = y_tr.clone()
        cls_y[cls_idx] = 1
        cls_y[not_cls_idx] = -1
        cls_labels[i] = cls_y
        # cls_labels[i] = F.one_hot(cls_y, 2).double()

        cls_idx = torch.where(y_te == i)
        not_cls_idx = torch.where(y_te != i)
        cls_y = y_te.clone()
        cls_y[cls_idx] = 1
        cls_y[not_cls_idx] = -1
        te_cls_labels[i] = cls_y

    X_tr = F.one_hot(X_tr, p).view(-1, 2*p).double()
    y_tr = F.one_hot(y_tr, p).double()
    X_te = F.one_hot(X_te, p).view(-1, 2*p).double()
    y_te = F.one_hot(y_te, p).double()

    return X_tr, cls_labels, y_tr, X_te, te_cls_labels, y_te

def multitask_op_mod_p_data(op1, op2, p, train_frac_per_op):
    inp1, lab1 = operation_mod_p_data(op1, p)
    X_tr1, y_tr1, X_te1, y_te1 = make_data_splits(inp1, lab1, 0.5)
    X_tr1 = F.one_hot(X_tr1, p).view(-1, 2*p).double()
    zeros = torch.zeros((X_tr1.shape[0],1))
    X_tr1 = torch.hstack((X_tr1, zeros))
    y_tr1 = F.one_hot(y_tr1, p).double()
    y_tr1 = torch.hstack((y_tr1, torch.zeros((y_tr1.shape[0], 2)).double()))

    n_train2 = 500
    tr_inp2 = torch.randint(0, 2, (n_train2, 2*p))
    y_tr2 = torch.logical_xor(tr_inp2[:, :p], tr_inp2[:, p:]).double()
    ones = torch.ones((tr_inp2.shape[0],1))
    X_tr2 = torch.hstack((tr_inp2, ones))
    even_parity = torch.sum(y_tr2, axis=-1) % 2 == 0
    y_tr2 = (~even_parity).long()
    y_tr2 = torch.hstack((torch.zeros(y_tr2.shape[0], p).double(), F.one_hot(y_tr2, 2)))

    X_tr = torch.vstack((X_tr1, X_tr2))
    y_tr = torch.vstack((y_tr1, y_tr2))


    X_te1 = F.one_hot(X_te1, p).view(-1, 2*p).double()
    zeros = torch.zeros((X_te1.shape[0],1))
    X_te1 = torch.hstack((X_te1, zeros))
    y_te1 = F.one_hot(y_te1, p).double()
    y_te1 = torch.hstack((y_te1, torch.zeros((y_te1.shape[0], 2)).double()))

    n_test2 = 500
    te_inp2 = torch.randint(0, 2, (n_test2, 2*p))
    y_te2 = torch.logical_xor(te_inp2[:, :p], te_inp2[:, p:]).double()
    ones = torch.ones((te_inp2.shape[0],1))
    X_te2 = torch.hstack((te_inp2, ones))
    even_parity = torch.sum(y_te2, axis=-1) % 2 == 0
    y_te2 = (~even_parity).long()
    y_te2 = torch.hstack((torch.zeros(y_te2.shape[0], p).double(), F.one_hot(y_te2, 2)))

    return X_tr, y_tr, X_te1, y_te1, X_te2, y_te2

'''
multitask with XOR using same one-hot encoded samples
'''
# def multitask_op_mod_p_data(op1, op2, p, train_frac_per_op):
#     inp1, lab1 = operation_mod_p_data(op1, p)
#     inp2 = inp1.clone()
#     lab2_dig1 = (inp1[:,0] % 2 == 0).long()
#     lab2_dig2 = (inp1[:,1] % 2 == 0).long()
#     lab2 = torch.logical_xor(lab2_dig1, lab2_dig2).long()
#
#     X_tr1, y_tr1, X_te1, y_te1, perm = make_data_splits_with_perm(inp1, lab1, 0.5)
#     X_tr2, y_tr2, X_te2, y_te2, _ = make_data_splits_with_perm(inp2, lab2, 0.5, perm=perm)
#     X_tr1 = F.one_hot(X_tr1, p).view(-1, 2*p).double()
#     X_tr2 = F.one_hot(X_tr2, p).view(-1, 2*p).double()
#     zeros = torch.zeros((X_tr1.shape[0],1))
#     ones = torch.ones((X_tr2.shape[0],1))
#
#     X_tr1 = torch.hstack((X_tr1, zeros))
#     X_tr2 = torch.hstack((X_tr2, ones))
#     X_tr = torch.vstack((X_tr1, X_tr2))
#
#     y_tr1 = torch.hstack((F.one_hot(y_tr1, p).double(), torch.zeros((y_tr1.shape[0], 2)).double()))
#     y_tr2 = torch.hstack((torch.zeros(y_tr2.shape[0], p).double(), F.one_hot(y_tr2, 2).double()))
#     y_tr = torch.vstack((y_tr1, y_tr2))
#
#     X_te1 = F.one_hot(X_te1, p).view(-1, 2*p).double()
#     X_te2 = F.one_hot(X_te2, p).view(-1, 2*p).double()
#     zeros = torch.zeros((X_te1.shape[0],1))
#     ones = torch.ones((X_te2.shape[0],1))
#     X_te1 = torch.hstack((X_te1, zeros))
#     X_te2 = torch.hstack((X_te2, ones))
#     y_te1 = torch.hstack((F.one_hot(y_te1, p).double(), torch.zeros((y_te1.shape[0], 2)).double()))
#     y_te2 = torch.hstack((torch.zeros(y_te2.shape[0], p).double(), F.one_hot(y_te2, 2).double()))
#
#     X_te = torch.vstack((X_te1, X_te2))
#     y_te = torch.vstack((y_te1, y_te2))
#
#     return X_tr, y_tr, X_te1, y_te1, X_te2, y_te2

# def multitask_op_mod_p_data(op1, op2, p, train_frac_per_op):
#     inp1, lab1 = operation_mod_p_data(op1, p)
#     inp2, lab2 = operation_mod_p_data(op2, p)
#     X_tr1, y_tr1, X_te1, y_te1, perm = make_data_splits_with_perm(inp1, lab1, train_frac_per_op)
#     X_tr2, y_tr2, X_te2, y_te2, _ = make_data_splits_with_perm(inp2, lab2, train_frac_per_op, perm=perm)
#     X_tr1 = F.one_hot(X_tr1, p).view(-1, 2*p).double()
#     X_tr2 = F.one_hot(X_tr2, p).view(-1, 2*p).double()
#     ones = torch.ones((X_tr2.shape[0],))
#     zeros = torch.zeros((X_tr1.shape[0],))
#
#     X_tr1 = torch.hstack((X_tr1, zeros.unsqueeze(-1)))
#     X_tr2 = torch.hstack((X_tr2, ones.unsqueeze(-1)))
#     X_tr = torch.vstack((X_tr1, X_tr2))
#
#     y_tr1 = F.one_hot(y_tr1, p).double()
#     y_tr2 = F.one_hot(y_tr2, p).double()
#     y_tr = torch.vstack((y_tr1, y_tr2))
#
#     X_te1 = F.one_hot(X_te1, p).view(-1, 2*p).double()
#     X_te2 = F.one_hot(X_te2, p).view(-1, 2*p).double()
#     zeros = torch.zeros((X_te1.shape[0],))
#     ones = torch.ones((X_te2.shape[0],))
#     X_te1 = torch.hstack((X_te1, zeros.unsqueeze(-1)))
#     X_te2 = torch.hstack((X_te2, ones.unsqueeze(-1)))
#     y_te1 = F.one_hot(y_te1, p).double()
#     y_te2 = F.one_hot(y_te2, p).double()
#
#     return X_tr, y_tr, X_te1, y_te1, X_te2, y_te2

def make_data_splits_with_perm(inputs, labels, training_fraction, perm=None):
    train_size = int(training_fraction * inputs.shape[0])
    val_size = inputs.shape[0] - train_size
    # X_tr, X_te, y_tr, y_te = train_test_split(inputs, labels, test_size=val_size, stratify=labels)
    # return X_tr, y_tr, X_te, y_te

    if perm is None:
        perm = torch.randperm(inputs.shape[0])

    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    return inputs[train_idx], labels[train_idx], inputs[val_idx], labels[val_idx], perm

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
