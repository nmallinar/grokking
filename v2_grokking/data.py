from math import ceil
import torch
import itertools
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: (x*y % p, y, x),
    "(x//y)if(y%2==1)else(x-y)": lambda x, y, _: torch.where(y % 2 == 1, x // y, x - y),
}

ALL_MODULO_OPERATIONS = {
    "x+y": lambda x, y, _: (x, y, x + y),
    "x-y": lambda x, y, _: (x, y, x - y),
    "x*y": lambda x, y, _: (x, y, x * y),
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

def get_s5_data():
    perms = torch.tensor(list(itertools.permutations([0, 1, 2, 3, 4])))
    perms = F.one_hot(perms, 5).double()
    perms = perms.view(-1, 5*5)

    true_f = torch.randn(perms.shape[1])
    labels = perms @ true_f

    return perms, labels

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

    return inputs, labels

# def operation_mod_p_data(operation: str, p: int):
#     """
#     x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
#     x◦y (mod p) for 0 <= x, y < p otherwise
#     """
#     x = torch.arange(0, p)
#     y = torch.arange(0 if not operation in DIVISION_MODULO_OPERATIONS else 1, p)
#     x, y = torch.cartesian_prod(x, y).T
#
#     # eq = torch.ones_like(x) * eq_token
#     # op = torch.ones_like(x) * op_token
#
#     x1, y1, z1 = ALL_OPERATIONS[operation](x, y, p)
#     results1 = z1.remainder(p)
#
#     x2, y2, z2 = ALL_OPERATIONS['x-y'](x, y, p)
#     results2 = z2.remainder(p)
#
#     inputs = torch.stack([x1, y1], dim=1)
#     # inputs2 = torch.stack([x2, op2, y2], dim=1)
#     # inputs = torch.cat([inputs1, inputs2])
#     # inputs = torch.stack([x, op, y, eq], dim=1)
#     labels = torch.stack([results1, results2], dim=1)
#     # labels1 = results1
#     # labels2 = results2
#     # labels = torch.cat([labels1, labels2])
#
#     return inputs, labels

def make_data_splits(inputs, labels, training_fraction):
    train_size = int(training_fraction * inputs.shape[0])
    val_size = inputs.shape[0] - train_size

    perm = torch.randperm(inputs.shape[0])
    # perm = torch.arange(inputs.shape[0])
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    return inputs[train_idx], labels[train_idx], inputs[val_idx], labels[val_idx]

def make_dataloader(inputs, labels, batch_size, shuffle=False, drop_last=False):
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    batch_size = min(batch_size, ceil(len(dataset) / 2))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
