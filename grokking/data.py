from math import ceil
import torch

torch.set_default_dtype(torch.float64)

DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: (x*y % p, y, x),
    "(x//y)if(y%2==1)else(x-y)": lambda x, y, _: torch.where(y % 2 == 1, x // y, x - y)
}

ALL_MODULO_OPERATIONS = {
    "x+y": lambda x, y, _: (x, y, x + y),
    "x-y": lambda x, y, _: (x, y, x - y),
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

def operation_mod_p_data_augmented2(operation: str, p: int, n: int):
    """
    x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
    x◦y (mod p) for 0 <= x, y < p otherwise
    """
    x1 = torch.arange(0, p)
    y1 = torch.arange(0 if not operation in DIVISION_MODULO_OPERATIONS else 1, p)

    x1, y1 = torch.cartesian_prod(x1, y1).T
    x1, y1, z1 = ALL_OPERATIONS[operation](x1, y1, p)
    results1 = z1.remainder(p)

    inputs1 = torch.stack([x1, y1], dim=1)
    labels1 = results1

    if n > p:
        x2 = torch.arange(p, n)
        y2 = torch.arange(p, n)
        x2, y2 = torch.cartesian_prod(x2, y2).T

        _x2 = torch.arange(0, p)
        _y2 = torch.arange(p, n)
        _x2, _y2 = torch.cartesian_prod(_x2, _y2).T

        __x2 = torch.arange(p, n)
        __y2 = torch.arange(0, p)
        __x2, __y2 = torch.cartesian_prod(__x2, __y2).T

        x2 = torch.cat([x2, _x2, __x2])
        y2 = torch.cat([y2, _y2, __y2])
        x2, y2, z2 = ALL_OPERATIONS[operation](x2, y2, p)
        results2 = z2.remainder(p)

        inputs2 = torch.stack([x2, y2], dim=1)
        labels2 = results2
    else:
        inputs2, labels2 = None, None

    return inputs1, labels1, inputs2, labels2

def operation_mod_p_data_augmented(operation: str, p: int, n: int):
    # buggy
    """
    x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
    x◦y (mod p) for 0 <= x, y < p otherwise
    """
    x1 = torch.arange(0, p)
    y1 = torch.arange(0 if not operation in DIVISION_MODULO_OPERATIONS else 1, p)

    if n > p:
        x2 = torch.arange(p, n)
        y2 = torch.arange(p, n)
        x2, y2 = torch.cartesian_prod(x2, y2).T
        x2, y2, z2 = ALL_OPERATIONS[operation](x2, y2, p)
        results2 = z2.remainder(p)
        inputs2 = torch.stack([x2, y2], dim=1)
        labels2 = results2
    else:
        inputs2 = None
        labels2 = None

    x1, y1 = torch.cartesian_prod(x1, y1).T
    x1, y1, z1 = ALL_OPERATIONS[operation](x1, y1, p)

    results1 = z1.remainder(p)

    inputs1 = torch.stack([x1, y1], dim=1)
    labels1 = results1

    return inputs1, labels1, inputs2, labels2

    # x = torch.arange(0, n)
    # y = torch.arange(0 if not operation in DIVISION_MODULO_OPERATIONS else 1, n)
    # x, y = torch.cartesian_prod(x, y).T
    #
    # x, y, z = ALL_OPERATIONS[operation](x, y, p)
    # results = z.remainder(p)
    #
    # inputs = torch.stack([x, y], dim=1)
    # labels = results
    #
    # return inputs, labels

def operation_mod_p_data(operation: str, p: int, eq_token: int, op_token: int):
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

def get_data(operation: str, prime: int, training_fraction: float, batch_size: int):
    inputs, labels = operation_mod_p_data(operation, prime, prime, prime+1)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    context_len = inputs.shape[1]

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = min(batch_size, ceil(len(dataset) / 2))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, context_len, train_dataset, val_dataset

def get_data_with_agop_loader(operation, prime, training_fraction, batch_size, agop_batch_size, drop_last=True):
    inputs, labels = operation_mod_p_data(operation, prime, prime, prime+1)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    context_len = inputs.shape[1]

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = min(batch_size, ceil(len(dataset) / 2))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    agop_loader = torch.utils.data.DataLoader(train_dataset, batch_size=agop_batch_size, shuffle=True, drop_last=drop_last)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_feats = []
    train_labels = []
    val_feats = []
    val_labels = []
    for batch in train_loader:
        train_feats.append(batch[0])
        train_labels.append(batch[1])
    for batch in val_loader:
        val_feats.append(batch[0])
        val_labels.append(batch[1])
    train_feats = torch.cat(train_feats)
    train_labels = torch.cat(train_labels)
    val_feats = torch.cat(val_feats)
    val_labels = torch.cat(val_labels)

    return train_loader, agop_loader, val_loader, context_len, train_dataset, val_dataset, train_feats, train_labels, val_feats, val_labels

def get_augmented_data_with_agop_loader(operation, prime, n, training_fraction, batch_size, agop_batch_size, drop_last=True):
    inputs1, labels1, inputs2, labels2 = operation_mod_p_data_augmented2(operation, prime, n)

    # inputs, labels = operation_mod_p_data_augmented(operation, prime, n)
    dataset1 = torch.utils.data.TensorDataset(inputs1, labels1)

    context_len = inputs1.shape[1]
    train_size1 = int(training_fraction * len(dataset1))
    val_size1 = len(dataset1) - train_size1

    train_dataset1, val_dataset1 = torch.utils.data.random_split(dataset1, [train_size1, val_size1])

    if n > prime:
        dataset2 = torch.utils.data.TensorDataset(inputs2, labels2)
        train_size2 = int(training_fraction * len(dataset2))
        val_size2 = len(dataset2) - train_size2
        train_dataset2, val_dataset2 = torch.utils.data.random_split(dataset2, [train_size2, val_size2])
        batch_size = min(batch_size, ceil((len(dataset1) + len(dataset2))/2))
        train_dataset = torch.utils.data.ConcatDataset([train_dataset1, train_dataset2])
        val_dataset = torch.utils.data.ConcatDataset([val_dataset1, val_dataset2])
    else:
        batch_size = min(batch_size, ceil(len(dataset1)/2))
        train_dataset = train_dataset1
        val_dataset = val_dataset1

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    agop_loader = torch.utils.data.DataLoader(train_dataset, batch_size=agop_batch_size, shuffle=True, drop_last=drop_last)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_loader1 = torch.utils.data.DataLoader(val_dataset1, batch_size=batch_size, shuffle=False)

    train_feats = []
    train_labels = []
    val_feats = []
    val_labels = []
    for batch in train_loader:
        train_feats.append(batch[0])
        train_labels.append(batch[1])
    for batch in val_loader:
        val_feats.append(batch[0])
        val_labels.append(batch[1])
    train_feats = torch.cat(train_feats)
    train_labels = torch.cat(train_labels)
    val_feats = torch.cat(val_feats)
    val_labels = torch.cat(val_labels)

    return train_loader, agop_loader, val_loader, val_loader1, context_len, train_dataset, val_dataset, train_feats, train_labels, val_feats, val_labels
