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

def get_data_with_agop_loader(operation, prime, training_fraction, batch_size, agop_batch_size):
    inputs, labels = operation_mod_p_data(operation, prime, prime, prime+1)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    context_len = inputs.shape[1]

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = min(batch_size, ceil(len(dataset) / 2))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    agop_loader = torch.utils.data.DataLoader(train_dataset, batch_size=agop_batch_size, shuffle=True, drop_last=True)
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
