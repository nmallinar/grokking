import numpy as np
import torch

def get_raw_splits(data_dir='/scratch/bbjr/mallina1/data/shakespeare_char'):
    train = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    return train, val

def get_encode_decode_fns(data_dir='/scratch/bbjr/mallina1/data/shakespeare_char'):
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as fp:
        meta = pickle.load(fp)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])

    return encode, decode

def block_inputs(data, block_size, batch_size, next_token_only=False):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])

    if next_token_only:
        y = torch.stack([torch.from_numpy(data[i+1+block_size].astype(np.int64)) for i in ix])
    else:
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    return x, y
