import os
import sys
import argparse
import wandb
import torch
import torch.nn.functional as F
import numpy as np
import scipy
import random
from tqdm import tqdm

from data import operation_mod_p_data, make_data_splits, make_dataloader
from models import neural_nets
import utils

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
# torch.manual_seed(3143)
# random.seed(253)
# np.random.seed(1145)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', default='belkinlab')
    parser.add_argument('--wandb_proj_name', default='mar22-rfm-grokking')
    parser.add_argument('--wandb_offline', default=False, action='store_true')
    parser.add_argument('--group_key', default='', type=str)
    parser.add_argument('--out_dir', default='./')
    parser.add_argument('--operation', '-op', default="x+y")
    parser.add_argument('--prime', '-p', default=97, type=int)
    parser.add_argument('--training_fraction', default=0.5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--agop_batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--device', default='cuda', choices={'cuda', 'cpu'})

    parser.add_argument('--hidden_width', default=256, type=int)
    parser.add_argument('--init_scale', default=1.0, type=float)
    parser.add_argument("--act_fn", type=str, default="relu")

    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--momentum', default=0.0, type=float)
    args = parser.parse_args()

    mode = 'online'
    if args.wandb_offline:
        mode = 'offline'

    wandb.init(entity=args.wandb_entity, project=args.wandb_proj_name, mode=mode, config=args,
               dir=args.out_dir)

    out_dir = os.path.join(args.out_dir, args.wandb_proj_name, wandb.run.id)
    os.makedirs(out_dir, exist_ok=True)

    wandb.run.name = f'{wandb.run.id} - p: {args.prime}, train_frac: {args.training_fraction}'

    all_inputs, all_labels = operation_mod_p_data(args.operation, args.prime)
    X_tr, y_tr, X_te, y_te = make_data_splits(all_inputs, all_labels, args.training_fraction)

    X_tr = F.one_hot(X_tr, args.prime).view(-1, 2*args.prime).double()
    y_tr_onehot = F.one_hot(y_tr, args.prime).double()
    X_te = F.one_hot(X_te, args.prime).view(-1, 2*args.prime).double()
    y_te_onehot = F.one_hot(y_te, args.prime).double()

    train_loader = make_dataloader(X_tr, y_tr_onehot, args.batch_size, shuffle=True, drop_last=False)
    test_loader = make_dataloader(X_te, y_te_onehot, args.batch_size, shuffle=False, drop_last=False)

    model = neural_nets.OneLayerFCN(
        num_tokens=args.prime,
        hidden_width=args.hidden_width,
        context_len=2,
        init_scale=args.init_scale,
        n_classes=args.prime
    ).to(device)


    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=args.learning_rate,
    #     weight_decay=args.weight_decay,
    #     momentum=args.momentum
    # )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay
    )
    criterion = torch.nn.MSELoss()

    global_step = 0
    for epoch in tqdm(range(args.epochs)):

        model.train()
        for idx, batch in enumerate(train_loader):
            batch = tuple(t.to(config.device) for t in batch)
            inputs, labels = batch

            optimizer.zero_grad()
            output = model(inputs, act=args.act_fn)

            count = (output.argmax(-1) == labels.argmax(-1)).sum()
            acc = count / output.shape[0]

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            wandb.log({
                'training/accuracy': acc,
                'training/loss': loss,
                'epoch': epoch
            }, step=global_step)

            global_step += 1

        model.eval()
        with torch.no_grad():
            count = 0
            total_loss = 0
            total = 0
            for idx, batch in enumerate(test_loader):
                batch = tuple(t.to(config.device) for t in batch)
                inputs, labels = batch

                optimizer.zero_grad()
                output = model(inputs, act=args.act_fn)

                count += (output.argmax(-1) == labels.argmax(-1)).sum()
                total += output.shape[0]
                loss = criterion(output, labels)
                total_loss += loss * output.shape[0]

            total_loss /= total
            acc = count / total

            wandb.log({
                'validation/accuracy': acc,
                'validation/loss': total_loss,
                'epoch': epoch
            }, step=global_step)