import os
import sys
from math import ceil
import torch
from tqdm import tqdm
import wandb
import numpy as np
import scipy
import torchvision
import matplotlib.pyplot as plt

from data import get_data
from model import Transformer, FCN
import torch.nn.functional as F
from torch import nn
from torch.func import jacrev
from rfm import main as rfm_main

torch.set_default_dtype(torch.float32)

def main(args: dict):
    if args.wandb_offline:
        mode = 'offline'
    else:
        mode = 'online'

    wandb.init(entity='belkinlab', project=args.wandb_proj_name, mode=mode, config=args)
    # TODO: add wandb name
    wandb.run.name = f'{wandb.run.id} - agop_weight={args.agop_weight}, agop_subsample_n={args.agop_subsample_n}, wd={args.weight_decay}, bs={args.batch_size}, n_layers={args.num_layers}'
    wandb.run.save()

    out_dir = os.path.join(args.out_dir, args.wandb_proj_name, wandb.run.id)
    os.makedirs(out_dir, exist_ok=True)

    config = wandb.config
    device = torch.device(config.device)

    # Define time scales
    wandb.define_metric("step")
    wandb.define_metric("epoch")

    # Define metrics
    wandb.define_metric("training/accuracy", step_metric='step')
    wandb.define_metric("training/loss", step_metric='step')
    wandb.define_metric("validation/accuracy", step_metric='epoch')
    wandb.define_metric("validation/loss", step_metric='epoch')

    train_loader, val_loader, context_len, train_dataset, val_dataset = \
        get_data(
            config.operation,
            config.prime,
            config.training_fraction,
            config.batch_size
        )

    embedding_layer = None
    if config.model == 'transformer':
        model = Transformer(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            num_heads=config.num_heads,
            num_tokens=config.prime + 2,
            seq_len=5
            ).to(device)
    elif config.model == 'fcn':
        embedding_layer = nn.Embedding(config.prime + 2, config.dim_model)
        embedding_layer.requires_grad_(False)
        embedding_layer = embedding_layer.to(device)

        model = FCN(
            dim_model=config.dim_model,
            num_tokens=config.prime + 2,
            num_layers=config.num_layers,
            hidden_width=config.fcn_hidden_width,
            context_len=context_len
        ).to(device)
    elif config.model == 'rfm':
        embedding_layer = nn.Embedding(config.prime + 2, config.dim_model)
        embedding_layer.requires_grad_(False)
        rfm_main(config.prime + 2, config.dim_model,
                 train_loader, val_loader,
                 wandb, config.kernel_bandwidth, embedding_layer,
                 config.agop_weight)
        sys.exit(0)

    print("======= MODEL DEFINITION =======")
    print(model)

    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum
        )
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=config.weight_decay
            )
    else:
        raise Exception('optimizer not implemented!')
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer, start_factor = 0.1, total_iters=9
    # )

    num_epochs = ceil(config.num_steps / len(train_loader))

    # viz_indices = [0, 1, 2, 3, 4, 5, 10, 50, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 24000]
    viz_indices = [0, 1, 5, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 24000]
    for epoch in tqdm(range(num_epochs)):
        if epoch in viz_indices:
            visual_weights(model, epoch)

        train(model, train_loader, optimizer, scheduler, device,
              config.num_steps, config.prime + 2, args.loss,
              embedding_layer=embedding_layer,
              agop_weight=config.agop_weight,
              agop_subsample_n=config.agop_subsample_n)
        val_acc = evaluate(model, val_loader, device, epoch, config.prime + 2, args.loss, embedding_layer=embedding_layer)

        if val_acc == 1.0:
            final_agops = []
            total_n = 0
            for idx, batch in enumerate(train_loader):
                # Copy data to device if needed
                batch = tuple(t.to(device) for t in batch)
                # Unpack the batch from the loader
                inputs, labels = batch
                total_n += inputs.size(0)

                if embedding_layer is not None:
                    inputs = embedding_layer(inputs)
                    inputs = inputs.view(inputs.size(0), -1)

                _, agops = calc_agops(model, inputs, config.agop_subsample_n, device, normalize=False)
                if idx == 0:
                    for agop in agops:
                        final_agops.append(agop)
                else:
                    for idx in range(len(agops)):
                        final_agops[idx] += agops[idx]
            for idx in range(len(agops)):
                final_agops[idx] /= (total_n**2)
                np.save(os.path.join(out_dir, f'agop_{idx}.npy'), final_agops[idx])
            np.save(os.path.join(out_dir, f'embedding_layer.npy'), embedding_layer.detach().cpu().numpy())

def visual_weights(model, epoch_idx):
    params = dict(model.named_parameters())
    # [d, h] weights
    w0 = params['layers.0.weight']
    w0w0t = w0 @ w0.T
    w0w0t = w0w0t.unsqueeze(0).unsqueeze(0)
    w0w0t = torchvision.utils.make_grid(w0w0t)

    image = wandb.Image(
        w0w0t,
        caption=f"Epoch {epoch_idx}, W0 @ W0.T"
    )
    wandb.log({"w0_w0.T": image})

    w0w0t = w0 @ w0.T
    w0w0t = w0w0t.detach().cpu().numpy()
    eigvals, _ = np.linalg.eig(w0w0t)
    plt.clf()
    plt.plot(range(len(eigvals)), np.log(eigvals + 1e-12))
    plt.title(f'Epoch {epoch_idx}, eigenvalues of W0 @ W0.T')
    plt.xlabel('eigenvalue index')
    plt.ylabel('log(eigenvalue)')
    wandb.log({"spectra": wandb.Image(plt)})

def calc_agops(model, inputs, agop_subsample_n, device, normalize=True):
    if agop_subsample_n > 0:
        indices = torch.randperm(inputs.size(0), dtype=torch.int32, device=device)[:agop_subsample_n]
        inp_sample = inputs[indices]
    else:
        inp_sample = inputs

    # all of these methods work for computing jacobians, they have different
    # tradeoffs depending on layer and batch sizes, but they can be
    # used interchangeably if one is too slow
    #jacs = torch.func.jacrev(model.forward)(inp_sample)
    jacs = torch.func.jacfwd(model.forward)(inp_sample)
    #jacs = torch.autograd.functional.jacobian(model, inp_sample, create_graph=True)
    #jacs = list(jacs)
    agop_tr = 0.0
    agops = []
    for idx in range(len(jacs)-1):
        jac = torch.sum(jacs[idx], dim=(1,2))
        if normalize:
            jac = jac / inp_sample.size(0)

        jac = jac.t() @ jac
        agops.append(jac.detach().cpu().numpy())
        # jac = jac / torch.max(jac)
        agop_tr += torch.trace(jac)

    return agop_tr, agops

def train(model, train_loader, optimizer, scheduler,
          device, num_steps, num_classes, loss_arg,
          embedding_layer=None, agop_weight=0.0,
          agop_subsample_n=-1):
    # Set model to training mode
    model.train()

    if loss_arg == 'mse':
        criterion = torch.nn.MSELoss()
    elif loss_arg == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()

    # Loop over each batch from the training set
    for batch in train_loader:

        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        if embedding_layer is not None:
            #inputs = F.one_hot(inputs.long(), num_classes=num_classes)
            inputs = embedding_layer(inputs)
            inputs = inputs.view(inputs.size(0), -1)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Forward pass
        # output = model(inputs)
        #output, hid = model(inputs, return_hid=True)
        hid = model(inputs)
        output = hid[-1]
        for idx in range(len(hid)):
            hid[idx].requires_grad_(True)

        acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)

        if loss_arg == 'mse':
            labels = F.one_hot(labels, num_classes).float()
        loss = criterion(output, labels)

        # Backward pass
        #loss.backward(retain_graph=True)
        mse_loss = loss.clone()

        agop_tr, _ = calc_agops(model, inputs, agop_subsample_n, device)
        if agop_weight > 0:
            loss += agop_weight * agop_tr

        loss.backward()

        # Update weights
        optimizer.step()
        # scheduler.step()

        metrics = {
            "training/accuracy": acc,
            "training/loss": loss,
            "training/mse_loss": mse_loss,
            "training/agop_tr": agop_tr.detach().cpu().numpy(),
            "step": wandb.run.step
        }
        wandb.log(metrics)

        # Finish training at maximum gradient updates
        if wandb.run.step == num_steps:
            return

def evaluate(model, val_loader, device, epoch, num_classes, loss_arg, embedding_layer=None):
    # Set model to evaluation mode
    model.eval()

    if loss_arg == 'mse':
        criterion = torch.nn.MSELoss()
    elif loss_arg == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    loss = 0.

    # Loop over each batch from the validation set
    for batch in val_loader:

        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        if embedding_layer is not None:
            inputs = embedding_layer(inputs)
            inputs = inputs.view(inputs.size(0), -1)

        # Forward pass
        with torch.no_grad():
            output = model(inputs)
            output = output[-1]
            correct += (torch.argmax(output, dim=1) == labels).sum()

            if loss_arg == 'mse':
                labels = F.one_hot(labels, num_classes).double()

            loss += criterion(output, labels) * len(labels)

    acc = correct / len(val_loader.dataset)
    loss = loss / len(val_loader.dataset)

    metrics = {
        "validation/accuracy": acc,
        "validation/loss": loss,
        "epoch": epoch
    }
    wandb.log(metrics, commit=False)

    return acc
