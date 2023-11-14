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
# from functorch import make_functional, vmap, vjp, jvp, jacrev


torch.set_default_dtype(torch.float32)

def main(args: dict):
    if args.eval_entk > 0 and args.model != 'fcn':
        raise Exception('empirical NTK evaluation only supported for FCN')

    if args.wandb_offline:
        mode = 'offline'
    else:
        mode = 'online'
    wandb.init(entity='belkinlab', project="nov8-grokking", mode=mode, config=args)
    # TODO: add wandb name
    # wandb.run.name = f'lr={args.learning_rate}'
    # wandb.run.save()

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

    print("======= MODEL DEFINITION =======")
    print(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=config.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor = 0.1, total_iters=9
    )

    num_epochs = ceil(config.num_steps / len(train_loader))

    # viz_indices = [0, 1, 2, 3, 4, 5, 10, 50, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 24000]
    viz_indices = [0, 1, 5, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 24000]
    for epoch in tqdm(range(num_epochs)):
        if epoch in viz_indices:
            visual_weights(model, epoch)

        train(model, train_loader, optimizer, scheduler, device, config.num_steps, config.prime + 2, args.loss, embedding_layer=embedding_layer, agop_weight=config.agop_weight)
        evaluate(model, val_loader, device, epoch, config.prime + 2, args.loss, embedding_layer=embedding_layer)

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
    plt.plot(range(len(eigvals)), np.log(eigvals))
    plt.title(f'Epoch {epoch_idx}, eigenvalues of W0 @ W0.T')
    plt.xlabel('eigenvalue index')
    plt.ylabel('log(eigenvalue)')
    wandb.log({"spectra": wandb.Image(plt)})

def train(model, train_loader, optimizer, scheduler, device, num_steps, num_classes, loss_arg, embedding_layer=None, agop_weight=0.0):
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
        
        if agop_weight > 0:
            jacs = torch.autograd.functional.jacobian(model, inputs, create_graph=True)
            jacs = list(jacs)
            for idx in range(len(jacs)):
                jacs[idx] = torch.sum(jacs[idx], dim=(1,2))
                loss += agop_weight * torch.trace(jacs[idx].t() @ jacs[idx])
            #del jacs
            #torch.cuda.empty_cache()

        loss.backward()

        # Update weights
        optimizer.step()
        scheduler.step()

        metrics = {
            "training/accuracy": acc,
            "training/loss": loss,
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
