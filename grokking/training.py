import os
import sys
from math import ceil
import torch
from tqdm import tqdm
import wandb

from data import get_data
from model import Transformer, FCN
from rfm import main as rfm_main
import torch.nn.functional as F

def main(args: dict):
    if args.wandb_offline:
        mode = 'offline'
    else:
        mode = 'online'
    wandb.init(entity='jonathanxue', project="fcn x divide y", mode=mode, config=args)
    # TODO: add wandb name
    wandb.run.name = f'lr={args.learning_rate}'
    wandb.run.save()

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

    train_loader, val_loader, context_len = get_data(
        config.operation,
        config.prime,
        config.training_fraction,
        config.batch_size
        )

    if config.model == 'transformer':
        model = Transformer(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            num_heads=config.num_heads,
            num_tokens=config.prime + 2,
            seq_len=5
            ).to(device)
    elif config.model == 'fcn':
        model = FCN(
            dim_model=config.dim_model,
            num_tokens=config.prime + 2,
            num_layers=config.num_layers,
            hidden_width=config.fcn_hidden_width,
            context_len=context_len
        )
    elif config.model == 'rfm':
        rfm_main(config.prime + 2, config.dim_model, train_loader, val_loader)
        sys.exit(0)

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

    for epoch in tqdm(range(num_epochs)):
        train(model, train_loader, optimizer, scheduler, device, config.num_steps)
        evaluate(model, val_loader, device, epoch)

def train(model, train_loader, optimizer, scheduler, device, num_steps):
    # Set model to training mode
    model.train()
    # Change torch.nn.CrossEntropyLoss() to torch.nn.MSELoss() without 
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.CrossEntropyLoss()
    # if (args.loss == 'mse'):
    #     criterion = torch.nn.MSELoss()

    # Loop over each batch from the training set
    for batch in train_loader:

        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Zero gradient buffers
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs)

        labels_one_hot = F.one_hot(labels, 99).float()
        loss = criterion(output, labels_one_hot)
        acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)

        # Backward pass
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

def evaluate(model, val_loader, device, epoch):
    # Set model to evaluation mode
    model.eval()
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.CrossEntropyLoss()
    # if (args.loss == 'mse'):
    #     criterion = torch.nn.MSELoss()
    

    correct = 0
    loss = 0.

    # Loop over each batch from the validation set
    for batch in val_loader:

        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Forward pass
        with torch.no_grad():
            output = model(inputs)
            correct += (torch.argmax(output, dim=1) == labels).sum()
            loss += criterion(output, labels) * len(labels)

    acc = correct / len(val_loader.dataset)
    loss = loss / len(val_loader.dataset)

    metrics = {
        "validation/accuracy": acc,
        "validation/loss": loss,
        "epoch": epoch
    }
    wandb.log(metrics, commit=False)
