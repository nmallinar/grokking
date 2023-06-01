import os
import sys
from math import ceil
import torch
from tqdm import tqdm
import wandb

from data import get_data
from model import Transformer, FCN
from rfm import main as rfm_main
import matplotlib.pyplot as plt
import torch.nn.functional as F

def main(args: dict):
    if args.wandb_offline:
        mode = 'offline'
    else:
        mode = 'online'
    wandb.init(entity='jonathanxue', project="x+y", mode=mode, config=args)
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
        rfm_main(config.prime + 2, config.dim_model,
                 train_loader, val_loader,
                 wandb, config.kernel_bandwidth)
        sys.exit(0)

    print("======= MODEL DEFINITION =======")
    print(model)

    # Optimizer
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
        train(model, train_loader, optimizer, scheduler, device, config.num_steps, config.prime + 2, args.loss)
        evaluate(model, val_loader, device, epoch, config.prime + 2, args.loss)
        # TODO: list epochs
        if epoch == 5 or epoch == 320 or epoch == 500:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    }, 'saves/epoch{}.pt'.format(epoch))

def train(model, train_loader, optimizer, scheduler, device, num_steps, num_classes, loss_arg):
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

        # Zero gradient buffers
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs)

        acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)

        if loss_arg == 'mse':
            labels = F.one_hot(labels, num_classes).float()
        loss = criterion(output, labels)

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

def evaluate(model, val_loader, device, epoch, num_classes, loss_arg):
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

        # Forward pass
        with torch.no_grad():
            output = model(inputs)
            correct += (torch.argmax(output, dim=1) == labels).sum()

            if loss_arg == 'mse':
                labels = F.one_hot(labels, num_classes).float()

            loss += criterion(output, labels) * len(labels)

    acc = correct / len(val_loader.dataset)
    loss = loss / len(val_loader.dataset)

    metrics = {
        "validation/accuracy": acc,
        "validation/loss": loss,
        "epoch": epoch
    }
    wandb.log(metrics, commit=False)
