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
from model import Transformer, FCN, FCNEmbedded
from rfm import main as rfm_main
from rfm_entk import main as rfm_entk_main
import torch.nn.functional as F
from empirical_ntk import get_eNTK_batched

torch.set_default_dtype(torch.float64)

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
        ).to(device)
    elif config.model == 'rfm':
        rfm_main(config.prime + 2, config.dim_model,
                 train_loader, val_loader,
                 wandb, config.kernel_bandwidth)
        sys.exit(0)
    elif config.model == 'rfm_fcn':
        model = FCNEmbedded(
            dim_model=config.dim_model,
            num_tokens=config.prime + 2,
            num_layers=config.num_layers,
            hidden_width=config.fcn_hidden_width,
            context_len=context_len
        ).to(device)
        rfm_entk_main(model, config.prime + 2, config.dim_model,
                      train_loader, val_loader, wandb, config.device)
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

    # viz_indices = [0, 1, 2, 3, 4, 5, 10, 50, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 24000]
    #viz_indices = [0, 1, 5, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 24000]
    viz_indices = [0, 1, 5, 25, 50, 100] + list(range(250, 25000, 250))
    for epoch in tqdm(range(num_epochs)):
        if epoch in viz_indices:
            visual_weights(model, epoch)

        if config.eval_entk > 0 and (
            (epoch <= 50 and epoch % 5 == 0) or
            (epoch > 50 and epoch % config.eval_entk == 0)
        ):
            viz = False
            if epoch in viz_indices:
                viz = True
            eval_entk(model, train_dataset, val_dataset, device, epoch, config.prime + 2, config.batch_size, viz=viz)
            if device == 'cuda':
                torch.cuda.empty_cache()

        train(model, train_loader, optimizer, scheduler, device, config.num_steps, config.prime + 2, args.loss)
        evaluate(model, val_loader, device, epoch, config.prime + 2, args.loss)

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
            labels = F.one_hot(labels, num_classes).double()
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

def eval_entk(model, train_dataset, val_dataset, device, epoch, num_classes, batch_size, viz=False):
    model.eval()
    train_data = train_dataset.dataset[train_dataset.indices]
    val_data = val_dataset.dataset[val_dataset.indices]

    n_train = train_data[1].shape[0]
    n_ctrain = n_train * num_classes
    n_val = val_data[1].shape[0]

    # [n_train*num_classes, n_train*num_classes]
    train_ntk = get_eNTK_batched(model, train_data, num_classes, device, batch_size)
    train_ntk = train_ntk.numpy()

    # [n_train*num_classes, n_test*num_classes]
    train_test_ntk = get_eNTK_batched(model, train_data, num_classes, device, batch_size, val_dataset=val_data)
    train_test_ntk = train_test_ntk.numpy()

    y_tr = F.one_hot(train_data[1], num_classes=num_classes).reshape(n_ctrain)
    alpha = scipy.linalg.solve(train_ntk + 1e-8*np.eye(n_ctrain), y_tr, assume_a='pos')

    # training loss / accuracy first
    preds = torch.from_numpy(train_ntk.T @ alpha)
    mse = torch.mean((preds - y_tr)**2)
    count = torch.argmax(preds.reshape(n_train, num_classes), dim=1)
    acc = sum(count == train_data[1]) / float(n_train)
    # print(f'Train MSE: {mse}, acc: {acc}')
    del y_tr

    metrics = {
        "training/entk_accuracy": acc,
        "training/entk_mse": mse,
        "epoch": epoch
    }
    wandb.log(metrics, commit=False)

    y_te = F.one_hot(val_data[1], num_classes=num_classes)
    preds = torch.from_numpy(train_test_ntk.T @ alpha).reshape(n_val, num_classes)
    mse = torch.mean((preds - y_te)**2)
    count = torch.argmax(preds, dim=1)
    acc = sum(count == val_data[1]) / float(n_val)
    del y_te

    metrics = {
        "validation/entk_accuracy": acc,
        "validation/entk_mse": mse,
        "epoch": epoch
    }
    wandb.log(metrics, commit=False)

    train_ntk = train_ntk.unsqueeze(0).unsqueeze(0)
    train_ntk = torchvision.utils.make_grid(train_ntk)

    image = wandb.Image(
        train_ntk,
        caption=f"Epoch {epoch_idx}, train entk"
    )
    wandb.log({"train entk": image})

    train_test_ntk = train_test_ntk.unsqueeze(0).unsqueeze(0)
    train_test_ntk = torchvision.utils.make_grid(train_test_ntk)

    image = wandb.Image(
        train_test_ntk,
        caption=f"Epoch {epoch_idx}, train-test entk"
    )
    wandb.log({"train-test entk": image})

    del train_ntk, train_test_ntk

    # print(f'Val MSE: {mse}, acc: {acc}')

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
