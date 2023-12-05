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
from model import Transformer, TwoLayerFCN, FCN
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

    wandb.init(entity='belkinlab', project=args.wandb_proj_name, mode=mode, config=args,
               dir=args.out_dir)
    # TODO: add wandb name
    wandb.run.name = f'{wandb.run.id} - act_fn={args.act_fn}, agop_weight={args.agop_weight}, agop_subsample_n={args.agop_subsample_n}, wd={args.weight_decay}, bs={args.batch_size}, n_layers={args.num_layers}'
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
    num_tokens = config.prime + 2
    num_tokens = config.prime

    embedding_layer = None
    if config.model == 'transformer':
        model = Transformer(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            num_heads=config.num_heads,
            num_tokens=num_tokens,
            seq_len=5
            ).to(device)
    elif config.model == 'fcn':
        embedding_layer = nn.Embedding(num_tokens, config.dim_model)
        embedding_layer.requires_grad_(False)
        embedding_layer = embedding_layer.to(device)

        model = FCN(
            dim_model=config.dim_model,
            num_tokens=num_tokens,
            num_layers=config.num_layers,
            hidden_width=config.fcn_hidden_width,
            context_len=context_len
        ).to(device)
    elif config.model == 'TwoLayerFCN':
        # embedding_layer = nn.Embedding(num_tokens, config.dim_model)
        # embedding_layer.requires_grad_(False)
        # embedding_layer = embedding_layer.to(device)

        model = TwoLayerFCN(
            dim_model=config.dim_model,
            num_tokens=num_tokens,
            hidden_width=config.fcn_hidden_width,
            context_len=context_len
        ).to(device)
    elif config.model == 'rfm':
        embedding_layer = nn.Embedding(num_tokens, config.dim_model)
        emb_state = np.load('grokking_outputs/nov27_proper_agop/embedding_layer.npy')
        embedding_layer.load_state_dict({
            'weight': torch.Tensor(emb_state)
        })
        embedding_layer.requires_grad_(False)
        rfm_main(num_tokens, config.dim_model,
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

    viz_indices = [0, 1, 5, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 24000]
    val_save_freq = 500

    if embedding_layer is not None:
        np.save(os.path.join(out_dir, f'embedding_layer.npy'), embedding_layer.state_dict()['weight'].detach().cpu().numpy())

    for epoch in tqdm(range(num_epochs)):
        if epoch in viz_indices:
            visual_weights(model, epoch)

        train(model, train_loader, optimizer, scheduler, device,
              config.num_steps, num_tokens, args.loss, config,
              embedding_layer=embedding_layer,
              agop_weight=config.agop_weight,
              agop_subsample_n=config.agop_subsample_n)
        val_acc = evaluate(model, val_loader, device, epoch, num_tokens, args.loss, config, embedding_layer=embedding_layer)
        ols_feats(model, train_loader, val_loader, device, epoch, num_tokens, config, embedding_layer=embedding_layer, return_layer='lin1')
        ols_feats(model, train_loader, val_loader, device, epoch, num_tokens, config, embedding_layer=embedding_layer, return_layer='relu1')
        ols_feats(model, train_loader, val_loader, device, epoch, num_tokens, config, embedding_layer=embedding_layer, return_layer='lin2')
        ols_feats(model, train_loader, val_loader, device, epoch, num_tokens, config, embedding_layer=embedding_layer, return_layer='relu2')
        ols_feats(model, train_loader, val_loader, device, epoch, num_tokens, config, embedding_layer=embedding_layer, return_layer='M^.5x')
        ols_feats(model, train_loader, val_loader, device, epoch, num_tokens, config, embedding_layer=embedding_layer, return_layer='relu(M^.5x)')


        if val_acc >= 0.98 and epoch % val_save_freq == 0:
            final_agops = []
            total_n = 0
            final_data = []
            final_labels = []
            for idx, batch in enumerate(train_loader):
                # Copy data to device if needed
                batch = tuple(t.to(device) for t in batch)
                # Unpack the batch from the loader
                inputs, labels = batch

                if embedding_layer is not None:
                    inputs = embedding_layer(inputs)
                    inputs = inputs.view(inputs.size(0), -1)
                else:
                    inputs = F.one_hot(inputs, num_tokens).float()
                    inputs = inputs.view(inputs.size(0), -1)

                if config.agop_subsample_n <= 0:
                    nsamps = len(inputs)
                else:
                    nsamps = config.agop_subsample_n

                with torch.no_grad():
                    hid_states = model(inputs, return_layer='act_fn(lin1)', act=config.act_fn)

                final_data.append(hid_states.detach().cpu())
                final_labels.append(labels.detach().cpu())

                total_n += nsamps
                dumb1 = torch.zeros((nsamps, model.hidden_width)).to(device)
                dumb2 = torch.zeros((nsamps, model.hidden_width)).to(device)
                dumb3 = torch.zeros((nsamps, model.num_tokens)).to(device)

                dumb4 = torch.zeros((nsamps, model.inp_dim)).to(device)
                dumb5 = torch.zeros((nsamps, model.hidden_width)).to(device)
                dumb6 = torch.zeros((nsamps, model.hidden_width)).to(device)

                _, agops = calc_agops(model, inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, config.agop_subsample_n, device, config)
                for jdx in range(len(agops)):
                    if idx == 0:
                        final_agops.append(agops[jdx]*nsamps)
                    else:
                        final_agops[jdx] += agops[jdx]*nsamps
                    # np.save(os.path.join(out_dir, f'ep_{epoch}_batch_{idx}_agop_{jdx}.npy'), agops[jdx])

            for jdx, agop in enumerate(final_agops):
                np.save(os.path.join(out_dir, f'ep_{epoch}_agop_{jdx}.npy'), agop / total_n)
            nfm = model.fc1.weight.t() @ model.fc1.weight
            np.save(os.path.join(out_dir, f'ep_{epoch}_neural_feature_matrix.npy'), nfm.detach().cpu().numpy())
            final_data = torch.cat(final_data, dim=0)
            final_labels = torch.cat(final_labels, dim=0)
            np.save(os.path.join(out_dir, f'ep_{epoch}_train_feats.npy'), final_data.numpy())
            np.save(os.path.join(out_dir, f'ep_{epoch}_train_labels.npy'), final_labels.numpy())

            final_data = []
            final_labels = []
            for batch in val_loader:
                # Copy data to device if needed
                batch = tuple(t.to(device) for t in batch)

                # Unpack the batch from the loader
                inputs, labels = batch

                if embedding_layer is not None:
                    inputs = embedding_layer(inputs)
                    inputs = inputs.view(inputs.size(0), -1)
                else:
                    inputs = F.one_hot(inputs, num_tokens).float()
                    inputs = inputs.view(inputs.size(0), -1)

                # Forward pass
                with torch.no_grad():
                    hid_states = model(inputs, return_layer='act_fn(lin1)', act=config.act_fn)

                final_data.append(hid_states.detach().cpu())
                final_labels.append(labels.detach().cpu())

            final_data = torch.cat(final_data, dim=0)
            final_labels = torch.cat(final_labels, dim=0)
            np.save(os.path.join(out_dir, f'ep_{epoch}_test_feats.npy'), final_data.numpy())
            np.save(os.path.join(out_dir, f'ep_{epoch}_test_labels.npy'), final_labels.numpy())


def visual_weights(model, epoch_idx):
    #params = dict(model.named_parameters())
    # [d, h] weights

    #w0 = params['layers.0.weight']
    w0 = model.fc1.weight.t()
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
    eigvals = np.sort(eigvals)[::-1]
    plt.clf()
    plt.plot(range(len(eigvals)), np.log(eigvals + 1e-12))
    plt.title(f'Epoch {epoch_idx}, eigenvalues of W0 @ W0.T')
    plt.xlabel('eigenvalue index')
    plt.ylabel('log(eigenvalue)')
    wandb.log({"spectra": wandb.Image(plt)})

def calc_agops(model, inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, agop_subsample_n, device, config):
    if agop_subsample_n > 0:
        indices = torch.randperm(inputs.size(0), dtype=torch.int32, device=device)[:agop_subsample_n]
        inp_sample = inputs[indices]
    else:
        inp_sample = inputs

    # all of these methods work for computing jacobians, they have different
    # tradeoffs depending on layer and batch sizes, but they can be
    # used interchangeably if one is too slow
    #jacs = torch.func.jacrev(model.forward)(inp_sample)
    jacs = torch.func.jacfwd(model.forward, argnums=(1, 2, 3, 4, 5, 6))(inp_sample, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, None, config.act_fn)
    jacs = list(jacs)

    weights = [model.fc1.weight.detach(), model.fc2.weight.detach()]

    agop_tr = 0.0
    agops = []
    for idx in range(3):
        jacs[idx] = torch.sum(jacs[idx], dim=(1, 2)).reshape(len(inp_sample), -1)
        agop = jacs[idx].t() @ jacs[idx] / len(inp_sample)
        agop_tr += torch.trace(agop)
        agop = agop.detach().cpu().numpy()
        agops.append(agop)

        if idx != 2:
            right_nfm = weights[idx] @ weights[idx].t()
            right_nfm = right_nfm.cpu().numpy()
            corr = np.corrcoef(right_nfm.flatten(), agop.flatten())
            wandb.log({
                f'right_agop{idx}_corr_to_right_nfm_w{idx}': corr[0][1]
            }, commit=False)

            left_agop = torch.sum(jacs[idx + 3], dim=(1, 2)).reshape(len(inp_sample), -1)
            left_agop = left_agop.t() @ left_agop / len(inp_sample)
            left_agop = left_agop.detach().cpu().numpy()
            left_nfm = weights[idx].t() @ weights[idx]
            left_nfm = left_nfm.cpu().numpy()
            corr = np.corrcoef(left_nfm.flatten(), left_agop.flatten())
            wandb.log({
                f'left_agop{idx}_corr_to_left_nfm_w{idx}': corr[0][1]
            }, commit=False)

    return agop_tr, agops

def train(model, train_loader, optimizer, scheduler,
          device, num_steps, num_classes, loss_arg, config,
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
        else:
            inputs = F.one_hot(inputs, num_classes).float()
            inputs = inputs.view(inputs.size(0), -1)

        if agop_subsample_n <= 0:
            nsamps = len(inputs)
        else:
            nsamps = agop_subsample_n

        dumb1 = torch.zeros((nsamps, model.hidden_width)).to(device)
        dumb2 = torch.zeros((nsamps, model.hidden_width)).to(device)
        dumb3 = torch.zeros((nsamps, model.num_tokens)).to(device)

        dumb4 = torch.zeros((nsamps, model.inp_dim)).to(device)
        dumb5 = torch.zeros((nsamps, model.hidden_width)).to(device)
        dumb6 = torch.zeros((nsamps, model.hidden_width)).to(device)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs, act=config.act_fn)
        # output.requires_grad_(True)
        agop_tr, _ = calc_agops(model, inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, agop_subsample_n, device, config)

        acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)

        if loss_arg == 'mse':
            labels = F.one_hot(labels, num_classes).float()
        loss = criterion(output, labels)

        # Backward pass
        mse_loss = loss.clone()

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

def evaluate(model, val_loader, device, epoch, num_classes, loss_arg, config, embedding_layer=None):
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
        else:
            inputs = F.one_hot(inputs, num_classes).float()
            inputs = inputs.view(inputs.size(0), -1)

        # Forward pass
        with torch.no_grad():
            output = model(inputs, act=config.act_fn)
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

def ols_feats(model, train_loader, val_loader, device, epoch, num_classes, config, embedding_layer=None, return_layer=None):
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():

        all_train_data = []
        all_train_labels = []
        for batch in train_loader:
            # Copy data to device if needed
            batch = tuple(t.to(device) for t in batch)

            # Unpack the batch from the loader
            inputs, labels = batch

            if embedding_layer is not None:
                inputs = embedding_layer(inputs)
                inputs = inputs.view(inputs.size(0), -1)
            else:
                inputs = F.one_hot(inputs, num_classes).float()
                inputs = inputs.view(inputs.size(0), -1)

            # Forward pass
            with torch.no_grad():
                output = model(inputs, return_layer=return_layer, act=config.act_fn)
                all_train_data.append(output.detach().cpu())
                all_train_labels.append(labels.detach().cpu())

        all_train_data = torch.cat(all_train_data).numpy()
        all_train_labels = torch.cat(all_train_labels)

        all_val_data = []
        all_val_labels = []
        # Loop over each batch from the validation set
        for batch in val_loader:
            # Copy data to device if needed
            batch = tuple(t.to(device) for t in batch)

            # Unpack the batch from the loader
            inputs, labels = batch

            if embedding_layer is not None:
                inputs = embedding_layer(inputs)
                inputs = inputs.view(inputs.size(0), -1)
            else:
                inputs = F.one_hot(inputs, num_classes).float()
                inputs = inputs.view(inputs.size(0), -1)

            # Forward pass
            with torch.no_grad():
                output = model(inputs, return_layer=return_layer, act=config.act_fn)
                all_val_data.append(output.detach().cpu())
                all_val_labels.append(labels.detach().cpu())

        all_val_labels = torch.cat(all_val_labels)
        all_val_data = torch.cat(all_val_data).numpy()

        sol = np.linalg.pinv(all_train_data.T @ all_train_data) @ all_train_data.T @ F.one_hot(all_train_labels, num_classes).numpy()
        pred_scores = all_val_data @ sol
        pred_labels = np.argmax(pred_scores, axis=1)

        mse = np.mean(np.square(pred_scores - F.one_hot(all_val_labels, num_classes).numpy()))
        count = np.sum(all_val_labels.numpy() == pred_labels)

        metrics = {
            f"validation/ols_{return_layer}_accuracy": count / len(all_val_labels),
            f"validation/ols_{return_layer}_loss": mse,
            "epoch": epoch
        }
        wandb.log(metrics, commit=False)

        return count / len(all_val_labels)
