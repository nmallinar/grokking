import os
import sys
from math import ceil
import torch
from tqdm import tqdm
import scipy
import wandb
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from data import get_data
from model import TwoLayerFCN, OneLayerFCN
import torch.nn.functional as F
from torch import nn
from torch.func import jacrev
from rfm import main as rfm_main

torch.manual_seed(34)
import random
random.seed(23)
np.random.seed(234)
torch.set_default_dtype(torch.float32)

def main(args: dict):
    if args.wandb_offline:
        mode = 'offline'
    else:
        mode = 'online'

    wandb.init(entity='belkinlab', project=args.wandb_proj_name, mode=mode, config=args,
               dir=args.out_dir)
    # TODO: add wandb name
    # wandb.run.name = f'{wandb.run.id} - {args.model} act_fn={args.act_fn}, agop_weight={args.agop_weight}, agop_subsample_n={args.agop_subsample_n}, wd={args.weight_decay}, bs={args.batch_size}'
    wandb.run.name = f'{wandb.run.id} - {args.model} act_fn={args.act_fn}, agop_weight={args.agop_weight}, wd={args.weight_decay}, init_scale={args.init_scale}'

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
    if config.model == 'TwoLayerFCN':
        # embedding_layer = nn.Embedding(num_tokens, config.dim_model)
        # embedding_layer.requires_grad_(False)
        # embedding_layer = embedding_layer.to(device)

        model = TwoLayerFCN(
            dim_model=config.dim_model,
            num_tokens=num_tokens,
            hidden_width=config.fcn_hidden_width,
            context_len=context_len,
            init_scale=config.init_scale
        ).to(device)
    elif config.model == 'OneLayerFCN':
        model = OneLayerFCN(
            dim_model=config.dim_model,
            num_tokens=num_tokens,
            hidden_width=config.fcn_hidden_width,
            context_len=context_len,
            init_scale=config.init_scale
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
    log_freq = 20

    if embedding_layer is not None:
        np.save(os.path.join(out_dir, f'embedding_layer.npy'), embedding_layer.state_dict()['weight'].detach().cpu().numpy())

    for epoch in tqdm(range(num_epochs)):
        # if epoch in viz_indices:
        #     visual_weights(model, epoch)

        train(model, train_loader, optimizer, scheduler, device,
              config.num_steps, num_tokens, args.loss, config,
              embedding_layer=embedding_layer,
              agop_weight=config.agop_weight,
              agop_subsample_n=config.agop_subsample_n)
        val_acc = evaluate(model, val_loader, device, epoch, num_tokens, args.loss, config, embedding_layer=embedding_layer)

        final_agops, final_left_agops = calc_full_agops(model, train_loader, config, embedding_layer=embedding_layer)
        if epoch % log_freq == 0:
            visual_weights(model, epoch)

        ols_feats(model, train_loader, val_loader, device, epoch, num_tokens, config, embedding_layer=embedding_layer, return_layer='lin1', feature_projection=final_left_agops[0], proj_key='left_agop')
        ols_feats(model, train_loader, val_loader, device, epoch, num_tokens, config, embedding_layer=embedding_layer, return_layer='lin1', feature_projection=np.real(scipy.linalg.sqrtm(final_left_agops[0])), proj_key='sqrt_left_agop')
        ols_feats(model, train_loader, val_loader, device, epoch, num_tokens, config, embedding_layer=embedding_layer, return_layer='act_fn(lin1)')

        if config.model == 'TwoLayerFCN':
            ols_feats(model, train_loader, val_loader, device, epoch, num_tokens, config, embedding_layer=embedding_layer, return_layer='lin2')
            ols_feats(model, train_loader, val_loader, device, epoch, num_tokens, config, embedding_layer=embedding_layer, return_layer='act_fn(lin2)')

        # ols_feats(model, train_loader, val_loader, device, epoch, num_tokens, config, embedding_layer=embedding_layer, return_layer='M^.5x')
        # ols_feats(model, train_loader, val_loader, device, epoch, num_tokens, config, embedding_layer=embedding_layer, return_layer='act_fn(M^.5x)')


        if config.model == 'OneLayerFCN':
            weights = [model.fc1.weight.detach()]
            idx_range = 1
        elif config.model == 'TwoLayerFCN':
            weights = [model.fc1.weight.detach(), model.fc2.weight.detach()]
            idx_range = 2

        for idx in range(idx_range):
            right_nfm = weights[idx].t() @ weights[idx]
            right_nfm = right_nfm.cpu().numpy()

            if epoch % log_freq == 0:
                plot_agop(final_agops[idx], f'Right AGOP {idx}, Epoch {epoch}', f'right_agop{idx}', commit=False)
            log_corr(right_nfm, final_agops[idx], f'right_agop{idx}_corr_to_right_nfm_w{idx}', commit=False)

            agop = np.real(scipy.linalg.sqrtm(final_agops[idx]))
            if epoch % log_freq == 0:
                plot_agop(agop, f'Sqrt Right AGOP {idx}, Epoch {epoch}', f'sqrt_right_agop{idx}', commit=False)
            log_corr(right_nfm, agop, f'sqrt_right_agop{idx}_corr_to_right_nfm_w{idx}', commit=False)

            left_nfm = weights[idx] @ weights[idx].t()
            left_nfm = left_nfm.cpu().numpy()

            if epoch % log_freq == 0:
                plot_agop(final_left_agops[idx], f'Left AGOP {idx}, Epoch {epoch}', f'left_agop{idx}', commit=False)
            log_corr(left_nfm, final_left_agops[idx], f'left_agop{idx}_corr_to_left_nfm_w{idx}', commit=False)
            
            l, v = np.linalg.eig(left_nfm)
            scaled_nfm = v @ np.diag(l/256.0) @ v.T
            log_corr(scaled_nfm, final_left_agops[idx], f'left_agop{idx}_corr_to_scaled_left_nfm_w{idx}', commit=False)

            left_agop = np.real(scipy.linalg.sqrtm(final_left_agops[idx]))
            if epoch % log_freq == 0:
                plot_agop(left_agop, f'Sqrt Left AGOP {idx}, Epoch {epoch}', f'sqrt_left_agop{idx}', commit=False)
            log_corr(left_nfm, left_agop, f'sqrt_left_agop{idx}_corr_to_left_nfm_w{idx}', commit=False)

            log_corr(scaled_nfm, left_agop, f'sqrt_left_agop{idx}_corr_to_scaled_left_nfm_w{idx}', commit=False)

        if val_acc >= 0.98 and epoch % val_save_freq == 0:
            nfm = model.fc1.weight.t() @ model.fc1.weight
            np.save(os.path.join(out_dir, f'ep_{epoch}_right_nfm.npy'), nfm.detach().cpu().numpy())

            nfm = model.fc1.weight @ model.fc1.weight.t()
            np.save(os.path.join(out_dir, f'ep_{epoch}_left_nfm.npy'), nfm.detach().cpu().numpy())

            # final_data, final_labels = extract_feats(model, train_loader, config, embedding_layer=embedding_layer)
            # np.save(os.path.join(out_dir, f'ep_{epoch}_train_feats.npy'), final_data.numpy())
            # np.save(os.path.join(out_dir, f'ep_{epoch}_train_labels.npy'), final_labels.numpy())
            #
            # final_data, final_labels = extract_feats(model, val_loader, config, embedding_layer=embedding_layer)
            # np.save(os.path.join(out_dir, f'ep_{epoch}_test_feats.npy'), final_data.numpy())
            # np.save(os.path.join(out_dir, f'ep_{epoch}_test_labels.npy'), final_labels.numpy())

def log_corr(nfm, agop, log_key, commit=False):
    corr = np.corrcoef(nfm.flatten(), agop.flatten())
    wandb.log({
        log_key: corr[0][1]
    }, commit=commit)

def plot_agop(agop, caption, log_key, commit=False):
    plt.clf()
    plt.imshow(agop)
    plt.colorbar()
    img = wandb.Image(
        plt,
        caption=caption
    )
    wandb.log({log_key: img}, commit=commit)

def visual_weights(model, epoch_idx):
    w0 = model.fc1.weight
    # w0: [h, d]
    w0w0t = w0 @ w0.T
    w0tw0 = w0.T @ w0

    w0w0t = w0w0t.detach().cpu().numpy()
    w0tw0 = w0tw0.detach().cpu().numpy()

    plt.clf()
    plt.imshow(w0w0t)
    plt.colorbar()
    image = wandb.Image(
        plt,
        caption=f"Epoch {epoch_idx}, W0 @ W0.T"
    )
    wandb.log({"w0_w0.T": image}, commit=False)

    plt.clf()
    plt.imshow(w0tw0)
    plt.colorbar()
    img2 = wandb.Image(
        plt,
        caption=f"Epoch {epoch_idx}, W0.T @ W0"
    )
    wandb.log({"w0.T_w0": img2}, commit=False)

    eigvals, _ = np.linalg.eig(w0w0t)
    eigvals = np.sort(eigvals)[::-1]
    plt.clf()
    plt.plot(range(len(eigvals)), np.log(eigvals + 1e-12))
    plt.title(f'Epoch {epoch_idx}, eigenvalues of W0 @ W0.T')
    plt.xlabel('eigenvalue index')
    plt.ylabel('log(eigenvalue)')
    wandb.log({"spectra w0_w0t": wandb.Image(plt)}, commit=False)

    eigvals, _ = np.linalg.eig(w0tw0)
    eigvals = np.sort(eigvals)[::-1]
    plt.clf()
    plt.plot(range(len(eigvals)), np.log(eigvals + 1e-12))
    plt.title(f'Epoch {epoch_idx}, eigenvalues of W0.T @ W0')
    plt.xlabel('eigenvalue index')
    plt.ylabel('log(eigenvalue)')
    wandb.log({"spectra w0t_w0": wandb.Image(plt)}, commit=False)

def calc_full_agops(model, loader, config, embedding_layer=None):
    num_tokens = config.prime

    final_agops = []
    final_left_agops = []
    total_n = 0
    for idx, batch in enumerate(loader):
        # Copy data to device if needed
        batch = tuple(t.to(config.device) for t in batch)
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

        total_n += nsamps
        dumb1 = torch.zeros((nsamps, model.hidden_width)).to(config.device)
        dumb2 = torch.zeros((nsamps, model.hidden_width)).to(config.device)
        dumb3 = torch.zeros((nsamps, model.num_tokens)).to(config.device)

        dumb4 = torch.zeros((nsamps, model.inp_dim)).to(config.device)
        dumb5 = torch.zeros((nsamps, model.hidden_width)).to(config.device)
        dumb6 = torch.zeros((nsamps, model.hidden_width)).to(config.device)

        _, _, agops, left_agops = calc_batch_agops(model, inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, config.agop_subsample_n, config.device, config)
        for jdx in range(len(agops)):
            if idx == 0:
                final_agops.append(agops[jdx]*nsamps)
                final_left_agops.append(left_agops[jdx]*nsamps)
            else:
                final_agops[jdx] += agops[jdx]*nsamps
                final_left_agops[jdx] += left_agops[jdx]*nsamps

    for idx in range(len(agops)):
        final_agops[idx] /= total_n
        final_left_agops[idx] /= total_n

    return final_agops, final_left_agops

def calc_batch_agops(model, inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, agop_subsample_n, device, config):
    if agop_subsample_n > 0:
        indices = torch.randperm(inputs.size(0), dtype=torch.int32, device=device)[:agop_subsample_n]
        inp_sample = inputs[indices]
    else:
        inp_sample = inputs

    # all of these methods work for computing jacobians, they have different
    # tradeoffs depending on layer and batch sizes, but they can be
    # used interchangeably if one is too slow
    #jacs = torch.func.jacrev(model.forward)(inp_sample)

    # left AGOP is (k, k)
    # right AGOP is (d, d)
    # w_0: (k, d)
    # left_nfm: w_0 @ w_0.T
    # right_nfm: w_0.T @ w_0
    if config.model == 'TwoLayerFCN':
        left_idx = [0, 1]
        right_idx = [2, 3]
        layer_idx = [0, 1, 0, 1]
        jacs = torch.func.jacfwd(model.forward, argnums=(1, 2, 4, 5))(inp_sample, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, None, config.act_fn)
        weights = [model.fc1.weight.detach(), model.fc2.weight.detach()]
    elif config.model == 'OneLayerFCN':
        left_idx = [0]
        right_idx = [1]
        layer_idx = [0, 0]
        jacs = torch.func.jacfwd(model.forward, argnums=(1, 3))(inp_sample, dumb1, dumb3, dumb4, dumb6, None, config.act_fn)
        weights = [model.fc1.weight.detach()]
    else:
        raise Exception()
    jacs = list(jacs)

    agop_tr = 0.0
    left_agop_tr = 0.0
    agops = []
    left_agops = []

    for idx in range(len(jacs)):
        jacs[idx] = torch.sum(jacs[idx], dim=(1, 2)).reshape(len(inp_sample), -1)
        agop = jacs[idx].t() @ jacs[idx] / len(inp_sample)
        if idx in right_idx:
            agop_tr += torch.trace(agop)
        else:
            left_agop_tr += torch.trace(agop)
        agop = agop.detach().cpu().numpy()

        if idx in left_idx:
            left_agops.append(agop)
            left_nfm = weights[layer_idx[idx]] @ weights[layer_idx[idx]].t()
            left_nfm = left_nfm.cpu().numpy()

            corr = np.corrcoef(left_nfm.flatten(), agop.flatten())
            wandb.log({
                f'batch_left_agop{layer_idx[idx]}_corr_to_left_nfm_w{layer_idx[idx]}': corr[0][1]
            }, commit=False)

            agop = np.real(scipy.linalg.sqrtm(agop))
            corr = np.corrcoef(left_nfm.flatten(), agop.flatten())
            wandb.log({
                f'batch_sqrt_left_agop{layer_idx[idx]}_corr_to_left_nfm_w{layer_idx[idx]}': corr[0][1]
            }, commit=False)
        else:
            agops.append(agop)
            right_nfm = weights[layer_idx[idx]].t() @ weights[layer_idx[idx]]
            right_nfm = right_nfm.cpu().numpy()

            corr = np.corrcoef(right_nfm.flatten(), agop.flatten())
            wandb.log({
                f'batch_right_agop{layer_idx[idx]}_corr_to_right_nfm_w{layer_idx[idx]}': corr[0][1]
            }, commit=False)

            agop = np.real(scipy.linalg.sqrtm(agop))
            corr = np.corrcoef(right_nfm.flatten(), agop.flatten())
            wandb.log({
                f'batch_sqrt_right_agop{layer_idx[idx]}_corr_to_right_nfm_w{layer_idx[idx]}': corr[0][1]
            }, commit=False)

    return agop_tr, left_agop_tr, agops, left_agops

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
        agop_tr, left_agop_tr, _, _ = calc_batch_agops(model, inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, agop_subsample_n, device, config)

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
            "training/agop_tr": agop_tr,
            "training/left_agop_tr": left_agop_tr,
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

def ols_feats(model, train_loader, val_loader, device, epoch, num_classes, config, embedding_layer=None, return_layer=None, feature_projection=None, proj_key=''):
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

        if feature_projection is not None:
            all_train_data = all_train_data @ feature_projection

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

        if feature_projection is not None:
            all_train_data = all_val_data @ feature_projection

        sol = np.linalg.pinv(all_train_data.T @ all_train_data) @ all_train_data.T @ F.one_hot(all_train_labels, num_classes).numpy()
        pred_scores = all_val_data @ sol
        pred_labels = np.argmax(pred_scores, axis=1)

        mse = np.mean(np.square(pred_scores - F.one_hot(all_val_labels, num_classes).numpy()))
        count = np.sum(all_val_labels.numpy() == pred_labels)

        if feature_projection is not None:
            log_key = f'validation/ols_{return_layer}_proj_{proj_key}'
        else:
            log_key = f'validation/ols_{return_layer}'

        metrics = {
            f"{log_key}_accuracy": count / len(all_val_labels),
            f"{log_key}_loss": mse,
            "epoch": epoch
        }
        wandb.log(metrics, commit=False)

        return count / len(all_val_labels)

def extract_feats(model, loader, config, embedding_layer=None):
    with torch.no_grad():
        num_tokens = config.prime

        final_data = []
        final_labels = []
        for idx, batch in enumerate(loader):
            # Copy data to device if needed
            batch = tuple(t.to(config.device) for t in batch)
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

            hid_states = model(inputs, return_layer='act_fn(lin1)', act=config.act_fn)

            final_data.append(hid_states.detach().cpu())
            final_labels.append(labels.detach().cpu())

        final_data = torch.cat(final_data, dim=0)
        final_labels = torch.cat(final_labels, dim=0)

        return final_data, final_labels
