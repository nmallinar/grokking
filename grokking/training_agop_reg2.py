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

from data import get_data_with_agop_loader, get_augmented_data_with_agop_loader
from model import TwoLayerFCN, OneLayerFCN, FourLayerFCN
import torch.nn.functional as F
from torch import nn
from torch.func import jacrev
from rfm import main as rfm_main
from inf_ntk import ntk_fn, jax_ntk_fn, get_jax_ntk_fn

# torch.manual_seed(34)
# import random
# random.seed(23)
# np.random.seed(234)
# torch.set_default_dtype(torch.float32)

kernel_fn = get_jax_ntk_fn(depth=2, bias=0)

def main(args: dict):
    if args.wandb_offline:
        mode = 'offline'
    else:
        mode = 'online'

    wandb.init(entity='belkinlab', project=args.wandb_proj_name, mode=mode, config=args,
               dir=args.out_dir)
    # TODO: add wandb name
    # wandb.run.name = f'{wandb.run.id} - {args.model} act_fn={args.act_fn}, agop_weight={args.agop_weight}, agop_subsample_n={args.agop_subsample_n}, wd={args.weight_decay}, bs={args.batch_size}'
    wandb.run.name = f'{wandb.run.id} - {args.model} n_toks={args.num_tokens}, p={args.prime}, act_fn={args.act_fn}, agop_weight={args.agop_weight}, wd={args.weight_decay}, init_scale={args.init_scale}'

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

    num_tokens = config.num_tokens

    train_loader, agop_loader, val_loader, val_loader1, context_len, train_dataset, val_dataset, base_train_feats, base_train_labels, base_val_feats, base_val_labels = \
        get_augmented_data_with_agop_loader(
            config.operation,
            config.prime,
            num_tokens,
            config.training_fraction,
            config.batch_size,
            config.agop_subsample_n
        )

    base_train_feats = F.one_hot(base_train_feats, num_tokens).view(base_train_feats.size(0), -1).numpy()
    base_val_feats = F.one_hot(base_val_feats, num_tokens).view(base_val_feats.size(0), -1).numpy()
    base_train_labels = F.one_hot(base_train_labels, config.prime).numpy()
    base_val_labels = F.one_hot(base_val_labels, config.prime).numpy()
    print(base_train_feats.shape)
    print(base_val_feats.shape)

    np.save(os.path.join(out_dir, f'base_train_data.npy'), base_train_feats)
    np.save(os.path.join(out_dir, f'base_train_labels.npy'), base_train_labels)
    np.save(os.path.join(out_dir, f'base_val_data.npy'), base_val_feats)
    np.save(os.path.join(out_dir, f'base_val_labels.npy'), base_val_labels)

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
            init_scale=config.init_scale,
            n_classes=config.prime
        ).to(device)
    elif config.model == 'FourLayerFCN':
        model = FourLayerFCN(
            dim_model=config.dim_model,
            num_tokens=num_tokens,
            hidden_width=config.fcn_hidden_width,
            context_len=context_len,
            init_scale=config.init_scale,
            n_classes=config.prime
        ).to(device)
    elif config.model == 'OneLayerFCN':
        model = OneLayerFCN(
            dim_model=config.dim_model,
            num_tokens=num_tokens,
            hidden_width=config.fcn_hidden_width,
            context_len=context_len,
            init_scale=config.init_scale,
            n_classes=config.prime
        ).to(device)
    elif config.model == 'rfm':
        embedding_layer = nn.Embedding(num_tokens, config.dim_model)
        # emb_state = np.load('grokking_outputs/nov27_proper_agop/embedding_layer.npy')
        # embedding_layer.load_state_dict({
        #     'weight': torch.Tensor(emb_state)
        # })
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

    num_epochs = config.num_steps
    # num_epochs = ceil(config.num_steps / len(train_loader))

    log_freq = 100

    if embedding_layer is not None:
        np.save(os.path.join(out_dir, f'embedding_layer.npy'), embedding_layer.state_dict()['weight'].detach().cpu().numpy())

    for epoch in tqdm(range(num_epochs)):
        train(model, train_loader, agop_loader, optimizer, scheduler, device,
              config.num_steps, num_tokens, args.loss, config,
              embedding_layer=embedding_layer,
              agop_weight=config.agop_weight)

        with torch.no_grad():
            val_acc, val_loss = evaluate(model, val_loader, device, epoch, num_tokens, args.loss, config, embedding_layer=embedding_layer, log_key='total_')
            train_acc, train_loss = evaluate(model, train_loader, device, epoch, num_tokens, args.loss, config, embedding_layer=embedding_layer, log_key='train_')
            val_acc1, val_loss1 = evaluate(model, val_loader1, device, epoch, num_tokens, args.loss, config, embedding_layer=embedding_layer, log_key='(n<p)_')

            #print(f'Epoch {epoch}:\t Train Acc: {train_acc}\t Total Val Acc: {val_acc}\t Val Acc (n <= p): {val_acc1}')

            if not args.skip_agop_comps:
                if epoch % log_freq == 0:
                    visual_weights(model, epoch)

                final_agops, final_left_agops = calc_full_agops(model, agop_loader, config, num_tokens, embedding_layer=embedding_layer)
                final_sqrt_agops = []
                final_sqrt_left_agops = []
                for idx in range(len(final_agops)):
                    final_agops[idx] = final_agops[idx].cpu().numpy()
                    final_left_agops[idx] = final_left_agops[idx].cpu().numpy()
                    final_sqrt_agops.append(np.real(scipy.linalg.sqrtm(final_agops[idx])))
                    final_sqrt_left_agops.append(np.real(scipy.linalg.sqrtm(final_left_agops[idx])))

                log_agop_norms(final_agops, final_sqrt_agops, final_left_agops, final_sqrt_left_agops, commit=False)

                if config.model == 'OneLayerFCN':
                    weights = [model.fc1.weight.detach()]
                    idx_range = 1
                elif config.model == 'TwoLayerFCN' or config.model == 'FourLayerFCN':
                    weights = [model.fc1.weight.detach(), model.fc2.weight.detach()]
                    idx_range = 2

                for idx in range(idx_range):
                    right_nfm = weights[idx].t() @ weights[idx]
                    right_nfm = right_nfm.cpu().numpy()

                    if epoch % log_freq == 0:
                        plot_agop(final_agops[idx], f'Right AGOP {idx}, Epoch {epoch}', f'right_agop{idx}', commit=False)
                    log_corr(right_nfm, final_agops[idx], f'right_agop{idx}_corr_to_right_nfm_w{idx}', commit=False)

                    if epoch % log_freq == 0:
                        plot_agop(final_sqrt_agops[idx], f'Sqrt Right AGOP {idx}, Epoch {epoch}', f'sqrt_right_agop{idx}', commit=False)
                    log_corr(right_nfm, final_sqrt_agops[idx], f'sqrt_right_agop{idx}_corr_to_right_nfm_w{idx}', commit=False)

                    left_nfm = weights[idx] @ weights[idx].t()
                    left_nfm = left_nfm.cpu().numpy()

                    if epoch % log_freq == 0:
                        plot_agop(final_left_agops[idx], f'Left AGOP {idx}, Epoch {epoch}', f'left_agop{idx}', commit=False)
                    log_corr(left_nfm, final_left_agops[idx], f'left_agop{idx}_corr_to_left_nfm_w{idx}', commit=False)

                    if epoch % log_freq == 0:
                        plot_agop(final_sqrt_left_agops[idx], f'Sqrt Left AGOP {idx}, Epoch {epoch}', f'sqrt_left_agop{idx}', commit=False)
                    log_corr(left_nfm, final_sqrt_left_agops[idx], f'sqrt_left_agop{idx}_corr_to_left_nfm_w{idx}', commit=False)


                if epoch % log_freq == 0:
                    ep_out_dir = os.path.join(out_dir, f'epoch_{epoch}')
                    os.makedirs(ep_out_dir, exist_ok=True)

                    nfm = model.fc1.weight.t() @ model.fc1.weight
                    np.save(os.path.join(ep_out_dir, f'right_nfm.npy'), nfm.detach().cpu().numpy())

                    nfm = model.fc1.weight @ model.fc1.weight.t()
                    np.save(os.path.join(ep_out_dir, f'left_nfm.npy'), nfm.detach().cpu().numpy())

                    for idx in range(len(final_agops)):
                        np.save(os.path.join(ep_out_dir, f'right_agop_{idx}.npy'), final_agops[idx])
                        np.save(os.path.join(ep_out_dir, f'sqrt_right_agop_{idx}.npy'), final_sqrt_agops[idx])
                        np.save(os.path.join(ep_out_dir, f'left_agop_{idx}.npy'), final_left_agops[idx])
                        np.save(os.path.join(ep_out_dir, f'sqrt_left_agop_{idx}.npy'), final_sqrt_left_agops[idx])


            if epoch % log_freq == 0:
                syn_data_dir = os.path.join(ep_out_dir, 'synthetic_data')
                os.makedirs(syn_data_dir, exist_ok=True)

                # flat covariance
                cov = torch.tensor([1.0 for eig_i in range(num_tokens*2)])
                data, labels = get_synthetic_data(model, config, num_tokens, embedding_layer=embedding_layer, n_points=1000000, cov=torch.diag(cov))

                np.save(os.path.join(syn_data_dir, f'flat_synthetic_data.npy'), data.numpy())
                np.save(os.path.join(syn_data_dir, f'flat_synthetic_labels.npy'), labels.numpy())

                # spiked covariance
                cov[-2:] = 1e-8
                data, labels = get_synthetic_data(model, config, num_tokens, embedding_layer=embedding_layer, n_points=1000000, cov=torch.diag(cov))
                np.save(os.path.join(syn_data_dir, f'minus2_synthetic_data.npy'), data.numpy())
                np.save(os.path.join(syn_data_dir, f'minus2_synthetic_labels.npy'), labels.numpy())

                cov[-4:] = 1e-8
                data, labels = get_synthetic_data(model, config, num_tokens, embedding_layer=embedding_layer, n_points=1000000, cov=torch.diag(cov))
                np.save(os.path.join(syn_data_dir, f'minus4_synthetic_data.npy'), data.numpy())
                np.save(os.path.join(syn_data_dir, f'minus4_synthetic_labels.npy'), labels.numpy())

                cov[-6:] = 1e-8
                data, labels = get_synthetic_data(model, config, num_tokens, embedding_layer=embedding_layer, n_points=1000000, cov=torch.diag(cov))
                np.save(os.path.join(syn_data_dir, f'minus6_synthetic_data.npy'), data.numpy())
                np.save(os.path.join(syn_data_dir, f'minus6_synthetic_labels.npy'), labels.numpy())

                if val_acc == 1.0:
                    # can save less frequently once we generalize
                    log_freq = 1000

def log_agop_norms(final_agops, final_sqrt_agops, final_left_agops, final_sqrt_left_agops, commit=False):
    for idx, agop in enumerate(final_agops):
        fro_norm = np.linalg.norm(agop, ord='fro')
        two_norm = np.linalg.norm(agop, ord=2)
        stable_rank = (fro_norm ** 2) / (two_norm ** 2)
        wandb.log({
            f'norms/right_agop{idx}_fro_norm': fro_norm,
            f'norms/right_agop{idx}_two_norm': two_norm,
            f'norms/right_agop{idx}_stable_rank': stable_rank
        }, commit=commit)

        fro_norm = np.linalg.norm(final_sqrt_agops[idx], ord='fro')
        two_norm = np.linalg.norm(final_sqrt_agops[idx], ord=2)
        stable_rank = (fro_norm ** 2) / (two_norm ** 2)
        wandb.log({
            f'norms/sqrt_right_agop{idx}_fro_norm': fro_norm,
            f'norms/sqrt_right_agop{idx}_two_norm': two_norm,
            f'norms/sqrt_right_agop{idx}_stable_rank': stable_rank
        }, commit=commit)

    for idx, agop in enumerate(final_left_agops):
        fro_norm = np.linalg.norm(agop, ord='fro')
        two_norm = np.linalg.norm(agop, ord=2)
        stable_rank = (fro_norm ** 2) / (two_norm ** 2)
        wandb.log({
            f'norms/left_agop{idx}_fro_norm': fro_norm,
            f'norms/left_agop{idx}_two_norm': two_norm,
            f'norms/left_agop{idx}_stable_rank': stable_rank
        }, commit=commit)

        fro_norm = np.linalg.norm(final_sqrt_left_agops[idx], ord='fro')
        two_norm = np.linalg.norm(final_sqrt_left_agops[idx], ord=2)
        stable_rank = (fro_norm ** 2) / (two_norm ** 2)
        wandb.log({
            f'norms/sqrt_left_agop{idx}_fro_norm': fro_norm,
            f'norms/sqrt_left_agop{idx}_two_norm': two_norm,
            f'norms/sqrt_left_agop{idx}_stable_rank': stable_rank
        }, commit=commit)

def log_corr(nfm, agop, log_key, commit=False):
    corr = np.corrcoef(nfm.flatten(), agop.flatten())
    wandb.log({
        f'corrs/{log_key}': corr[0][1]
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

    eigvals, _ = np.linalg.eig(agop)
    eigvals = np.sort(eigvals)[::-1]
    plt.clf()
    plt.plot(range(len(eigvals)), np.log(eigvals + 1e-10))
    plt.xlabel('eigenvalue index')
    plt.ylabel('log(eigenvalue)')
    wandb.log({f'spectra {log_key}': wandb.Image(plt, caption=caption)}, commit=False)

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

def calc_full_agops(model, loader, config, num_tokens, embedding_layer=None):

    dumb1 = torch.zeros((config.agop_subsample_n, model.hidden_width)).to(config.device)
    dumb2 = torch.zeros((config.agop_subsample_n, model.hidden_width)).to(config.device)
    dumb3 = torch.zeros((config.agop_subsample_n, config.prime)).to(config.device)

    dumb4 = torch.zeros((config.agop_subsample_n, model.inp_dim)).to(config.device)
    dumb5 = torch.zeros((config.agop_subsample_n, model.hidden_width)).to(config.device)
    dumb6 = torch.zeros((config.agop_subsample_n, model.hidden_width)).to(config.device)

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

        nsamps = inputs.size(0)
        total_n += nsamps

        agops, left_agops = calc_batch_agops(model, inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, config.device, config)
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

def calc_batch_agops(model, inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, device, config):
    # all of these methods work for computing jacobians, they have different
    # tradeoffs depending on layer and batch sizes, but they can be
    # used interchangeably if one is too slow
    #jacs = torch.func.jacrev(model.forward)(inputs)

    # left AGOP is (k, k)
    # right AGOP is (d, d)
    # w_0: (k, d)
    # left_nfm: w_0 @ w_0.T
    # right_nfm: w_0.T @ w_0
    if config.model == 'TwoLayerFCN' or config.model == 'FourLayerFCN':
        left_idx = [0, 1]
        right_idx = [2, 3]
        layer_idx = [0, 1, 0, 1]
        jacs = torch.func.jacfwd(model.forward, argnums=(1, 2, 4, 5))(inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, None, config.act_fn)
        weights = [model.fc1.weight.detach(), model.fc2.weight.detach()]
    elif config.model == 'OneLayerFCN':
        left_idx = [0]
        right_idx = [1]
        layer_idx = [0, 0]
        jacs = torch.func.jacfwd(model.forward, argnums=(1, 3))(inputs, dumb1, dumb3, dumb4, dumb6, None, config.act_fn)

        #left_idx = [0, 1]
        #right_idx = [2, 3]
        #layer_idx = [0, 0]
        #jacs = torch.func.jacfwd(model.forward, argnums=(1, 2, 3, 4))(inputs, dumb1, dumb3, dumb4, dumb6, None, config.act_fn)
        weights = [model.fc1.weight.detach()]
    else:
        raise Exception()
    jacs = list(jacs)

    agops = []
    left_agops = []

    for idx in range(len(jacs)):
        jacs[idx] = torch.sum(jacs[idx], dim=(1, 2)).reshape(len(inputs), -1)

        agop = jacs[idx].t() @ jacs[idx] / len(inputs)

        if idx in left_idx:
            left_agops.append(agop)
        else:
            agops.append(agop)

    return agops, left_agops

def train(model, train_loader, agop_loader, optimizer, scheduler,
          device, num_steps, num_tokens, loss_arg, config,
          embedding_layer=None, agop_weight=0.0):

    n_classes = config.prime
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
            #inputs = F.one_hot(inputs.long(), num_tokens=num_tokens)
            inputs = embedding_layer(inputs)
            inputs = inputs.view(inputs.size(0), -1)
        else:
            inputs = F.one_hot(inputs, num_tokens).float()
            inputs = inputs.view(inputs.size(0), -1)


        if not config.skip_agop_comps:
            dumb1 = torch.zeros((config.agop_subsample_n, model.hidden_width)).to(config.device)
            dumb2 = torch.zeros((config.agop_subsample_n, model.hidden_width)).to(config.device)
            dumb3 = torch.zeros((config.agop_subsample_n, n_classes)).to(config.device)

            dumb4 = torch.zeros((config.agop_subsample_n, model.inp_dim)).to(config.device)
            dumb5 = torch.zeros((config.agop_subsample_n, model.hidden_width)).to(config.device)
            dumb6 = torch.zeros((config.agop_subsample_n, model.hidden_width)).to(config.device)
            final_agops, final_left_agops = calc_batch_agops(model, inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, config.device, config)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs, act=config.act_fn)
        # output.requires_grad_(True)

        acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)

        if loss_arg == 'mse':
            labels = F.one_hot(labels, n_classes).float()
        loss = criterion(output, labels)

        # Backward pass
        mse_loss = loss.clone()

        if not config.skip_agop_comps:
            agop_tr = 0.0
            left_agop_tr = 0.0
            for idx in range(len(final_agops)):
                agop_tr += torch.trace(final_agops[idx])
                left_agop_tr += torch.trace(final_left_agops[idx])

            if agop_weight > 0:
                loss += agop_weight * agop_tr
        else:
            agop_tr = 0
            left_agop_tr = 0

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

def evaluate(model, val_loader, device, epoch, num_tokens, loss_arg, config, embedding_layer=None, log_key=''):
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
            inputs = F.one_hot(inputs, num_tokens).float()
            inputs = inputs.view(inputs.size(0), -1)

        # Forward pass
        with torch.no_grad():
            output = model(inputs, act=config.act_fn)
            correct += (torch.argmax(output, dim=1) == labels).sum()

            n_classes = config.prime
            if loss_arg == 'mse':
                labels = F.one_hot(labels, n_classes).double()

            loss += criterion(output, labels) * len(labels)

    acc = correct / len(val_loader.dataset)
    loss = loss / len(val_loader.dataset)

    metrics = {
        f"validation/{log_key}accuracy": acc,
        f"validation/{log_key}loss": loss,
        "epoch": epoch
    }
    wandb.log(metrics, commit=False)

    return acc, loss

def get_synthetic_data(model, config, num_tokens, embedding_layer=None, n_points=10000, cov=None):
    with torch.no_grad():
        # input1 = torch.distributions.MultivariateNormal(torch.zeros(num_tokens), torch.eye(2)).sample([n_points])
        # input2 = torch.distributions.MultivariateNormal(torch.zeros(num_tokens), torch.eye(2)).sample([n_points])
        # input1 = torch.rand(n_points, num_tokens)
        # input2 = torch.rand(n_points, num_tokens)
        # input1 /= torch.sum(input1, -1, keepdims=True)
        # input2 /= torch.sum(input2, -1, keepdims=True)
        # inputs = torch.cat((input1, input2), dim=-1)

        if cov is None:
            cov = torch.eye(num_tokens*2)

        inputs = torch.distributions.MultivariateNormal(torch.zeros(num_tokens*2), cov).sample([n_points])
        inputs[:num_tokens] /= torch.sum(inputs[:num_tokens], -1, keepdims=True)
        inputs[num_tokens:] /= torch.sum(inputs[num_tokens:], -1, keepdims=True)

        outputs = []
        for idx in range(0, n_points, config.batch_size):
            batch_input = inputs[idx:idx+config.batch_size,:].to(config.device)

            output = model(batch_input, act=config.act_fn)
            outputs.append(output.cpu())

        outputs = torch.cat(outputs, dim=0)

        return inputs, outputs
