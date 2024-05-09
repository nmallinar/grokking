import time
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
    wandb.run.name = f'{wandb.run.id} - {args.model} n_toks={args.num_tokens}, p={args.prime}, act_fn={args.act_fn}, agop_weight={args.agop_weight}, wd={args.weight_decay}, init_scale={args.init_scale}, width={args.fcn_hidden_width}, lr={args.learning_rate}, frac={args.training_fraction}'

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

    log_freq = 1000

    if embedding_layer is not None:
        np.save(os.path.join(out_dir, f'embedding_layer.npy'), embedding_layer.state_dict()['weight'].detach().cpu().numpy())

    for epoch in tqdm(range(num_epochs)):
        final_agops, final_left_agops, final_agips, final_left_agips = train(model, train_loader, agop_loader, optimizer, scheduler, device,
              config.num_steps, num_tokens, args.loss, config,
              embedding_layer=embedding_layer,
              agop_weight=config.agop_weight)

        with torch.no_grad():
            val_acc, val_loss = evaluate(model, val_loader, device, epoch, num_tokens, args.loss, config, embedding_layer=embedding_layer, log_key='total_')
            train_acc, train_loss = evaluate(model, train_loader, device, epoch, num_tokens, args.loss, config, embedding_layer=embedding_layer, log_key='train_', compute_margins=True)

            if num_tokens > args.prime:
                val_acc1, val_loss1 = evaluate(model, val_loader1, device, epoch, num_tokens, args.loss, config, embedding_layer=embedding_layer, log_key='(n<p)_')

            #print(f'Epoch {epoch}:\t Train Acc: {train_acc}\t Total Val Acc: {val_acc}\t Val Acc (n <= p): {val_acc1}')

            if not args.skip_agop_comps and epoch % log_freq == 0:
                visual_weights(model, epoch)

                if final_agops is None:
                    with torch.no_grad():
                        final_agops, final_left_agops, final_agips, final_left_agips = \
                            calc_full_agops(model, agop_loader, config, num_tokens, embedding_layer=embedding_layer)
                else:
                    for idx in range(len(final_agops)):
                        final_agops[idx] = final_agops[idx].detach()
                        final_left_agops[idx] = final_left_agops[idx].detach()
                        final_agips[idx] = final_agips[idx].detach()
                        final_left_agips[idx] = final_left_agips[idx].detach()

                final_sqrt_agops = []
                final_sqrt_left_agops = []
                final_sqrt_agips = []
                final_sqrt_left_agips = []
                for idx in range(len(final_agops)):
                    final_agops[idx] = final_agops[idx].cpu().numpy()
                    final_left_agops[idx] = final_left_agops[idx].cpu().numpy()
                    final_sqrt_agops.append(np.real(scipy.linalg.sqrtm(final_agops[idx])))
                    final_sqrt_left_agops.append(np.real(scipy.linalg.sqrtm(final_left_agops[idx])))

                    final_agips[idx] = final_agips[idx].cpu().numpy()
                    final_left_agips[idx] = final_left_agips[idx].cpu().numpy()
                    final_sqrt_agips.append(np.real(scipy.linalg.sqrtm(final_agips[idx])))
                    final_sqrt_left_agips.append(np.real(scipy.linalg.sqrtm(final_left_agips[idx])))

                log_agop_norms(final_agops, final_sqrt_agops, final_left_agops, final_sqrt_left_agops, commit=False)
                log_agop_norms(final_agips, final_sqrt_agips, final_left_agips, final_sqrt_left_agips, commit=False, is_agip=True)

                if config.model == 'OneLayerFCN':
                    weights = [model.fc1.weight.detach()]
                    idx_range = 1
                elif config.model == 'TwoLayerFCN' or config.model == 'FourLayerFCN':
                    weights = [model.fc1.weight.detach(), model.fc2.weight.detach()]
                    idx_range = 2

                out_w = model.out.weight.detach()
                nc_nfm = out_w @ out_w.t()
                nc_nfm = nc_nfm.cpu().numpy()
                log_corr(nc_nfm, final_agips[0], f'right_agip{idx}_corr_to_nc_nfm_w{idx}', commit=False)
                log_corr(nc_nfm, final_sqrt_agips[0], f'sqrt_right_agip{idx}_corr_to_nc_nfm_w{idx}', commit=False)
                log_corr(nc_nfm, final_left_agips[0], f'left_agip{idx}_corr_to_nc_nfm_w{idx}', commit=False)
                log_corr(nc_nfm, final_sqrt_left_agips[0], f'sqrt_left_agip{idx}_corr_to_nc_nfm_w{idx}', commit=False)

                plot_agop(final_agips[idx], f'Right AGIP {idx}, Epoch {epoch}', f'right_agip{idx}', commit=False)
                plot_agop(final_sqrt_agips[idx], f'Sqrt Right AGIP {idx}, Epoch {epoch}', f'sqrt_right_agip{idx}', commit=False)
                plot_agop(final_left_agips[idx], f'Left AGIP {idx}, Epoch {epoch}', f'left_agip{idx}', commit=False)
                plot_agop(final_sqrt_left_agips[idx], f'Sqrt Left AGIP {idx}, Epoch {epoch}', f'sqrt_left_agip{idx}', commit=False)

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

                if val_acc == 1.0:
                    # can save less frequently once we generalize
                    log_freq = 1000

def log_agop_norms(final_agops, final_sqrt_agops, final_left_agops, final_sqrt_left_agops, commit=False,
                   is_agip=False):
    key = 'agop'
    if is_agip:
        key = 'agip'

    for idx, agop in enumerate(final_agops):
        fro_norm = np.linalg.norm(agop, ord='fro')
        two_norm = np.linalg.norm(agop, ord=2)
        stable_rank = (fro_norm ** 2) / (two_norm ** 2)
        wandb.log({
            f'norms/right_{key}{idx}_fro_norm': fro_norm,
            f'norms/right_{key}{idx}_two_norm': two_norm,
            f'norms/right_{key}{idx}_stable_rank': stable_rank
        }, commit=commit)

        fro_norm = np.linalg.norm(final_sqrt_agops[idx], ord='fro')
        two_norm = np.linalg.norm(final_sqrt_agops[idx], ord=2)
        stable_rank = (fro_norm ** 2) / (two_norm ** 2)
        wandb.log({
            f'norms/sqrt_right_{key}{idx}_fro_norm': fro_norm,
            f'norms/sqrt_right_{key}{idx}_two_norm': two_norm,
            f'norms/sqrt_right_{key}{idx}_stable_rank': stable_rank
        }, commit=commit)

    for idx, agop in enumerate(final_left_agops):
        fro_norm = np.linalg.norm(agop, ord='fro')
        two_norm = np.linalg.norm(agop, ord=2)
        stable_rank = (fro_norm ** 2) / (two_norm ** 2)
        wandb.log({
            f'norms/left_{key}{idx}_fro_norm': fro_norm,
            f'norms/left_{key}{idx}_two_norm': two_norm,
            f'norms/left_{key}{idx}_stable_rank': stable_rank
        }, commit=commit)

        fro_norm = np.linalg.norm(final_sqrt_left_agops[idx], ord='fro')
        two_norm = np.linalg.norm(final_sqrt_left_agops[idx], ord=2)
        stable_rank = (fro_norm ** 2) / (two_norm ** 2)
        wandb.log({
            f'norms/sqrt_left_{key}{idx}_fro_norm': fro_norm,
            f'norms/sqrt_left_{key}{idx}_two_norm': two_norm,
            f'norms/sqrt_left_{key}{idx}_stable_rank': stable_rank
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
    final_agips = []
    final_left_agips = []
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

        # startt = time.time()
        # left_agop_test = 0.0
        # hid1 = inputs @ model.fc1.weight.T
        # hid1 = (hid1 > 0).float()
        # agop = 0.0
        # test1 = model.out.weight.T @ model.out.weight
        # test2 = model.fc1.weight @ model.fc1.weight.T
        # for jdx in range(hid1.shape[0]):
        #     dhid1 = torch.diag(hid1[jdx])
        #     left_agop_test += model.fc1.weight.T @ dhid1 @ model.out.weight.T @ model.out.weight @ dhid1 @ model.fc1.weight
        # comptime = time.time() - startt

        # startt = time.time()
        agops, left_agops, agips, left_agips = calc_batch_agops(model, inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, config.device, config)
        # comptime2 = time.time() - startt

        # print(torch.linalg.norm((agops[0] * nsamps) - left_agop_test))
        # print(f'custom: {comptime}, jac: {comptime2}')

        for jdx in range(len(agops)):
            if idx == 0:
                final_agops.append(agops[jdx]*nsamps)
                final_left_agops.append(left_agops[jdx]*nsamps)
                final_agips.append(agips[jdx]*nsamps)
                final_left_agips.append(left_agips[jdx]*nsamps)
            else:
                final_agops[jdx] += agops[jdx]*nsamps
                final_left_agops[jdx] += left_agops[jdx]*nsamps
                final_agips[jdx] += agips[jdx]*nsamps
                final_left_agips[jdx] += left_agips[jdx]*nsamps

    for idx in range(len(agops)):
        final_agops[idx] /= total_n
        final_left_agops[idx] /= total_n
        final_agips[idx] /= total_n
        final_left_agips[idx] /= total_n

    return final_agops, final_left_agops, final_agips, final_left_agips

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
    agips = []
    left_agips = []

    for idx in range(len(jacs)):
        cjac = torch.sum(jacs[idx], dim=(2, 3)).reshape(len(inputs), jacs[idx].shape[1])
        jacs[idx] = torch.sum(jacs[idx], dim=(1, 2)).reshape(len(inputs), -1)

        agop = jacs[idx].t() @ jacs[idx] / len(inputs)
        agip = cjac.t() @ cjac / len(inputs)

        if idx in left_idx:
            left_agops.append(agop)
            left_agips.append(agip)
        else:
            agops.append(agop)
            agips.append(agip)

    return agops, left_agops, agips, left_agips

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
            final_agops, final_left_agops, final_agips, final_left_agips = calc_full_agops(model, agop_loader, config, num_tokens, embedding_layer=None)
            # dumb1 = torch.zeros((config.agop_subsample_n, model.hidden_width)).to(config.device)
            # dumb2 = torch.zeros((config.agop_subsample_n, model.hidden_width)).to(config.device)
            # dumb3 = torch.zeros((config.agop_subsample_n, n_classes)).to(config.device)
            #
            # dumb4 = torch.zeros((config.agop_subsample_n, model.inp_dim)).to(config.device)
            # dumb5 = torch.zeros((config.agop_subsample_n, model.hidden_width)).to(config.device)
            # dumb6 = torch.zeros((config.agop_subsample_n, model.hidden_width)).to(config.device)
            # final_agops, final_left_agops = calc_batch_agops(model, inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, config.device, config)

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

        weight_norm_fc1 = torch.linalg.norm(model.fc1.weight.data).detach()
        weight_norm_out = torch.linalg.norm(model.out.weight.data).detach()

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
            "training/weight_norm_fc1": weight_norm_fc1,
            "training/weight_norm_out": weight_norm_out,
            "step": wandb.run.step
        }
        wandb.log(metrics)

        # Finish training at maximum gradient updates
        # if wandb.run.step * len(train_loader) == num_steps:
        #     return

    if not config.skip_agop_comps:
        return final_agops, final_left_agops, final_agips, final_left_agips
    else:
        return None, None, None, None

def evaluate(model, val_loader, device, epoch, num_tokens, loss_arg, config, embedding_layer=None, log_key='', compute_margins=False):
    # Set model to evaluation mode
    model.eval()

    if loss_arg == 'mse':
        criterion = torch.nn.MSELoss()
    elif loss_arg == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    loss = 0.

    min_margin = 10000
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

            if compute_margins:
                for idx in range(len(labels)):
                    margin = output[idx][labels[idx]]
                    if margin < min_margin:
                        min_margin = margin
                    # max_other_class = [output[idx][x] for x in range(len(labels)) if x != labels[idx]]
                    # margin = output[idx][labels[idx]] - max(max_other_class)


            n_classes = config.prime
            if loss_arg == 'mse':
                labels = F.one_hot(labels, n_classes).double()

            loss += criterion(output, labels) * len(labels)

    min_margin /= torch.sqrt(torch.pow(torch.linalg.norm(model.fc1.weight.data), 2) + torch.pow(torch.linalg.norm(model.out.weight.data), 2))
    acc = correct / len(val_loader.dataset)
    loss = loss / len(val_loader.dataset)

    metrics = {
        f"validation/{log_key}accuracy": acc,
        f"validation/{log_key}loss": loss,
        "epoch": epoch
    }
    if compute_margins:
        metrics[f'validation/{log_key}margin'] = margin

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