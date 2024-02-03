import os
import sys
from math import ceil
import torch
from tqdm import tqdm
import scipy
import random
import wandb
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from data import get_data_with_agop_loader
from model import TwoLayerFCN, OneLayerFCN
import torch.nn.functional as F
from torch import nn
from torch.func import jacrev
from rfm import main as rfm_main
from inf_ntk import ntk_fn, jax_ntk_fn, get_jax_ntk_fn

torch.manual_seed(34)
random.seed(23)
np.random.seed(234)
torch.set_default_dtype(torch.float32)

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
    wandb.run.name = f'{wandb.run.id} - {args.model} width={args.fcn_hidden_width}, act_fn={args.act_fn}, agop_weight={args.agop_weight}, wd={args.weight_decay}, init_scale={args.init_scale}'

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

    train_loader, agop_loader, val_loader, context_len, \
    train_dataset, val_dataset, og_train_feats, og_train_labels, og_val_feats, og_val_labels = \
        get_data_with_agop_loader(
            config.operation,
            config.prime,
            config.training_fraction,
            config.batch_size,
            config.agop_subsample_n,
            drop_last=False
        )
    num_tokens = config.prime + 2
    num_tokens = config.prime

    og_train_feats = F.one_hot(og_train_feats, num_tokens).float()
    og_train_feats = og_train_feats.view(og_train_feats.size(0), -1)
    og_val_feats = F.one_hot(og_val_feats, num_tokens).float()
    og_val_feats = og_val_feats.view(og_val_feats.size(0), -1)

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
        # model.fc1.requires_grad_(False)
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

        train(model, train_loader, agop_loader, optimizer, scheduler, device,
              config.num_steps, num_tokens, args.loss, config,
              embedding_layer=embedding_layer,
              agop_weight=config.agop_weight)

        with torch.no_grad():
            U, S, Vt = np.linalg.svd(model.fc1.weight.detach().T.cpu().numpy(), full_matrices=False)
            val_acc = evaluate(model, val_loader, device, epoch, num_tokens, args.loss, config, embedding_layer=embedding_layer)

            U = torch.tensor(U)
            S = torch.tensor(S)
            Vt = torch.tensor(Vt)

            # print(U.shape, S.shape, Vt.shape)
            # torch.Size([62, 62]) (62,) torch.Size([62, 1024])

            #w0 = model.get_random_matrix()
            w0 = model.init_w0
            #U2, S2, Vt2 = np.linalg.svd(w0.numpy(), full_matrices=False)
            #U2 = torch.tensor(U2)
            #S2 = torch.tensor(S2)
            #Vt2 = torch.tensor(Vt2)

            mat = U.T @ w0 @ Vt.T

            if config.act_fn == 'pow2':
                ols_feats((og_train_feats @ mat).pow(2).numpy(), og_train_labels.numpy(), (og_val_feats @ mat).pow(2).numpy(), og_val_labels.numpy(), num_tokens, epoch, return_layer='rf')
            elif config.act_fn == 'relu':
                ols_feats(F.relu(og_train_feats @ mat).numpy(), og_train_labels.numpy(), F.relu(og_val_feats @ mat).numpy(), og_val_labels.numpy(), num_tokens, epoch, return_layer='rf')
            elif config.act_fn == 'swish':
                ols_feats(F.silu(og_train_feats @ mat).numpy(), og_train_labels.numpy(), F.silu(og_val_feats @ mat).numpy(), og_val_labels.numpy(), num_tokens, epoch, return_layer='rf')
            else:
                raise Exception()

            train_feats, train_labels = extract_feats(model, train_loader, config, embedding_layer=embedding_layer, return_layer='lin1')
            val_feats, val_labels = extract_feats(model, val_loader, config, embedding_layer=embedding_layer, return_layer='lin1')
            ols_feats(train_feats, train_labels, val_feats, val_labels, num_tokens, epoch, return_layer='lin1')
            #ntk_feats(train_feats, train_labels, val_feats, val_labels, num_tokens, epoch, return_layer='lin1')

            train_feats, train_labels = extract_feats(model, train_loader, config, embedding_layer=embedding_layer, return_layer='act_fn(lin1)')
            val_feats, val_labels = extract_feats(model, val_loader, config, embedding_layer=embedding_layer, return_layer='act_fn(lin1)')
            ols_feats(train_feats, train_labels, val_feats, val_labels, num_tokens, epoch, return_layer='act_fn(lin1)')
            #ntk_feats(train_feats, train_labels, val_feats, val_labels, num_tokens, epoch, return_layer='act_fn(lin1)')

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

def calc_full_agops(model, loader, config, embedding_layer=None):
    num_tokens = config.prime

    dumb1 = torch.zeros((config.agop_subsample_n, model.hidden_width)).to(config.device)
    dumb2 = torch.zeros((config.agop_subsample_n, model.hidden_width)).to(config.device)
    dumb3 = torch.zeros((config.agop_subsample_n, model.num_tokens)).to(config.device)

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
    if config.model == 'TwoLayerFCN':
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
          device, num_steps, num_classes, loss_arg, config,
          embedding_layer=None, agop_weight=0.0):
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

        # Zero gradient buffers
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs, act=config.act_fn)
        # output.requires_grad_(True)

        acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)

        if loss_arg == 'mse':
            labels = F.one_hot(labels, num_classes).float()
        loss = criterion(output, labels)

        # Backward pass
        mse_loss = loss.clone()

        #agop_tr = 0.0
        #left_agop_tr = 0.0
        #for idx in range(len(final_agops)):
        #    agop_tr += torch.trace(final_agops[idx])
        #    left_agop_tr += torch.trace(final_left_agops[idx])

        #if agop_weight > 0:
        #    loss += agop_weight * left_agop_tr

        loss.backward()

        # Update weights
        optimizer.step()
        # scheduler.step()

        metrics = {
            "training/accuracy": acc,
            "training/loss": loss,
            "training/mse_loss": mse_loss,
            #"training/agop_tr": agop_tr,
            #"training/left_agop_tr": left_agop_tr,
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

def ntk_feats(train_feats, train_labels, val_feats, val_labels, num_classes, epoch, return_layer, feature_projection=None, proj_key=''):
    if feature_projection is not None:
        train_feats = train_feats @ feature_projection
        val_feats = val_feats @ feature_projection

    train_feats = torch.tensor(train_feats)
    val_feats = torch.tensor(val_feats)
    _, K_tr = jax_ntk_fn(train_feats, train_feats, depth=2, kernel_fn=kernel_fn)
    # [train, test]
    _, K_te = jax_ntk_fn(train_feats, val_feats, depth=2, kernel_fn=kernel_fn)

    sol = np.linalg.inv(K_tr) @ F.one_hot(torch.tensor(train_labels).long(), num_classes).numpy()

    pred_scores = K_te.T @ sol
    pred_labels = np.argmax(pred_scores, axis=1).numpy()


    mse = (pred_scores - F.one_hot(torch.tensor(val_labels).long(), num_classes)).pow(2).mean().numpy()
    count = np.sum(val_labels == pred_labels)

    if feature_projection is not None:
        log_key = f'ntk_validation/ntk_{return_layer}_proj_{proj_key}'
    else:
        log_key = f'ntk_validation/ntk_{return_layer}'

    metrics = {
        f"{log_key}_accuracy": count / len(val_labels),
        f"{log_key}_loss": mse,
        "epoch": epoch
    }
    wandb.log(metrics, commit=False)

    return count / len(val_labels)

def ols_feats(train_feats, train_labels, val_feats, val_labels, num_classes, epoch, return_layer, feature_projection=None, proj_key=''):
    if feature_projection is not None:
        train_feats = train_feats @ feature_projection
        val_feats = val_feats @ feature_projection

    #sol = np.linalg.pinv(train_feats.T @ train_feats) @ train_feats.T @ F.one_hot(torch.tensor(train_labels).long(), num_classes).numpy()
    sol = np.linalg.pinv(train_feats) @ F.one_hot(torch.tensor(train_labels).long(), num_classes).numpy()
    pred_scores = val_feats @ sol
    pred_labels = np.argmax(pred_scores, axis=1)

    mse = np.mean(np.square(pred_scores - F.one_hot(torch.tensor(val_labels).long(), num_classes).numpy()))
    count = np.sum(val_labels == pred_labels)

    if feature_projection is not None:
        log_key = f'ols_validation/ols_{return_layer}_proj_{proj_key}'
    else:
        log_key = f'ols_validation/ols_{return_layer}'

    metrics = {
        f"{log_key}_accuracy": count / len(val_labels),
        f"{log_key}_loss": mse,
        "epoch": epoch
    }
    wandb.log(metrics, commit=False)

    return count / len(val_labels)

def extract_feats(model, loader, config, embedding_layer=None, return_layer='act_fn(lin1)', to_numpy=True):
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

            hid_states = model(inputs, return_layer=return_layer, act=config.act_fn)

            final_data.append(hid_states.detach().cpu())
            final_labels.append(labels.detach().cpu())

        final_data = torch.cat(final_data, dim=0)
        final_labels = torch.cat(final_labels, dim=0)

        if to_numpy:
            return final_data.numpy(), final_labels.numpy()
        return final_data, final_labels

# def random_feats_ols(train_feats, train_labels, val_feats, val_labels, num_classes, epoch, feature_projection=None, proj_key=''):
#     if feature_projection is not None:
#         train_feats = train_feats @ feature_projection
#         val_feats = val_feats @ feature_projection
#
#     #sol = np.linalg.pinv(train_feats.T @ train_feats) @ train_feats.T @ F.one_hot(torch.tensor(train_labels).long(), num_classes).numpy()
#     sol = np.linalg.pinv(train_feats) @ F.one_hot(torch.tensor(train_labels).long(), num_classes).numpy()
#     pred_scores = val_feats @ sol
#     pred_labels = np.argmax(pred_scores, axis=1)
#
#     mse = np.mean(np.square(pred_scores - F.one_hot(torch.tensor(val_labels).long(), num_classes).numpy()))
#     count = np.sum(val_labels == pred_labels)
#
#     if feature_projection is not None:
#         log_key = f'ols_validation/ols_{return_layer}_proj_{proj_key}'
#     else:
#         log_key = f'ols_validation/ols_{return_layer}'
#
#     metrics = {
#         f"{log_key}_accuracy": count / len(val_labels),
#         f"{log_key}_loss": mse,
#         "epoch": epoch
#     }
#     wandb.log(metrics, commit=False)
#
#     return count / len(val_labels)