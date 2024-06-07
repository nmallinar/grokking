import os
import sys
import argparse
import wandb
import torch
import torch.nn.functional as F
import numpy as np
import scipy
import random
from tqdm import tqdm
import umap

from data import operation_mod_p_data, make_data_splits, make_dataloader, held_out_op_mod_p_data
from models import neural_nets
import utils
import agop_utils

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
#torch.manual_seed(3143)
#random.seed(253)
#np.random.seed(1145)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', default='belkinlab')
    parser.add_argument('--wandb_proj_name', default='mar22-rfm-grokking')
    parser.add_argument('--wandb_offline', default=False, action='store_true')
    parser.add_argument('--group_key', default='', type=str)
    parser.add_argument('--out_dir', default='./')
    parser.add_argument('--operation', '-op', default="x+y")
    parser.add_argument('--prime', '-p', default=97, type=int)
    parser.add_argument('--training_fraction', default=0.5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--agop_batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--device', default='cuda', choices={'cuda', 'cpu'})

    parser.add_argument('--model', default='OneLayerFCN')
    parser.add_argument('--hidden_width', default=256, type=int)
    parser.add_argument('--init_scale', default=1.0, type=float)
    parser.add_argument("--act_fn", type=str, default="relu")

    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--momentum', default=0.0, type=float)

    parser.add_argument('--viz_umap', default=False, action='store_true')
    args = parser.parse_args()

    mode = 'online'
    if args.wandb_offline:
        mode = 'offline'

    wandb.init(entity=args.wandb_entity, project=args.wandb_proj_name, mode=mode, config=args,
               dir=args.out_dir)

    out_dir = os.path.join(args.out_dir, args.wandb_proj_name, wandb.run.id)
    os.makedirs(out_dir, exist_ok=True)

    wandb.run.name = f'{wandb.run.id} - p: {args.prime}, train_frac: {args.training_fraction}'

    all_inputs, all_labels = operation_mod_p_data(args.operation, args.prime)
    X_tr, y_tr, X_te, y_te = make_data_splits(all_inputs, all_labels, args.training_fraction)

    #X_tr, y_tr, X_te, y_te = held_out_op_mod_p_data(args.operation, args.prime)

    X_tr = F.one_hot(X_tr, args.prime).view(-1, 2*args.prime).double()
    #y_tr_onehot = y_tr.double()
    y_tr_onehot = F.one_hot(y_tr, args.prime).double()
    X_te = F.one_hot(X_te, args.prime).view(-1, 2*args.prime).double()
    #y_te_onehot = y_te.double()
    y_te_onehot = F.one_hot(y_te, args.prime).double()

    train_loader = make_dataloader(X_tr, y_tr_onehot, args.batch_size, shuffle=True, drop_last=False)
    agop_loader = make_dataloader(X_tr.clone(), y_tr_onehot.clone(), args.agop_batch_size, shuffle=False, drop_last=True)
    test_loader = make_dataloader(X_te, y_te_onehot, args.batch_size, shuffle=False, drop_last=False)

    model = neural_nets.OneLayerFCN(
        num_tokens=args.prime,
        hidden_width=args.hidden_width,
        context_len=2,
        init_scale=args.init_scale,
        n_classes=args.prime
    ).to(args.device)


    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=args.learning_rate,
    #     weight_decay=args.weight_decay,
    #     momentum=args.momentum
    # )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay
    )
    criterion = torch.nn.MSELoss()

    global_step = 0
    for epoch in tqdm(range(args.epochs)):

        model.train()
        for idx, batch in enumerate(train_loader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs, labels = batch

            optimizer.zero_grad()
            output = model(inputs, act=args.act_fn)

            count = (output.argmax(-1) == labels.argmax(-1)).sum()
            acc = count / output.shape[0]

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            wandb.log({
                'training/accuracy': acc,
                'training/loss': loss,
                'epoch': epoch
            }, step=global_step)

            global_step += 1

        model.eval()
        with torch.no_grad():
            count = 0
            total_loss = 0
            total = 0
            for idx, batch in enumerate(test_loader):
                batch = tuple(t.to(args.device) for t in batch)
                inputs, labels = batch

                output = model(inputs, act=args.act_fn)

                count += (output.argmax(-1) == labels.argmax(-1)).sum()
                total += output.shape[0]
                loss = criterion(output, labels)
                total_loss += loss * output.shape[0]

            total_loss /= total
            acc = count / total

            # if acc == 1.0:
                # nfm = model.fc1.weight.data.T @ model.fc1.weight.data
                # np.save('nfm.npy', nfm.detach().cpu().numpy())

            #print(f'Epoch {epoch}:\tacc: {acc}\tloss: {loss}')
            wandb.log({
                'validation/accuracy': acc,
                'validation/loss': total_loss,
                'epoch': epoch
            }, step=global_step)

            if args.viz_umap and epoch % 200 == 0:
                mapper = umap.UMAP(n_neighbors=15, min_dist=0.1,
                                   metric='euclidean', n_components=2)
                cmap = utils.generate_colormap(args.prime)

                w1 = model.fc1.weight.data.detach().cpu().numpy()
                os.makedirs('outdir', exist_ok=True)
                np.save(os.path.join('outdir', f'ep_{epoch}_w1.npy'), w1)

                embeddings = mapper.fit_transform(w1)
                utils.scatter_umap_embeddings(embeddings, None, wandb, 'UMAP, w1: (h, d)', 'umap/w1_hxd', global_step)
                np.save(os.path.join('outdir', f'ep_{epoch}_embs.npy'), embeddings)

                embeddings = mapper.fit_transform(w1.T)
                utils.scatter_umap_embeddings(embeddings, None, wandb, 'UMAP, w1: (d, h)', 'umap/w1_dxh', global_step)

                U, S, Vh = np.linalg.svd(w1, full_matrices=False)
                embeddings = mapper.fit_transform(U)
                utils.scatter_umap_embeddings(embeddings, None, wandb, 'UMAP, left sing vecs U', 'umap/U', global_step)
                embeddings = mapper.fit_transform(U.T)
                utils.scatter_umap_embeddings(embeddings, None, wandb, 'UMAP, left sing vecs U.T', 'umap/U.T', global_step)

        if epoch % 100 == 0:
            # agops, _, _, _, per_class_agops = agop_utils.calc_full_agops_per_class(model, agop_loader, args)
            # utils.display_all_agops(agops, per_class_agops, wandb, global_step)

            agop, per_class_agops = agop_utils.calc_full_agop(model, agop_loader, args)
            #print(agop)
            utils.display_all_agops([agop], per_class_agops, wandb, global_step)

            # agops, _, _, _ = agop_utils._calc_full_agops(model, agop_loader, args)
            # utils.display_all_agops(agops, [], wandb, global_step)

            #agops2 = agop_utils.calc_full_agops_exact(model, agop_loader, args)
            #print(agops2)
            #utils.display_all_agops([agops2], [], wandb, global_step, prefix='exact_')

            nfm = model.fc1.weight.data.T @ model.fc1.weight.data
            nfm = nfm.detach().cpu().numpy()

            sqrt_agop = np.real(scipy.linalg.sqrtm(agop.numpy()))
            nfa_corr = np.corrcoef(sqrt_agop.flatten(), nfm.flatten())
            nfa_no_diag_corr = np.corrcoef((sqrt_agop - np.diag(np.diag(sqrt_agop))).flatten(), (nfm - np.diag(np.diag(nfm))).flatten())
            wandb.log({
                'nfa/nfa_corr': nfa_corr[0][1],
                'nfa/nfa_no_diag_corr': nfa_no_diag_corr[0][1]
            })

            plt.clf()
            plt.imshow(nfm)
            plt.colorbar()
            img = wandb.Image(
                plt,
                caption='NFM'
            )
            wandb.log({'NFM': img}, step=global_step)
            np.save('nfm.npy', nfm)

            plt.clf()
            plt.imshow(nfm - np.diag(np.diag(nfm)))
            plt.colorbar()
            img = wandb.Image(
                plt,
                caption='NFM_no_diag'
            )
            wandb.log({'NFM_no_diag': img}, step=global_step)

if __name__=='__main__':
    main()
