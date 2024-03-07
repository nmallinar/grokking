from argparse import ArgumentParser

from data import ALL_OPERATIONS
# from training import main
from training_agop_reg import main as main1
from training_agop_reg2 import main as main2
from training_rf import main as main3

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run", type=str, default="agop2", choices={'agop1', 'agop2', 'random_feats'})
    parser.add_argument("--wandb_offline", action='store_true', default=False)
    parser.add_argument("--wandb_proj_name", type=str, default="neil-grokking-test")
    parser.add_argument("--operation", type=str, choices=ALL_OPERATIONS.keys(), default="x/y")
    parser.add_argument("--training_fraction", type=float, default=0.5)
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--num_tokens", type=int, default=97)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--optimizer", default="sgd", choices={'sgd', 'adamw'})
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--agop_weight", type=float, default=1e-4)
    parser.add_argument("--agop_subsample_n", type=int, default=-1)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--num_steps", type=int, default=1e5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default='fcn', choices={'fcn', 'transformer', 'rfm', 'rfm_fcn', 'TwoLayerFCN', 'OneLayerFCN', 'FourLayerFCN'})
    parser.add_argument("--fcn_hidden_width", type=int, default=512)
    parser.add_argument("--loss", type=str, default='mse', choices={'cross_entropy', 'mse'})
    parser.add_argument("--kernel_bandwidth", type=float, default=1.0)
    parser.add_argument("--eval_entk", type=int, default=-1)
    parser.add_argument("--out_dir", type=str, default="/scratch/bbjr/mallina1/grokking_output")
    parser.add_argument("--act_fn", type=str, default="relu")
    parser.add_argument("--init_scale", type=float, default=1.0)
    parser.add_argument("--skip_agop_comps", action='store_true', default=False)
    args = parser.parse_args()

    if args.run == 'agop1':
        main1(args)
    elif args.run == 'agop2':
        main2(args)
    elif args.run == 'random_feats':
        main3(args)
