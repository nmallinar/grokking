from argparse import ArgumentParser

from data import ALL_OPERATIONS
from training import main

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--wandb_offline", action='store_true', default=False)
    parser.add_argument("--operation", type=str, choices=ALL_OPERATIONS.keys(), default="x/y")
    parser.add_argument("--training_fraction", type=float, default=0.5)
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--num_steps", type=int, default=1e5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default='fcn', choices={'fcn', 'transformer', 'rfm'})
    parser.add_argument("--fcn_hidden_width", type=int, default=512)
    parser.add_argument("--loss", type=str, default='mse', choices={'cross_entropy', 'mse'})
    parser.add_argument("--kernel_bandwidth", type=float, default=1.0)
    args = parser.parse_args()

    main(args)
