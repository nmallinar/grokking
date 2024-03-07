#!/bin/bash
#SBATCH -J nmallina-grokking          # Job name
#SBATCH -o logs/nmallina-grokking.%j.log   # define stdout filename; %j expands to jobid; to redirect stderr elsewhere, duplicate this line with -e instead
#
#SBATCH --mail-user=nmallina@ucsd.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT # get notified via email on job failure or time limit reached
#
#SBATCH --account bbjr-delta-gpu
#SBATCH --partition gpuA100x4         # specify queue, if this doesnt submit try gpu-shared
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 8G
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --tasks-per-node 1
#SBATCH -t 4:00:00       # set maximum run time in H:M:S
#SBATCH --no-requeue     # dont automatically requeue job id node fails, usually errors need to be inspected and debugged

python cli.py \
  --run "agop2" \
  --wandb_offline \
  --operation "x/y" \
  --training_fraction 0.5 \
  --prime 3 \
  --num_tokens 31 \
  --batch_size 32 \
  --device cpu \
  --model "OneLayerFCN" \
  --eval_entk -1 \
  --num_layers 1 \
  --weight_decay 1.0 \
  --agop_weight 0.0 \
  --wandb_offline \
  --wandb_proj_name "feb13-grokking" \
  --agop_subsample_n 32 \
  --learning_rate 1e-3 \
  --optimizer "adamw" \
  --init_scale 1.0 \
  --momentum 0.0 \
  --num_steps 100000 \
  --out_dir "./" \
  --kernel_bandwidth 1 \
  --skip_agop_comps
