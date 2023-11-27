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
#SBATCH -t 24:00:00       # set maximum run time in H:M:S
#SBATCH --no-requeue     # dont automatically requeue job id node fails, usually errors need to be inspected and debugged

source /projects/bbjr/mallina1/envs/torch2/bin/activate
python cli.py \
  --training_fraction 0.5 \
  --prime 31 \
  --batch_size 32 \
  --dim_model 128 \
  --device cuda \
  --model fcn \
  --fcn_hidden_width 256 \
  --eval_entk -1 \
  --num_layers 1 \
  --weight_decay 0.0 \
  --agop_weight 1.0 \
  --wandb_proj_name "nov21-grokking" \
  --agop_subsample_n -1 \
  --learning_rate 1e-3 \
  --optimizer "adamw" \
  --momentum 0.0 \
  --num_steps 100000 \
  --out_dir "/scratch/bbjr/mallina1/grokking_output"
