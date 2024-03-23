#!/bin/bash
#SBATCH -J nmallina-grokking          # Job name
#SBATCH -o /scratch/bbjr/mallina1/grokking_output/logs/nmallina-grokking.%j.log   # define stdout filename; %j expands to jobid; to redirect stderr elsewhere, duplicate this line with -e instead
#
#SBATCH --mail-user=nmallina@ucsd.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT # get notified via email on job failure or time limit reached
#
#SBATCH --account bbjr-delta-gpu
#SBATCH --partition gpuA100x4         # specify queue, if this doesnt submit try gpu-shared
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 16G
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --tasks-per-node 1
#SBATCH -t 10:00:00       # set maximum run time in H:M:S
#SBATCH --no-requeue     # dont automatically requeue job id node fails, usually errors need to be inspected and debugged

#source /projects/bbjr/mallina1/envs/torch2/bin/activate
source /projects/bbjr/mallina1/envs/torch_and_jax/bin/activate
python cli.py \
  --run "agop2" \
  --training_fraction 0.5 \
  --prime 31 \
  --num_tokens 31 \
  --batch_size 32 \
  --dim_model 128 \
  --device cuda \
  --model OneLayerFCN \
  --fcn_hidden_width 256 \
  --eval_entk -1 \
  --num_layers 1 \
  --weight_decay 0.0 \
  --agop_weight 10.0 \
  --wandb_proj_name "mar18-grokking" \
  --agop_subsample_n 32 \
  --learning_rate 0.001 \
  --optimizer "adamw" \
  --momentum 0.0 \
  --num_steps 10000 \
  --out_dir "/scratch/bbjr/mallina1/grokking_output" \
  --act_fn "relu" \
  --init_scale 0.0001 \
  --wandb_offline
