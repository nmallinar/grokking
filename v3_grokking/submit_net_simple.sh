#!/bin/bash
#SBATCH -J nmallina-grokking          # Job name
#SBATCH -o /scratch/bbjr/mallina1/grokking_output/logs/nmallina-grokking.%j.log   # define stdout filename; %j expands to jobid; to redirect stderr elsewhere, duplicate this line with -e instead
#
#SBATCH --mail-user=nmallina@ucsd.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT # get notified via email on job failure or time limit reached
#
#SBATCH --account bbjr-delta-gpu
#SBATCH --partition gpuA40x4         # specify queue, if this doesnt submit try gpu-shared
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 32G
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --tasks-per-node 1
#SBATCH -t 4:00:00       # set maximum run time in H:M:S
#SBATCH --no-requeue     # dont automatically requeue job id node fails, usually errors need to be inspected and debugged

#source /projects/bbjr/mallina1/envs/torch_and_jax/bin/activate
source ~/envs/torch_jax2/bin/activate

#for i in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6;
for i in 0.175;
do
  python train_net_simple.py \
    --wandb_proj_name "july17_nn_random_circulant" \
    --out_dir "/scratch/bbjr/mallina1/grokking_output" \
    --operation "x+y" \
    --prime 61 \
    --training_fraction ${i} \
    --batch_size 32 \
    --agop_batch_size 4 \
    --device "cuda" \
    --epochs 10000 \
    --model "OneLayerFCN" \
    --hidden_width 1024 \
    --init_scale 1.0 \
    --act_fn "pow2" \
    --learning_rate 1e-3 \
    --weight_decay 1.0 \
    --agop_decay 0.0 \
    --momentum 0.0 \
    --group_key 'test' 
done
