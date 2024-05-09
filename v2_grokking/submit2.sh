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
#SBATCH -t 1:00:00       # set maximum run time in H:M:S
#SBATCH --no-requeue     # dont automatically requeue job id node fails, usually errors need to be inspected and debugged

#source /projects/bbjr/mallina1/envs/torch_and_jax/bin/activate

for i in $(seq 1 20);
do
  python train_kernel3.py \
    --wandb_proj_name "apr13_grok" \
    --out_dir "./" \
    --operation "x+y" \
    --prime 37 \
    --training_fraction 0.5 \
    --kernel_type "gaussian" \
    --iters 500 \
    --ridge 0.0 \
    --bandwidth 2.5 \
    --jac_reg_weight 0.0 \
    --agip_rdx_weight 0.0 \
    --agop_sma_size 1 \
    --ema_alpha 1.0 \
    --agop_power 0.5 \
    --agip_power 0.5 \
    --group_key "p37_bw2.5_r0_no_avg_doubly_centered_max_normalized"
done
