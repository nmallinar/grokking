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

# POWS=(0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6)
# BW=(3.0 2.5 2.0 1.5 1.0)
# RIDGE=(0.0 1e-3 1e-5)
# EMA=(1.0 0.7 0.5)

POWS=(0.6)
BW=(2.5)
RIDGE=(0.0)
EMA=(0.2)

for ridge in "${RIDGE[@]}"
do
  for bw in "${BW[@]}"
  do
    for pow in "${POWS[@]}"
    do
      for ema in "${EMA[@]}"
      do
        for i in $(seq 1 1);
        do
          python train_kernel2.py \
            --wandb_proj_name "apr4_test4" \
            --out_dir "./" \
            --operation "x+y" \
            --prime 31 \
            --training_fraction 0.5 \
            --kernel_type "gaussian" \
            --iters 500 \
            --ridge ${ridge} \
            --bandwidth 1 \
            --jac_reg_weight 1e-5 \
            --agip_rdx_weight 1e-5 \
            --agop_sma_size 1 \
            --ema_alpha ${ema} \
            --use_ema \
            --agop_power ${pow} \
            --agip_power 1.0 \
            --wandb_offline
        done
      done
    done
  done
done
