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

# for i in $(seq 1 1);
# for i in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.4 0.45;
for i in 0.5;
# for i in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6;
do
  python train_kernel.py \
    --wandb_proj_name "june4_p61_quadratic_agops" \
    --out_dir "./wandb" \
    --operation "x/y" \
    --prime 61 \
    --training_fraction ${i} \
    --kernel_type "quadratic" \
    --iters 51 \
    --ridge 0.0 \
    --bandwidth 2.5 \
    --ntk_depth 1 \
    --agop_sma_size 1 \
    --agop_power 0.5 \
    --jac_reg_weight 0.0 \
    --agip_rdx_weight 0.0 \
    --group_key 'x/y' \
    --save_agops
done
