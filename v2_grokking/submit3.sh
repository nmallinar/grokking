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

for i in $(seq 1 1);
do
  python train_kernel4.py \
    --wandb_proj_name "may7_test2" \
    --out_dir "./wandb" \
    --operation "x+y" \
    --prime 19 \
    --training_fraction 0.5 \
    --kernel_type "jax_fcn_ntk" \
    --iters 500 \
    --ridge 1e-4 \
    --bandwidth 2.5 \
    --ntk_depth 1 \
    --jac_reg_weight 0.0 \
    --agip_rdx_weight 0.0 \
    --ema_alpha 0.5 \
    --agop_sma_size 1 \
    --agop_power 0.5 \
    --agip_power 1.0 \
    --group_key 'baseline'
    # --kernel_type "fcn_relu_ntk" \
    # --save_agops
done
