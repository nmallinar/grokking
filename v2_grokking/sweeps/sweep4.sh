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
#SBATCH --cpus-per-task 16
#SBATCH --mem 32G
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --tasks-per-node 1
#SBATCH -t 08:00:00       # set maximum run time in H:M:S
#SBATCH --no-requeue     # dont automatically requeue job id node fails, usually errors need to be inspected and debugged

source /projects/bbjr/mallina1/envs/torch_and_jax/bin/activate
cd ..

RIDGES=(1e-3)
BANDWIDTHS=(10 5 1 5e-1 1e-1 5e-2 1e-2 5e-3 1e-3)
JACS=(0.0 1.0 1e-1 1e-2 1e-3 1e-4 1e-5)
AGIPS=(0.0 1.0 1e-1 1e-2 1e-3 1e-4 1e-5)
AVGSIZE=(10 2 1)

for ridge in "${RIDGES[@]}"
do
  for bw in "${BANDWIDTHS[@]}"
  do
    for jac in "${JACS[@]}"
    do
      for agip in "${AGIPS[@]}"
      do
        for avg in "${AVGSIZE[@]}"
        do
          python train_kernel.py \
            --wandb_proj_name "mar24-grokking" \
            --out_dir "/scratch/bbjr/mallina1/grokking_output" \
            --operation "x+y" \
            --prime 31 \
            --training_fraction 0.5 \
            --iters 1000 \
            --ridge ${ridge} \
            --bandwidth ${bw} \
            --jac_reg_weight ${jac} \
            --agip_rdx_weight ${agip} \
            --agop_avg_size ${avg}
        done
      done
    done
  done
done
