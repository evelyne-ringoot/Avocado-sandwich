#!/bin/bash
#SBATCH -J SVD_TEST
#SBATCH -o SVD_TEST_%j.out
#SBATCH -e SVD_TEST_%j.err
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --exclusive
module load cuda

    echo ""
    echo " Nodelist:= " $SLURM_JOB_NODELIST
    echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
    echo " GPUs per node:= " $SLURM_JOB_GPUS
    echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
    echo " Total Ntasks:=" $SLURM_NTASKS

date
echo "starting"
~/julia/bin/julia --t 4 --project=~ Avocado/KAbasedSVD/benchmark/runbenchmark.jl $1
echo "finished"
date
