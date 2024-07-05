#!/bin/bash
#SBATCH -J SVD_TEST
#SBATCH -o SVD_TEST_%j.out
#SBATCH -e SVD_TEST_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=1T
#SBATCH --ntasks=36
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --exclusive
module load julia

    echo ""
    echo " Nodelist:= " $SLURM_JOB_NODELIST
    echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
    echo " GPUs per node:= " $SLURM_JOB_GPUS
    echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
    echo " Total Ntasks:=" $SLURM_NTASKS

date
echo "starting"
~/julia/.julia --project=~ runbenchmarks.jl
echo "finished"
date
