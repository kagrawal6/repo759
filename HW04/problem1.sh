#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH --cpus-per-task=8
#SBATCH -J problem1run
#SBATCH -o problem1run.out -e problem1run.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

echo "Compiling nbody.py ..."
python3 nbody.py

echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="