#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH --cpus-per-task=8
#SBATCH -J problem2run
#SBATCH -o problem2run.out -e problem2run.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

g++ task2.cpp -Wall -O3 -std=c++17 -o task2

#./task2 number_of_particles simulation_end_time
./task2 100 10


#check if C++ graph is equivalent to python
python3 plot_positions.py



echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="