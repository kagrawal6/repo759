#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH --cpus-per-task=8
#SBATCH -J problem4run_dynamic
#SBATCH -o problem4run_dynamic.out -e problem4run_dynamic.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

#compile task3_dynamic.cpp
g++ task3_dynamic.cpp -Wall -O3 -std=c++17 -o task3_dynamic -fopenmp

# 2) Generate timing_data.txt
echo "t time_taken" > timing_problem4_dynamic.txt

for i in {1..8}; do
t=$i

echo "Running task 3 without scheduling with number_of_particles = 100, simulation_end_time = 100, num_threads = $t"
output=$(./task3_dynamic 800 100 "$t")

time_taken=$(echo "$output" | sed -n '1p' )
echo "$t $time_taken" >> timing_problem4_dynamic.txt

done

echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="