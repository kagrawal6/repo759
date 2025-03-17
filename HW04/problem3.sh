#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH --cpus-per-task=8
#SBATCH -J problem3run
#SBATCH -o problem3run.out -e problem3run.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

#compile task3.cpp
g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

# 2) Generate timing_data.txt
echo "t time_taken" > timing_problem3.txt

for i in {1..8}; do
t=$i

echo "Running task 3 without scheduling with number_of_particles = 100, simulation_end_time = 100, num_threads = $t"
output=$(./task3 800 100 "$t")

time_taken=$(echo "$output" | sed -n '1p' )
echo "$t $time_taken" >> timing_problem3.txt

done

echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="