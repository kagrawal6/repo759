#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH --cpus-per-task=20
#SBATCH -J task1run
#SBATCH -o task1run.out -e task1run.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

echo "Compiling matmul.cpp and task1.cpp..."
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

# 2) Generate timing_data.txt
echo "t time_ms first_val last_val" > timing_data.txt

for i in {1..20}; do

t=$i

echo "Running task1 with n = 1024 and t = $t .."

output=$(./task1 1024 "$t")

# parse each line from output
time_ms=$( echo "$output" | sed -n '3p' )
first_val=$(echo "$output" | sed -n '1p' )
last_val=$( echo "$output" | sed -n '2p' )

echo "$t $time_ms $first_val $last_val" >> timing_data.txt

done

echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="