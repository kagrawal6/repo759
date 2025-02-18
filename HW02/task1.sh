#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:10:00
#SBATCH -J task1run
#SBATCH -o task1run.out -e task1run.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

# 1) Compile
echo "Compiling scan.cpp and task1.cpp..."
g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1

# 2) Generate timing_data.txt
echo "n time_ms first_val last_val" > timing_data.txt

for i in {10..30}; do
  n=$((2**i))  # compute 2^i
  echo "Running task1 with n = 2^$i = $n ..."
  
  # capture the program output (three lines)
  output=$(./task1 "$n")

  # parse each line from output
  time_ms=$( echo "$output" | sed -n '1p' )
  first_val=$(echo "$output" | sed -n '2p' )
  last_val=$( echo "$output" | sed -n '3p' )

  # append a single line to timing_data.txt
  echo "$n $time_ms $first_val $last_val" >> timing_data.txt
done

echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="
