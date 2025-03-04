#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH --cpus-per-task=20
#SBATCH -J task3_ts
#SBATCH -o task3_ts.out -e task3_ts.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

echo "Compiling msort.cpp and task3.cpp..."
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

# Example: let's do n=10^6, t=8
n=1000000
t=8

echo "t ts time_ms first_val last_val" > timing_ts.txt

# Loop through powers of 2 from 2^2 (4) up to 2^10 (1024)
for i in {1..10}; do
  ts=$((2**i))

  echo "Running task3 with n=$n, t=$t, ts=$ts ..."

  output=$(./task3 $n $t $ts)

  # Parse each line from output
  first_val=$(echo "$output" | sed -n '1p')
  last_val=$( echo "$output" | sed -n '2p')
  time_ms=$( echo "$output" | sed -n '3p')

  echo "$t $ts $time_ms $first_val $last_val" >> timing_ts.txt
done

echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="
