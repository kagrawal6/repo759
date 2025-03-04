#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH --cpus-per-task=20
#SBATCH -J task3_t
#SBATCH -o task3_t.out -e task3_t.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

echo "Compiling msort.cpp and task3.cpp..."
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

# Suppose n=10^6, 
# and use the best threshold from your first script
n=1000000
best_ts=256  #best val

echo "t time_ms first_val last_val" > timing_t.txt

for t in {1..20}; do
  echo "Running task3 with n=$n, t=$t, ts=$best_ts ..."

  output=$(./task3 $n $t $best_ts)

  # parse each line
  first_val=$(echo "$output" | sed -n '1p')
  last_val=$( echo "$output" | sed -n '2p')
  time_ms=$( echo "$output" | sed -n '3p')

  echo "$t $time_ms $first_val $last_val" >> timing_t.txt
done

echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="
