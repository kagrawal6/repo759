#!/usr/bin/env zsh

#SBATCH -p instruction

#SBATCH -c 2
#SBATCH -J FirstSlurm
#SBATCH -o FirstSlurm.out -e FirstSlurm.err
hostname
