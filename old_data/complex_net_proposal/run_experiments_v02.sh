#!/bin/bash
#SBATCH --job-name=aw
#SBATCH --nodes=1
#SBATCH --mincpus=12
#SBATCH --time=8:00:00
#SBATCH --output=complex_net_proposal/log/production-%j.qlog
#SBATCH -p bii
module load anaconda
source activate threshold_sim
~/.conda/envs/threshold_sim/bin/python block_with_connected_astroph.py
