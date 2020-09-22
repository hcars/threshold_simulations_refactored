#!/bin/bash
#SBATCH --job-name=aw
#SBATCH --nodes=1
#SBATCH --mincpus=16
#SBATCH --time=16:00:00
#SBATCH --output=complex_net_proposal/log/production-%j.qlog
#SBATCH -p bii
module load anaconda
module load gurobi/9.0.1
source activate threshold_sim
export PYTHONPATH=/apps/software/vendor/gurobi/9.0.1/lib/python3.7_utf32
~/.conda/envs/threshold_sim/bin/python compute_blocking_ilp.py
